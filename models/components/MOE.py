import random
import threading
import typing
from collections.abc import Callable
from collections import defaultdict
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple, List

import torch
import copy

from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from models.components.Attention import ScaledDotProductAttention

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


MOE_TOP_K = 2
Constant = 2


class CopyExpert(torch.nn.Module):
    def __init__(self, expert):
        super(CopyExpert, self).__init__()
        pass

    def forward(self, inputs):
        return inputs


class ZeroExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ZeroExpert, self).__init__()
        pass

    def forward(self, inputs):
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)


class ConstantExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(
            torch.empty((expert.hidden_size)))
        torch.nn.init.normal_(self.constant)

        self.wg = torch.nn.Linear(expert.hidden_size, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        # print(inputs.size())
        weight = self.wg(inputs)
        weight = self.softmax(weight)
        return torch.einsum('b,bd->bd', [weight[:, 0].type_as(inputs), inputs]) + torch.einsum(
                'b,d->bd', [weight[:, 1].type_as(inputs), self.constant.type_as(inputs)])


def gating(logits: Tensor, moe_use_mixtral_gating=False, moe_use_logits_norm=False, moe_gate_norm_std=1.0) -> Dict[int, List[Tuple[int, float]]]:
    # gates shape [num_tokens, num_experts]

    num_experts = logits.size(1)
    if moe_use_mixtral_gating:
        if moe_use_logits_norm:
            target_std = moe_gate_norm_std
            logits_std = logits.std(dim=1, keepdim=True)
            logits = logits / (logits_std / target_std)
        gates, indices = torch.topk(logits, k=MOE_TOP_K, dim=1)
        gates = F.softmax(gates, dim=1)
    else:
        target_std = moe_gate_norm_std
        if moe_use_logits_norm:
            logits_std = logits.std(dim=1, keepdim=True)
            gates = F.softmax(logits / (logits_std / target_std), dim=1)
        else:
            gates = F.softmax(logits, dim=1)
        # gates shape [num_tokens, MOE_TOP_K]
        # indices shape [num_tokens, MOE_TOP_K]
        gates, indices = torch.topk(gates, k=MOE_TOP_K, dim=1)
        gates = torch.where(indices==(num_experts-1), torch.zeros_like(gates).to(gates.dtype).to(gates.device), gates)
        gates = gates / torch.sum(gates, dim=1, keepdim=True)

    expert_info = defaultdict(list)
    for expert_id in range(num_experts):
        token_ids, score_ids = torch.nonzero(indices == expert_id, as_tuple=True)
        expert_info[expert_id] = [token_ids, gates[token_ids, score_ids]]

    return expert_info


class Router(Module):
    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 moe_use_mixtral_gating: bool,
                 moe_2layer_gate: bool,
                 moe_use_logits_norm: bool,
                 moe_gate_norm_std: float,
                 ) -> None:
        super().__init__()

        if moe_2layer_gate:
            self.wg = torch.nn.Sequential(
                torch.nn.Linear(model_dim, num_experts * 8, bias=False).float(),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts * 8, num_experts, bias=False).float()).float()
        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()

        self.gate_map = torch.nn.Linear(num_experts, num_experts, bias=False)
        self.dropout = torch.nn.Dropout(0.15)
        self.gate = gating
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std

    def forward(self, input: torch.Tensor, gate_residual=True) -> Dict[int, List[Tuple[int, float]]]:
        if isinstance(self.wg, torch.nn.Linear):
            if self.wg.weight.dtype != torch.float32:
                self.wg = self.wg.float()
                setattr(self.wg.weight, 'router', True)
        else:
            if self.wg[0].weight.dtype != torch.float32:
                self.wg = self.wg.float()
                setattr(self.wg[0].weight, "router", True)
                setattr(self.wg[2].weight, "router", True)
        input_fp32 = input.float()
        # ERROR
        logits = self.wg(input_fp32)

        if gate_residual:
            def shift(x):
                zero_pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device)  # shape: [64, 1, 8]
                x_shifted = torch.cat([zero_pad, x[:, :-1, :]], dim=1)  # shape: [64, 29, 8]
                return x_shifted
            logits = logits + self.gate_map(shift(logits).to(self.gate_map.weight.dtype))

        logits = logits.reshape(-1, logits.shape[-1])
        self.dropout(logits)
        gate_output = self.gate(logits, self.moe_use_mixtral_gating, self.moe_use_logits_norm, self.moe_gate_norm_std)

        return gate_output, logits

class Expert(torch.nn.Module):
    def __init__(self, model, dim_size=512):
        super().__init__()
        self.hidden_size=dim_size
        self.model = model

    def forward(self, x):
        return self.model(x)

class FFNExpert(Expert):
    def __init__(self, dim_size=128, ffn_times = 2, dropout=0.25):
        class SkipFFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ffn = torch.nn.Sequential(
                    torch.nn.LayerNorm(dim_size),
                    torch.nn.Linear(dim_size, ffn_times * dim_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(ffn_times * dim_size, dim_size),
                    torch.nn.Dropout(p=dropout)
                )
            def forward(self, x):
                return x + self.ffn(x)
        super().__init__(dim_size=dim_size, model=SkipFFN())


class RefExpert(torch.nn.Module):
    def __init__(self, dim_size=512,
                 middle_times=2,
                 ref_size=512,
                 ref_data=None,
                 dropout=0.5,
                 complex_attention=False,
                 num_attn_heads=8,
                 ref_encoder=False,
                 ffn=False,
                 input_encoder=False,
                 skip_link=True):
        super().__init__()
        self.hidden_size=dim_size
        if ref_data is not None:
            self.ref = ref_data

        self.ref_trans = False
        if ref_size != dim_size:
            self.ref_prj = torch.nn.Linear(ref_size, dim_size)
            self.ref_trans = True

        self.has_encoder=ref_encoder
        if ref_encoder:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(dim_size, dim_size),
                torch.nn.LayerNorm(dim_size),
                torch.nn.Dropout(p=dropout)
            )

        self.has_input_encoder = input_encoder
        if input_encoder:
            self.input_encoder = torch.nn.Sequential(
                torch.nn.Linear(dim_size, dim_size),
                torch.nn.LayerNorm(dim_size),
                torch.nn.Dropout(p=dropout)
            )

        self.complex_attention = complex_attention
        if complex_attention:
            self.attention = torch.nn.MultiheadAttention(dim_size, num_attn_heads)
        else:
            def keyword_attention_fusion(query, keywords):
                # keywords: (B, N, D)   N=25
                # query: (B, D)
                scores = torch.matmul(keywords, query.unsqueeze(-1)).squeeze(-1)  # (B, N)
                attn_weights = F.softmax(scores, dim=-1)  # (B, N)
                fused = torch.sum(attn_weights.unsqueeze(-1) * keywords, dim=1)  # (B, D)
                return fused
            self.attention = keyword_attention_fusion
        self.skip = skip_link
        self.has_ffn = ffn
        if ffn:
            self.ffn = torch.nn.Sequential(
                torch.nn.LayerNorm(dim_size),
                torch.nn.Linear(dim_size, middle_times * dim_size),
                torch.nn.ReLU(),
                torch.nn.Linear(middle_times * dim_size, dim_size),
                torch.nn.Dropout(p=dropout)
            )

        self.norm_query = torch.nn.LayerNorm(dim_size)
        self.norm_keywords = torch.nn.LayerNorm(dim_size)

    def fusion(self, query, keywords):
        if self.has_encoder:
            keywords = self.encoder(keywords)

        norm_query = self.norm_query(query).unsqueeze(0)
        norm_key = self.norm_keywords(keywords)

        if self.complex_attention:
            norm_key = norm_key.permute(1, 0, 2)
            temp= self.attention(norm_query, norm_key, norm_key)[0].squeeze()
        else:
            temp = self.attention(norm_query.squeeze(0), norm_key)
        if self.skip:
            temp = query + temp
        if self.has_ffn:
            temp = self.ffn(temp) + temp

        return temp

    def load_ref(self, ref, seq_len):
        if self.ref_trans:
            ref = self.ref_prj(ref)
        self.ref = ref.unsqueeze(1).repeat(1, seq_len, 1, 1).reshape(-1, ref.shape[-2], ref.shape[-1])


    def forward(self, query, index):
        assert self.ref is not None, "RefExpert Please Load Ref"
        if self.has_input_encoder:
            query = self.input_encoder(query)
        return self.fusion(query, self.ref[index])

class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts):
        super(Experts, self).__init__()
        if isinstance(expert, list):
            expert = torch.nn.ModuleList(expert)
        if isinstance(expert, torch.nn.ModuleList):
            assert len(expert) + 2 <= num_local_experts, "Experts_Num_Error"
            expert.extend(torch.nn.ModuleList(
            [ConstantExpert(expert[0]) for _ in range(num_local_experts - len(expert) - 2)] +
            [CopyExpert(expert[0]), ZeroExpert(expert[0])])
            )
            self.experts = expert
        else:
            self.experts = torch.nn.ModuleList(
                [copy.deepcopy(expert) for _ in range(num_local_experts - 2 - Constant)] +
                [ConstantExpert(expert) for _ in range(Constant)] +
                [CopyExpert(expert), ZeroExpert(expert)])


    def forward(self, inputs):
        raise NotImplementedError



class MOELayer(Base):
    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_size,
                 num_local_experts: int,
                 moe_use_mixtral_gating: bool,
                 moe_feature_no_mul_topk: bool) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

    def forward(self, input: Tensor, gate_residual=True) -> Tensor:
        d_model = input.shape[-1]
        reshaped_input = input.reshape(-1, d_model)
        output = torch.zeros_like(reshaped_input, requires_grad=True)
        # Error
        expert_info, gate_residual = self.gate(input, gate_residual)
        if not (self.moe_use_mixtral_gating or self.moe_feature_no_mul_topk):
            reshaped_input = reshaped_input * MOE_TOP_K
        for expert, token_indices_and_gates in expert_info.items():
            indices, gating = token_indices_and_gates
            gating = gating.unsqueeze(-1)
            tokens = reshaped_input.index_select(dim=0, index=indices)
            if isinstance(self.experts.experts[expert], RefExpert):
                expert_output = self.experts.experts[expert](tokens, indices)
            else:
                expert_output = self.experts.experts[expert](tokens)
            expert_output =  expert_output * gating
            output = output.index_add(dim=0, index=indices, source=expert_output.to(output.dtype))
        output = output.reshape(input.shape)
        gate_residual = gate_residual.reshape(input.shape[0], input.shape[1], gate_residual.shape[-1])
        return output, gate_residual


class MOE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=8,
                 ep_size=1,
                 moe_use_mixtral_gating=False,
                 moe_2layer_gate=True,
                 moe_use_logits_norm=False,
                 moe_gate_norm_std=1.0,
                 moe_feature_no_mul_topk=False):
        super(MOE, self).__init__()

        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_2layer_gate = moe_2layer_gate
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

        experts = Experts(expert, self.num_local_experts)
        self.moe = MOELayer(Router(hidden_size,
                                   num_experts,
                                   self.moe_use_mixtral_gating,
                                   self.moe_2layer_gate,
                                   self.moe_use_logits_norm,
                                   self.moe_gate_norm_std),
                            experts,
                            self.ep_size,
                            self.num_local_experts,
                            self.moe_use_mixtral_gating,
                            self.moe_feature_no_mul_topk,
                            )

    def forward(self, hidden_states, gate_residual=True):
        output, gate_residual = self.moe(hidden_states, gate_residual=gate_residual)
        return output, gate_residual

if __name__ == '__main__':
    # Mock Expert Class for Testing
    class MockExpert(torch.nn.Module):
        def __init__(self, hidden_size):
            super(MockExpert, self).__init__()
            self.hidden_size = hidden_size
            self.linear = torch.nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            return self.linear(x)


    # Test the MOE Class
    def test_moe():
        # Parameters
        hidden_size = 512
        num_experts = 8
        num_words = 25
        ep_size = 1
        batch_size = 64
        sequence_length = 29


        # Create a mock expert instance
        mock_expert = MockExpert(hidden_size)
        ref_experts = [RefExpert(complex_attention=True) for _ in range(2)]
        # Initialize the MOE model
        moe_model = MOE(hidden_size,
                        expert=ref_experts,
                        num_experts=num_experts,
                        ep_size=ep_size,
                        moe_use_mixtral_gating=False,
                        moe_2layer_gate=True,
                        moe_use_logits_norm=False,
                        moe_gate_norm_std=1.0,
                        moe_feature_no_mul_topk=False)

        # Create random input data
        ref_1 = torch.randn(batch_size, num_words, hidden_size)
        ref_2 = torch.randn(batch_size, num_words, hidden_size)
        for i, j in zip(ref_experts, [ref_1, ref_2]):
            i.load_ref(j, sequence_length)
        hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

        # Forward pass
        with torch.autograd.set_detect_anomaly(True):
            output, gate_residual = moe_model(hidden_states)
            output.sum().backward()

        # Print the output shape
        print(f"Output shape: {output.shape}")
        print(f"Gate residual shape: {gate_residual.shape if gate_residual is not None else None}")


        # Run the test
    test_moe()