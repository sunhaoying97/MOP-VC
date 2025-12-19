''' Define sublayers in the encoder/decoder layer of Transformer'''
import json
import pickle
import random
import threading
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .Attention import ScaledDotProductAttention, CompositionalSDPA
from .MOE import MOE, RefExpert, FFNExpert
from .activations import get_activation
from .basic import CompositionalLinear


class MultiHeadAttention(nn.Module):
    def __init__(self, 
            dim_hidden: int,             
            hidden_dropout_prob: float = 0.5, 
            has_ln: bool = True,
            pre_ln: bool = False,
            layer_norm_eps: float = 1e-12,
            skip_connection: bool = True,
            attention_class = ScaledDotProductAttention,
            **kwargs,
        ):
        super(MultiHeadAttention, self).__init__()
        self.SDPA = attention_class(dim_hidden=dim_hidden, **kwargs)
        
        self.dim_hidden = dim_hidden
        self.define_dense()    
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps) if has_ln else None
        
        self.pre_ln = pre_ln
        self.skip_connection = skip_connection
    
    def define_dense(self):
        self.dense = nn.Linear(self.dim_hidden, self.dim_hidden)

    def forward_dense(self, hidden_states, **kwargs):
        context = self.dense(hidden_states)
        return context
        
    def forward(self, 
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            input_tensor: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            q: Optional[torch.Tensor] = None,
            k: Optional[torch.Tensor] = None,
            v: Optional[torch.Tensor] = None,
            early_return: bool = False,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_tensor is None:
            input_tensor = hidden_states.clone()

        if self.pre_ln and self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        
        if q is not None:
            assert k is not None
            assert v is not None
        else:
            if encoder_hidden_states is None:
                q = k = v = hidden_states
            else:
                q = hidden_states
                k = v = encoder_hidden_states

        hidden_states, attention_probs = self.SDPA(q, k, v, attention_mask, head_mask, **kwargs)
        context = self.forward_dense(hidden_states, **kwargs)
        context = self.dropout(context)

        if early_return:
            return context, attention_probs

        if self.skip_connection:
            hidden_states = context + input_tensor
        
        if not self.pre_ln and self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states, attention_probs, context


class GatedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, dim_hidden, scalar_gate=False, **kwargs):
        super().__init__(dim_hidden, **kwargs)
        self.gate = nn.Sequential(
            nn.Linear(dim_hidden * 2, 1 if scalar_gate else dim_hidden),
            nn.Sigmoid()
        )
    
    def forward(self, 
            hidden_states: torch.Tensor,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context, attention_probs = super().forward(hidden_states, early_return=True, **kwargs)
        gate_probs = self.gate(torch.cat([hidden_states, context], dim=-1))

        hidden_states = hidden_states + gate_probs * context
            
        if not self.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states, (attention_probs, gate_probs), context
        

class PositionwiseFeedForward(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_intermediate: int, 
            hidden_act: str = 'relu',
            hidden_dropout_prob: float = 0.5, 
            layer_norm_eps: float = 1e-12,
            pre_ln: bool = False,
            **kwargs,
        ):
        super(PositionwiseFeedForward, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_intermediate = dim_intermediate
        self.define_dense()
        self.act = get_activation(hidden_act)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)
        self.pre_ln = pre_ln
    
    def define_dense(self):
        self.dense1 = nn.Linear(self.dim_hidden, self.dim_intermediate)
        self.dense2 = nn.Linear(self.dim_intermediate, self.dim_hidden)
    
    def forward_dense1(self, hidden_states, **kwargs):
        return self.dense1(hidden_states)

    def forward_dense2(self, hidden_states, **kwargs):
        return self.dense2(hidden_states)

    def forward(self, hidden_states, **kwargs):
        input_tensor = hidden_states.clone()

        if self.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.forward_dense1(hidden_states, **kwargs)
        hidden_states = self.act(hidden_states)
        hidden_states = self.forward_dense2(hidden_states, **kwargs)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor
        if not self.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states


class CompositionalMHA(MultiHeadAttention):
    def __init__(self, **kwargs):
        self.dim_factor = kwargs['dim_hidden'] // kwargs.get('dim_factor_scale', 2)
        self.dim_semantic = kwargs['dim_semantic']
        super().__init__(**kwargs, attention_class=CompositionalSDPA)
    
    def define_dense(self):
        self.dense = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_hidden)
    
    def forward_dense(self, hidden_states, preds_attr, **kwargs):
        return self.dense(hidden_states, preds_attr.detach())


class CompositionalFFN(PositionwiseFeedForward):
    def __init__(self, **kwargs):
        self.dim_factor = kwargs['dim_hidden'] // kwargs.get('dim_factor_scale', 2)
        self.dim_semantic = kwargs['dim_semantic']
        super().__init__(**kwargs)
    
    def define_dense(self):
        self.dense1 = CompositionalLinear(self.dim_intermediate, self.dim_factor, self.dim_semantic, self.dim_hidden)
        self.dense2 = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_intermediate)
    
    def forward_dense1(self, hidden_states, preds_attr, **kwargs):
        return self.dense1(hidden_states, preds_attr.detach())
    
    def forward_dense2(self, hidden_states, preds_attr, **kwargs):
        return self.dense2(hidden_states, preds_attr.detach())

def write_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text + '\n')

def maybe_sample_and_write(tensor: torch.Tensor, p: float, filename: str, kn=1):
    def to_khot(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        将 tensor 的最后一维变为 k-hot 向量（top-k 为1，其它为0）

        参数：
            x: 任意形状的 tensor，转换作用只在最后一维
            k: 要选出多少个1

        返回：
            khot_tensor: 与 x 形状相同的 tensor，最后一维是 k-hot 向量（float 类型）
        """
        # 获取 top-k 索引
        topk_indices = torch.topk(x, k=k, dim=-1).indices
        # 创建同形状的全0 tensor
        khot = torch.zeros_like(x, dtype=torch.float)
        # 在 top-k 的位置填 1
        khot.scatter_(-1, topk_indices, 1.0)
        return khot
    if random.random() < p:
        tensor = to_khot(tensor, k=kn)
        vector = tensor.sum(dim=(0, 1))

        # 启动线程写入
        text = str([f"{x:.2f}" for x in vector])
        threading.Thread(target=write_to_file, args=(filename, text)).start()

def sample_and_write(tensor: torch.Tensor, p: float=0.1, filename: str="log.txt", num_samples: int=1):
    for _ in range(num_samples):
        maybe_sample_and_write(tensor, p, filename)

class PairMultiHeadAttention(nn.Module):
    class AdaptDict:
        def __init__(self):
            self.dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<mask>': 4, '<vis>': 5}
            self.index = 6
            self.max_size = 20480

        def get(self, key, default=1):
            if key in self.dict:
                return self.dict[key]
            else:
                if self.index >= self.max_size:
                    return default
                self.dict[key] = self.index
                self.index += 1
                return self.dict[key]

    def __init__(self, dim_hidden, **kwargs):
        super().__init__()
        self.gate_information = None
        dim_hidden = kwargs.get('dim_hidden', dim_hidden)
        kwargs.pop("add_hybrid_attention_bias")
        kwargs.pop("hybrid_length")

        #######################################################
        # Predict Tag Part
        #######################################################
        # self.tag_ffn = nn.Sequential(
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.ReLU(),
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.Dropout(p=0.5)
        # )
        self.tag_head = nn.Linear(512, 19)
        self.tag_pred = torch.Tensor([0])
        #######################################################



        #######################################################
        # Gate input Part
        #######################################################
        # self.feat_attention_tag = MultiHeadAttention(dim_hidden, **kwargs)
        # self.gate = nn.Sequential(nn.Linear(2 * dim_hidden, 3),
        #                           nn.LayerNorm(3),
        #                           nn.Softmax(dim=-1))

        self.prj_factor = 8
        # #################################################################################################################################
        num_expert = 6
        self.ref_expert = ([RefExpert(complex_attention=True, dim_size=dim_hidden//self.prj_factor, ref_size=dim_hidden, ffn=False) for _ in range(3)]

                           )
        # #################################################################################################################################
        self.MOE = nn.Sequential(
            nn.Linear(512, dim_hidden//self.prj_factor),
            MOE(hidden_size=dim_hidden//self.prj_factor, expert=self.ref_expert, num_experts=num_expert),
        )
        self.norm_priori = nn.LayerNorm(dim_hidden // self.prj_factor)
        self.priori_self_attention = nn.MultiheadAttention(dim_hidden//self.prj_factor, dim_hidden // 64)
        self.prior_encoder = nn.Linear(dim_hidden // self.prj_factor, dim_hidden)
        #######################################################



        #######################################################
        #  Load Part
        #######################################################
        print("Loading Information")
        self.clip_information = torch.load("/workspace/CARE/pretreatment/semantic_memory_by_pos.pt", map_location="cuda")
        # self.wtoi = self.AdaptDict()
        # with open("/home/lands54/Code/CARE/data/video_datasets/MSRVTT/info_corpus_tag.pkl", 'rb') as f:
        #     self.wtoi = {v: k for k, v in pickle.load(f)['info']['itow'].items()}
        # self.clip_similarity_n = json.load(open('/home/lands54/Code/CARE/data/video_datasets/MSRVTT/KGclip/MSRVTT/MSRVTT_similarity_n_topK.json', 'rb'))
        # self.clip_similarity_v = json.load(open('/home/lands54/Code/CARE/data/video_datasets/MSRVTT/KGclip/MSRVTT/MSRVTT_similarity_v_topK.json', 'rb'))
        # self.kg_n = json.load(open('/home/lands54/Code/CARE/data/video_datasets/MSRVTT/KGclip/MSRVTT/extracted_MSRVTT_similarity_n_topk10_head_KG.json', 'rb'))
        # self.kg_v = json.load(open('/home/lands54/Code/CARE/data/video_datasets/MSRVTT/KGclip/MSRVTT/extracted_MSRVTT_similarity_v_topk10_head_KG.json', 'rb'))
        print("Loading End")
        # #################################################################################
        #  Priori Transform Part
        # #################################################################################
        # self.priori_decider = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
        #                                     nn.ReLU(),
        #                                     nn.Linear(dim_hidden, dim_hidden),
        #                                     nn.LayerNorm(512),
        #                                     nn.Dropout(p=0.5))
        #
        # self.priori_prj = nn.Sequential(nn.LayerNorm(512),
        #                                 nn.Linear(512, 512),
        #                                 nn.LayerNorm(512),
        #                                 nn.Dropout(p=0.5))
        #################################################################################
        self.norm_hidden = nn.LayerNorm(dim_hidden)
        self.fusion_attention = nn.MultiheadAttention(dim_hidden, dim_hidden//64)


        # #################################################################################
        # VIDEO FEATS ATTENTION
        # #################################################################################
        self.norm_feats_hidden = nn.LayerNorm(dim_hidden)
        self.feat_attention = MultiHeadAttention(dim_hidden, add_hybrid_attention_bias=True, hybrid_length=84,  **kwargs)

        # #################################################################################
        # TEMP
        # #################################################################################
        self.self_attn_mask = None


    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # gate and gram prediction
        # gate, priori = self.pred_gate(kwargs['tag_hidden_state'], kwargs['feats'], hidden_states=hidden_states)
        # get relative word mean
        # clip_n, clip_v = self.get_kg(kwargs['video_ids'])
        # # gated
        # priori = self.priori_prj(gate[..., 0].unsqueeze(-1) * clip_n + gate[..., 1].unsqueeze(-1) * clip_v)
        # # log
        # if gate.shape[1] > 3 and random.randint(0, 10) == 1:
        #     async_write("log.txt", str(gate[0][3].tolist()) + ":" + str(self.tag_pred[0][3].max(-1)[1].item()))
        # #############################################################################################################
        tag_hidden = kwargs['tag_hidden_state']

        seq_len = tag_hidden.shape[1]
        self.self_attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()

        self.tag_pred = self.tag_head(tag_hidden)

        # ############################################################################################################
        #  Priori Line>MOE>SelfAttention>Line
        # ############################################################################################################
        kg = self.get_kg(kwargs['video_ids'])
        self.load_ref(kg[:3], seq_len)
        priori, gate_information = self.MOE(tag_hidden)
        self.gate_information = gate_information
        priori = priori.permute(1, 0, 2)
        normed_priori = self.norm_priori(priori)
        priori = self.priori_self_attention(normed_priori,
                                            normed_priori,
                                            normed_priori,
                                            attn_mask=self.self_attn_mask
                                            )[0] + priori
        priori = self.prior_encoder(priori)
        # ############################################################################################################
        # Log
        # ############################################################################################################
        # if gate_information.shape[1] > 9:
        #     maybe_sample_and_write(gate_information[:, :9, :], p=0.1, filename="log.txt", kn=2)
        #     maybe_sample_and_write(self.tag_pred[:, :9, :], p=0.1, filename="prob.txt", kn=1)
        # ############################################################################################################
        # Fusion Attention
        # ############################################################################################################
        normed_priori = self.norm_hidden(priori)
        hidden_states = self.fusion_attention(hidden_states.permute(1, 0, 2),
                                              normed_priori,
                                              normed_priori,
                                              attn_mask=self.self_attn_mask
                                              )[0].permute(1, 0, 2) + hidden_states

        # ############################################################################################################
        # Feat Attention
        # ############################################################################################################
        normed_hidden = self.norm_feats_hidden(hidden_states)
        hidden_states, a, b = self.feat_attention(hidden_states=normed_hidden,
                                                  encoder_hidden_states=kwargs['feats'])

        return hidden_states, torch.zeros_like(hidden_states, requires_grad=True), torch.zeros_like(hidden_states, requires_grad=True)


    def get_kg(self, video_ids):
        if isinstance(video_ids, torch.Tensor):
            video_ids = video_ids.tolist()
        video_ids  = ["video" + str(video_ids[i]) for i in range(len(video_ids))]
        information = []
        for  pos in self.clip_information[video_ids[0]].keys():
            information.append(torch.stack([self.clip_information[video_id][pos]["features"].to(dtype=torch.float32) for video_id in video_ids]).to('cuda'))

        return information

    def load_ref(self, refs, seq_len):
        for i, j in zip(self.ref_expert, refs):
            i.load_ref(j, seq_len=seq_len)

    def get_infor(self):
        return {"tag_pred": self.tag_pred, "MOE_logits": self.gate_information}

