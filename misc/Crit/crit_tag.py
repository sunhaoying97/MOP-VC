import math

import numpy
import torch
import torch.nn as nn
from config import Constants
from .base import CritBase
from ..logger import AverageMeter


class TagGeneration(CritBase):
    def __init__(self, opt):
        visual_word_generation = opt.get('visual_word_generation', False)

        if visual_word_generation:
            weights = opt.get('nv_weights', [0.8, 1.0])
            self.num_word_acc = 2
        else:
            weights = 1.0
            self.num_word_acc = 1

        super().__init__(keys=['tag_pred', 'taggings', 'probs', "MOE_logits"], weights=weights)
        self.label_smoothing = opt["label_smoothing"]
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.ignore_index = Constants.PAD
        self.visual_word_generation = visual_word_generation
        self.opt = opt
        self.l = 1
        self.step = 0

    def _step(self,
              index_indicator: int,
              logits: torch.Tensor,
              labels: torch.Tensor,
              probs: torch.Tensor = None,
              *others,
              ):
        """
            args:
                logits: [batch_size, seq_len, vocab_size]
                labels: [batch_size, seq_len]
        """

        def moe_load_balance_loss(router_logits: torch.Tensor,
                                  topk_mask: torch.Tensor,
                                  seq_mask: torch.Tensor) -> torch.Tensor:
            """
            MoE router 负载均衡损失，支持 top-k gating + 屏蔽无效 token（如 EOS 后的 padding）。

            参数:
                router_logits: (batch, seq_len, num_experts) -> router 的原始打分
                topk_mask:     (batch, seq_len, num_experts) -> top-k 路由 mask，0/1
                seq_mask:      (batch, seq_len) -> token 有效性 mask，0 表示 padding/EOS 后

            返回:
                scalar 负载均衡 loss
            """
            B, T, E = router_logits.shape

            # 扩展 seq_mask -> (B, T, 1)，用于 broadcast 到 expert 维度
            seq_mask_exp = seq_mask.unsqueeze(-1)  # shape = (B, T, 1)

            # 只在 top-k 且有效 token 的位置计算 softmax
            effective_mask = topk_mask & (seq_mask_exp > 0)  # shape = (B, T, E)
            masked_logits = router_logits.masked_fill(effective_mask == 0, float('-inf'))

            router_weights = torch.softmax(masked_logits, dim=-1)
            router_weights = torch.nan_to_num(router_weights, nan=0.0)

            # 展平前两维
            flat_weights = router_weights.view(-1, E)  # (B*T, E)
            flat_effective_mask = effective_mask.view(-1, E).float()  # (B*T, E)

            # importance: 每个 expert 分到的权重和
            importance = flat_weights.sum(dim=0)  # (E,)
            # load: 每个 expert 被选中的 token 数
            load = flat_effective_mask.sum(dim=0)  # (E,)

            # normalize
            importance /= importance.sum() + 1e-9
            load /= load.sum() + 1e-9

            loss = (importance * load).sum() * E

            return loss

        def get_topk_mask(alogits: torch.Tensor, k: int) -> torch.Tensor:
            topk = torch.topk(alogits, k, dim=-1)
            threshold = topk.values[..., -1].unsqueeze(-1)
            return alogits >= threshold


        labels = labels.to('cuda')[..., 1:]
        temp = labels
        if probs is not None:
            logits = probs

        if (self.opt.get('use_attr', False)) and 'prefix' in self.opt.get('use_attr_type', ''):
            assert logits.size(1) == labels.size(1) + self.opt['use_attr_topk']
            logits = logits[:, self.opt['use_attr_topk']:, :]
        elif (self.opt.get('use_attr', False)) and 'pp' in self.opt.get('use_attr_type', ''):
            assert logits.size(1) == labels.size(1) + 1
            logits = logits[:, 1:, :]
        elif logits.size(1) == labels.size(1) + 1:
            logits = logits[:, :-1, :]
        else:
            assert logits.size(1) == labels.size(1)

        if probs is not None:
            tgt_word_logprobs = (logits + 1e-6).log()
        else:
            tgt_word_logprobs = torch.log_softmax(logits, dim=-1)

        # calculate the top-1 accuracy of the generated words
        self.calculate_word_acc(index_indicator, tgt_word_logprobs, labels)
        # calculate the perplexity of the generated words
        self.calculate_perplexity(index_indicator, tgt_word_logprobs, labels)

        tgt_word_logprobs = tgt_word_logprobs.contiguous().view(-1, tgt_word_logprobs.size(2))
        labels = labels.contiguous().view(-1)
        loss = (1 - self.label_smoothing) * self.loss_fn(tgt_word_logprobs, labels) + \
               self.label_smoothing * - tgt_word_logprobs.mean(dim=-1)

        purpose = 256 * 1
        loss = loss

        self.step += 1
        if self.ignore_index is not None:
            mask = labels.ne(self.ignore_index).float()
            return torch.sum(loss * mask)
        else:
            return torch.sum(loss)

    def calculate_word_acc(self, index_indicator, preds, gts):
        gts = gts.to('cuda')
        ind = gts.ne(Constants.PAD)
        if index_indicator == 0 and self.visual_word_generation:
            ind = ind & gts.ne(Constants.MASK)

        predict_res = preds.max(-1)[1][ind]
        target_res = gts[ind]

        self.word_acc_recorder[index_indicator].update(
            (predict_res == target_res).sum().item(),
            predict_res.size(0),
            multiply=False
        )

    def calculate_perplexity(self, index_indicator, preds, gts):
        gts = gts.to('cuda')
        # for the methods with visual word generation
        # we only compute the perplexity of the caption genration process
        if index_indicator == 0 and self.visual_word_generation:
            return None

        assert len(preds.shape) == 3
        assert preds.shape[:-1] == gts.shape

        log_probs = preds.gather(2, gts.unsqueeze(2)).squeeze(2)
        mask = gts.ne(Constants.PAD)
        num_words = float(torch.sum(mask))

        per_word_cross_entropy = -torch.sum(log_probs * mask) / num_words
        self.perplexity_recorder.update(per_word_cross_entropy.item(), num_words)

    def get_fieldsnames(self):
        return ['Tag Acc%d' % i for i in range(self.num_word_acc)] + ['TagPerplexity']

    def get_info(self):
        info = [meter.avg for meter in self.word_acc_recorder]
        info += [math.exp(self.perplexity_recorder.avg)]
        return self.get_fieldsnames(), info

    def reset_recorder(self):
        self.word_acc_recorder = [AverageMeter() for _ in range(self.num_word_acc)]
        self.perplexity_recorder = AverageMeter()
