import math
import torch
import torch.nn as nn
from config import Constants
from .base import CritBase
from ..logger import AverageMeter


class LanguageGeneration(CritBase):
    def __init__(self, opt):
        class Mo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(11000, 512)
                self.pos = torch.nn.Embedding(30, 512)
                self.line = torch.nn.Linear(512, 128)
                self.attn = torch.nn.MultiheadAttention(512, 8)
                self.norm = torch.nn.LayerNorm(512)

            def forward(self, x):
                seq_len = x.shape[-1]
                pos_list = torch.tensor([i for i in range(seq_len)]).to('cuda')
                pos_embed = self.pos(pos_list)
                embed = (self.emb(x) + pos_embed).permute(1, 0, 2)
                temp = embed
                embed = self.attn(self.norm(embed), self.norm(embed), self.norm(embed),
                                  attn_mask=torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(
                                      'cuda'))[0] + temp
                embed = self.line(embed.permute(1, 0, 2))[:, -1, :]
                return embed.squeeze()
        self.model = Mo()
        self.model.load_state_dict(torch.load("/workspace/CARE/pretreatment/grammer.pth"))
        self.model = self.model.to('cuda')
        for p in self.model.parameters():
            p.requires_grad = False

        visual_word_generation = opt.get('visual_word_generation', False)

        if visual_word_generation:
            weights = opt.get('nv_weights', [0.8, 1.0])
            self.num_word_acc = 2
        else:
            weights = 1.0
            self.num_word_acc = 1
        
        super().__init__(keys=['logits', 'labels', 'probs'], weights=weights)
        self.label_smoothing = opt["label_smoothing"]
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.ignore_index = Constants.PAD
        self.visual_word_generation = visual_word_generation
        self.opt = opt

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
        def process_tensor(tensor, seq_length=30, padding_value=2):
            """
            对输入张量的每个序列先在开头添加 padding_value，然后将第一个值为 3 后的所有元素置为 0，
            最后裁剪或填充每个序列的长度到指定长度。
            :param tensor: 输入的张量，形状为 (batch_size, seq_length)
            :param seq_length: 输出张量的目标长度
            :param padding_value: 填充的值（在序列开头填充）
            :return: 处理后的张量
            """
            # 在每个序列的开头添加填充值 padding_value
            padded_tensor = torch.cat((tensor.new_ones(tensor.shape[0], 1) * padding_value, tensor), dim=1)

            # 将每行中，第一个值为 3 后的元素置为 0
            for i in range(padded_tensor.shape[0]):
                # 找到第一个值为 3 的位置
                idx = (padded_tensor[i] == 3).nonzero()
                if len(idx) > 0:  # 确保找到了值为3的元素
                    idx = idx[0][0].item()  # 获取第一个3的索引
                    padded_tensor[i, idx+1:] = 0  # 将第一个3及其后面的所有元素置为0

            # 裁剪每个序列的长度，如果长度不足则填充到指定长度
            if padded_tensor.shape[1] < seq_length:
                # 填充到 seq_length 的长度
                padding = seq_length - padded_tensor.shape[1]
                padded_tensor = torch.cat((padded_tensor, padded_tensor.new_zeros(padded_tensor.shape[0], padding)),
                                          dim=1)
            else:
                # 裁剪到 seq_length 的长度
                padded_tensor = padded_tensor[:, :seq_length]

            return padded_tensor

        # strangeLoss = ((self.model(process_tensor(logits.max(dim=-1)[1])) - self.model(process_tensor(labels))[1])**2).sum(dim=-1).mean()
        strangeLoss = 0
        if probs is not None:
            logits = probs

        assert not len(others)
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

        if self.ignore_index is not None:
            mask = labels.ne(self.ignore_index).float()
            return torch.sum(loss * mask) + strangeLoss
        else:
            return torch.sum(loss) + strangeLoss
    
    def calculate_word_acc(self, index_indicator, preds, gts):
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
        return ['Word Acc%d' % i for i in range(self.num_word_acc)] + ['Perplexity']

    def get_info(self):
        info = [meter.avg for meter in self.word_acc_recorder]
        info += [math.exp(self.perplexity_recorder.avg)]
        return self.get_fieldsnames(), info

    def reset_recorder(self):
        self.word_acc_recorder = [AverageMeter() for _ in range(self.num_word_acc)]
        self.perplexity_recorder = AverageMeter()
