import os
import pickle

import spacy
import torch
import torch.nn as nn
import spacy
from spacy_legacy.layers.staticvectors_v1 import forward

from config import Constants
from models.Decoder.Transformer import TransformerDecoder
from models.Predictor import Predictor_attribute
from models.components.Embeddings import Embeddings, LearnablePositionalEncoding
from models.components.SubLayers import MultiHeadAttention



class Tagger:
    def __init__(self):
        self.tagger = spacy.load("en_core_web_sm")
        with open("/home/lands54/Code/CARE/data/video_datasets/MSRVTT/info_corpus_tag.pkl", "rb") as f:
            temp = pickle.load(f)['info']
            self.itow = temp['itow']
            self.itot = temp['id2tag']
            self.ttoi = {v: k for k, v in self.itot.items()}

    def __call__(self, sentences):
        return self.forward(sentences)

    def forward(self, sentences):
        leng = sentences.size()[-1]
        temp = self.id2word(sentences)

        def process(x):
            if isinstance(x, str):
                l = x.find('>')
                r = x.find('<', l)
                r = None if r == -1 else r
                q = x[l+1:r].lstrip()
                t =  self.tagger(q)
                return ([self.ttoi['<bos>']] + [self.ttoi[i.tag_] for i in t] + ([] if r is None else [self.ttoi['<eos>']]) + [0]*29)[:leng]
            for i, n in zip(x, range(len(x))):
                x[n] = process(i)
            return x

        return process(temp)

    def id2word(self, sentences):
        def tensor_to_sentences(tensor, mapping_dict, join_char=' '):
            nested_list = tensor
            if isinstance(tensor, torch.Tensor):
                nested_list = tensor.tolist()  # 转换为嵌套列表

            def process_element(x):
                if isinstance(x, int):
                    return mapping_dict[x], True
                ismaped = False
                for i, n in zip(x, range(len(x))):
                    x[n], ismaped = process_element(i)

                return join_char.join(x) if ismaped else x, False

            process_element(nested_list)
            return nested_list

        return tensor_to_sentences(sentences, self.itow)

class Predictor_tag(Predictor_attribute):
    def __init__(self, opt):
        super().__init__(opt)
        dim_size = 512
        prj_factor = 1
        self.prj = nn.Linear(dim_size, dim_size // prj_factor)
        # self.tagger = Tagger()
        self.emb = nn.Embedding(num_embeddings=10547, embedding_dim=dim_size // prj_factor)
        self.pos_emb = LearnablePositionalEncoding(30, dim_size // prj_factor)
        self.norm = nn.LayerNorm(dim_size // prj_factor)
        self.self_attention = nn.MultiheadAttention(dim_size // prj_factor, dim_size // prj_factor // 64, batch_first=True)

    def forward(self, hidden_state=None, **kwargs):
        # embed = self.pos_emb(self.emb(kwargs['input_ids']))
        embed = self.prj(kwargs['embed'].detach())
        attention_mask = torch.triu(torch.ones((len(kwargs['input_ids'][0]), len(kwargs['input_ids'][0])), dtype=torch.bool), diagonal=1).to('cuda')
        # attention_mask = attention_mask.unsqueeze(0).expand(embed.size(0), -1, -1).to(embed.device)
        normed_embed = self.norm(embed)
        return {'tag_hidden_state': embed + self.self_attention(normed_embed, normed_embed, normed_embed, attn_mask=attention_mask)[0]}

    @staticmethod
    def add_specific_args(parent_parser: object) -> object:
        parser = parent_parser.add_argument_group(title='Tag Prediction Settings')
        parser.add_argument('-tp', '--tag_prediction', default=False, action='store_true')
        return parent_parser

    @staticmethod
    def check_args(args: object) -> None:
        if args.tag_prediction:
            if not isinstance(args.crits, list):
                args.crits = [args.crits]
            if 'attribute' not in args.crits:
                args.crits.append('tag')

        base_path = os.path.join(Constants.base_data_path, args.dataset, 'retrieval')
        arch_mapping = {
            'ViT': (512, os.path.join(base_path, 'CLIP_ViT-B-32_unique.hdf5')),
            'ViT16': (512, os.path.join(base_path, 'CLIP_ViT-B-16_unique.hdf5')),
            'RN101': (512, os.path.join(base_path, 'CLIP_RN101_unique.hdf5')),
            'RN50': (1024, os.path.join(base_path, 'CLIP_RN50_unique.hdf5')),
            'RN50x4': (640, os.path.join(base_path, 'CLIP_RN50x4_unique.hdf5')),
            'RN50x16': (768, os.path.join(base_path, 'CLIP_RN50x16_unique.hdf5')),
        }

        if args.retrieval:
            assert getattr(args, 'pointer', None) is not None
            args.modality = args.modality + 't'
            args.dim_t, args.feats_t = arch_mapping[args.retrieval_arch]

        if args.tag_prediction:
            assert args.feats, "Please specify --feats"
            if not any(k in args.task for k in ['VAP', 'TAP', 'DAP']):
                assert args.decoder_modality_flags, "Please specify --decoder_modality_flags instead of --modality"
                assert args.predictor_modality_flags, "Please specify --predictor_modality_flags instead of --modality"

                args.modality_for_decoder = Constants.flag2modality[args.decoder_modality_flags]
                args.modality_for_predictor = Constants.flag2modality[args.predictor_modality_flags]

                _all = args.modality_for_decoder + args.modality_for_predictor
                args.modality = ''
                for char in ['a', 'm', 'i', 'r']:
                    if char in _all:
                        args.modality = args.modality + char

            if getattr(args, 'pointer', None):
                args.modality = args.modality + 't'

            if 'r' in args.modality:
                args.dim_r, args.feats_r = arch_mapping[args.retrieval_arch]

