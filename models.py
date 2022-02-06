import sys

sys.path += ['../']
import torch
from torch import nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          BertModel, BertTokenizer, BertConfig)
import torch.nn.functional as F
from IPython import embed
import time


# well-trained ANCE model for DPR
class ANCE_BertDPR(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)

        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def query_emb_term_match(self, input_ids, attention_mask, term_match_ids):
        emb_all = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)[0]
        batch_size = emb_all.size(0)
        res = []
        for i in range(batch_size):
            res.append(emb_all[i][term_match_ids[i]])
        query_embs = self.norm(self.embeddingHead(emb_all[:, 0]))
        return query_embs, res

    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, term_match_ids = None):
        if term_match_ids is None:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.query_emb_term_match(input_ids, attention_mask, term_match_ids)





'''
model-related functions
'''

def load_model(pretrained_checkpoint_path):
    config = RobertaConfig.from_pretrained(
        pretrained_checkpoint_path,
        finetuning_task="MSMarco",
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        pretrained_checkpoint_path,
        do_lower_case=True
    )
    model = ANCE_BertDPR.from_pretrained(pretrained_checkpoint_path, config=config)
    return config, tokenizer, model