import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
from data_preprocessing import tools

tokenizer=tools.get_tokenizer()
torch.nn.Transformer
class myModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)

        self.roberta_encoder.resize_token_embeddings(len(tokenizer))
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=768),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_rate)
        )
        self.start_layer = torch.nn.Linear(in_features=768, out_features=2)
        self.end_layer = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids, input_mask, input_seg, is_training=False):
        bert_output = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)  # (bsz, seq, dim)

        encoder_rep = bert_output[0]

        start_logits = self.start_layer(encoder_rep)  # (bsz, seq, 2)
        end_logits = self.end_layer(encoder_rep)  # (bsz, seq, 2)

        start_prob_seq = torch.nn.functional.softmax(start_logits, dim=-1)  # (bsz, seq, 2)
        end_prob_seq = torch.nn.functional.softmax(end_logits, dim=-1)  # (bsz, seq, 2)


        if is_training:
            return start_logits, end_logits
        else:
            return start_prob_seq, end_prob_seq