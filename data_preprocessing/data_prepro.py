import os
import sys
sys.path.append('./')
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils.arguments_parse import args
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re
from data_preprocessing import tools
from tqdm import tqdm

tokenizer=tools.get_tokenizer()
label2id,id2label,num_labels=tools.load_schema()

def load_data(file_path):
    all_entity_type_list=[k for k in label2id.keys()]
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        arguments = []
        args_has_flag =[]
        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            entity_list = data['entity_list']
            entity_type_list=list(set([i['type'] for i in entity_list]))
            no_entiy_type_list = [entity_type for entity_type in all_entity_type_list if entity_type not in entity_type_list]
            for entity in entity_list:
                tmp_sent = entity['type'] +'[SEP]'+ text
                if tmp_sent not in sentences:
                    sentences.append(tmp_sent)
                    arguments.append([entity['argument']])
                    args_has_flag.append(0)
                else:
                    idx = sentences.index(tmp_sent)
                    arguments[idx].append(entity['argument'])
            for entity_type in no_entiy_type_list:
                args_has_flag.append(1)
                tmp_sent = entity_type + '[SEP]' + text
                sentences.append(tmp_sent)
                arguments.append([])       
        return sentences, arguments, args_has_flag


def encoder(sentence, argument,flag):
    encode_dict = tokenizer.encode_plus(sentence,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
    encode_sent = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict['attention_mask']
    seq_id = [i for i, v in enumerate(encode_sent) if v == 102]  #记录text前面的【SEP】位置和后面的【SEP】位置
    seq_mask = [0 for i in range(args.max_length)]
    for i in range(seq_id[0]-1, seq_id[1]+1):
        seq_mask[i] = 1

    start_label = [0 for i in range(args.max_length)]
    end_label = [0 for i in range(args.max_length)]

    # if flag==1:  #如果没有论元角色，则将【SEP】两个位置设为1
    #     start_label[seq_id[0]]=1
    #     end_label[seq_id[0]]=1

    span_label = [0 for i in range(args.max_length)]
    span_label = [span_label for i in range(args.max_length)]
    span_label = np.array(span_label)
    for arg in argument:
        encode_arg = tokenizer.encode(arg)
        start_idx = tools.search(encode_arg[1:-1], encode_sent)
        start_label[start_idx] = 1
        end_idx = start_idx + len(encode_arg[1:-1]) - 1
        end_label[end_idx] = 1
        span_label[start_idx, end_idx] = 1

    return encode_sent, token_type_ids, attention_mask, start_label, end_label, span_label, seq_mask, seq_id


def data_pre(file_path):
    sentences, arguments, args_has_flag = load_data(file_path)
    data = []
    for i in tqdm(range(len(sentences))):
        encode_sent, token_type_ids, attention_mask, start_label, end_label, span_label, seq_mask,seq_id = encoder(
            sentences[i], arguments[i],args_has_flag[i])
        tmp = {}
        tmp['input_ids'] = encode_sent
        tmp['input_seg'] = token_type_ids
        tmp['input_mask'] = attention_mask
        tmp['start_label'] = start_label
        tmp['end_label'] = end_label
        tmp['span_label'] = span_label
        tmp['seq_mask'] = seq_mask
        tmp['args_has_flag']=args_has_flag[i]
        tmp['seq_id']=seq_id
        data.append(tmp)

    return data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "input_seg": torch.tensor(item['input_seg']).long(),
            "input_mask": torch.tensor(item['input_mask']).float(),
            "start_label": torch.tensor(item['start_label']).long(),
            "end_label": torch.tensor(item['end_label']).long(),
            "span_label": torch.tensor(item['span_label']).long(),
            "seq_mask": torch.tensor(item['seq_mask']).long(),
            "args_has_flag": torch.tensor(item['args_has_flag']).long(),
            "seq_id":torch.tensor(item['seq_id']).long()
        }
        return one_data


def yield_data(file_path):
    tmp = MyDataset(data_pre(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':

    data = data_pre(args.test_path)
    print(data[0])

    # print(input_ids_list[0])
    # print(token_type_ids_list[0])
    # print(start_labels[0])
