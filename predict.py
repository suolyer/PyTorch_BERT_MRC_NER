import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch import optim
import numpy as np
from utils.arguments_parse import args
import data_preprocessing
from model.model import myModel
from model.loss_function import multilabel_cross_entropy
from model.metrics import metrics
from data_preprocessing import *
import json
from tqdm import tqdm
import unicodedata, re
from data_preprocessing import tools


device = torch.device('cuda')

added_token = ['[unused1]', '[unused1]']
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_path, additional_special_tokens=added_token)

label2id,id2label,num_labels=tools.load_schema()

def load_data(file_path):
    with open(file_path,'r',encoding='utf8') as f:
        lines=f.readlines() 
        sentences=[]
        for line in lines:
            data=json.loads(line)
            sentences.append(data['text'])
        return sentences


def get_actual_id(text,role_text_event_type):
    text_encode=tokenizer.encode(text)
    one_input_encode=tokenizer.encode(role_text_event_type)
    text_start_id=tools.search(text_encode[1:-1],one_input_encode)
    text_end_id=text_start_id+len(text_encode)-1
    if text_end_id>args.max_length:
        text_end_id=args.max_length
    
    text_token=tokenizer.tokenize(text)
    text_mapping = tools.token_rematch().rematch(text,text_token)

    return text_start_id,text_end_id,text_mapping


def get_start_end_i(start_logits,end_logits,span_logits,text_start_id,text_end_id):

    arg_index = []
    i_start, i_end = [], []
    for i in range(text_start_id, text_end_id):
        if start_logits[i][1] > 0.48:
            i_start.append(i)
        if end_logits[i][1] > 0.48:
            i_end.append(i)
    # 然后遍历i_end
    cur_end = -1
    for e in i_end:
        s = []
        for i in i_start:
            if e >= i >= cur_end:
                s.append(i)
        max_s = 0.4
        t = None
        for i in s:
            if span_logits[i][e][1] > max_s:
                t = (i, e)
                max_s = span_logits[i][e][1]
        print('max_s:',max_s)
        cur_end = e
        if t is not None:
            arg_index.append(t)
    return arg_index


def sapn_decode(span_logits):
    arg_index=[]
    for i in range(len(span_logits)):
        for j in range(i,len(span_logits[i])):
            if span_logits[i][j][1]>0.5:
                arg_index.append((i,j))
    return arg_index


def extract_entity_from_start_end_ids(start_logits, end_logits,text_start_id, text_end_id):
        # 根据开始，结尾标识，找到对应的实体
    start_ids=[0 for i in range(len(start_logits))]
    end_ids=[0 for i in range(len(end_logits))]

    for i in range(text_start_id, text_end_id):
        if start_logits[i][1] > 0.48:
            start_ids[i] = 1
        if end_logits[i][1] > 0.48:
            end_ids[i] = 1

    start_end_tuple_list = []
    for i, start_id in enumerate(start_ids):
        if start_id == 0:
            continue
        if end_ids[i] == 1:
            start_end_tuple_list.append((i, i))
            continue
        j = i + 1
        find_end_tag = False
        while j < len(end_ids):
            # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
            if start_ids[j] == 1:
                break
            if end_ids[j] == 1:
                start_end_tuple_list.append((i, j))
                find_end_tag = True
                break
            else:
                j += 1
        if not find_end_tag:
            start_end_tuple_list.append((i, i))
    return start_end_tuple_list



def main():

    with torch.no_grad():
        model = myModel(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(args.checkpoints))

        sentences = load_data(args.test_path)
        with open('./output/result.json','w',encoding='utf-8') as f:
            for sentence in tqdm(sentences):
                type_sent_list=[entity_type+'[SEP]'+sentence for entity_type in label2id.keys()]
                input_ids=[]
                input_seg=[]
                input_mask=[]
                for type_sent in type_sent_list:
                    encode_dict=tokenizer.encode_plus(type_sent,max_length=args.max_length,pad_to_max_length=True)
                    input_ids.append(encode_dict['input_ids'])
                    input_seg.append(encode_dict['token_type_ids'])
                    input_mask.append(encode_dict['attention_mask'])
                input_ids = torch.Tensor(input_ids).long()
                input_seg = torch.Tensor(input_seg).long()
                input_mask = torch.Tensor(input_mask).float()
                start_logit,end_logit = model( 
                            input_ids=input_ids.to(device), 
                            input_mask=input_mask.to(device),
                            input_seg=input_seg.to(device),
                            is_training=False)
                start_logit=start_logit.to(torch.device('cpu')).numpy().tolist()
                end_logit=end_logit.to(torch.device('cpu')).numpy().tolist()
        
                entity_list=[]
                for i,type_sent in enumerate(type_sent_list):
                    text_start_id,text_end_id,text_mapping=get_actual_id(sentence,type_sent)
                    args_dict=dict()
                    entity_type=type_sent.split('[SEP]')[0]


                    # args_index = get_start_end_i(start_logits[i],end_logits[i],span_logits[i],text_start_id,text_end_id)
                    args_index = extract_entity_from_start_end_ids(start_logit[i],end_logit[i],text_start_id,text_end_id)
                    # args_index=sapn_decode(span_logits[i])

                    for k in args_index:
                        tmp={}
                        dv = 0
                        while text_mapping[k[0]-text_start_id+dv] == []:
                            dv+=1
                        start_split=text_mapping[k[0]-text_start_id+dv]

                        dv = 0
                        while text_mapping[k[1]-text_start_id+dv] == []:
                            dv-=1
                        end_split=text_mapping[k[1]-text_start_id+dv]

                        tmp['type']=entity_type
                        tmp['argument']=sentence[start_split[0]:end_split[-1]+1]  #抽取实体
                        entity_list.append(tmp)
                result={}
                result['text']=sentence
                result['entity_list']=entity_list

                json_data=json.dumps(result,ensure_ascii=False)
                f.write(json_data+'\n')
            
if __name__ == '__main__': 
    main()