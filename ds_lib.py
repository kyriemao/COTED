# data structure library file

from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations


class CCDExample:
    def __init__(self, turn_id, actual_context, response, nc_ids, oq, query):
        self.turn_id = turn_id
        self.actual_context = actual_context
        self.response = response
        self.nc_ids = nc_ids    # necessary context ids
        self.oq = oq
        self.query = query
        

class CCDDataset(Dataset):
    def __init__(self, tokenizer, filenames, args):
        self.examples = []

        for filename in filenames:
            with open(filename, 'r') as f:
                data = f.readlines()
            n = len(data)
            if 'use_data_percent' in args:
                n = int(args.use_data_percent * n)  
            for i in trange(n):
                # basic
                data[i] = json.loads(data[i])
                turn_id = data[i]['turn_id']
                
                # query
                query = tokenizer.encode(data[i]['query'] , add_special_tokens=True, max_length=args.max_query_length)
                
                # actual context
                actual_context = []
                context_qs = data[i]['context_qs']
                for cq in context_qs:           
                    actual_context.append(tokenizer.encode(cq, add_special_tokens=True))
                
                # necessary context
                nc_ids = []
                if 'depen_ids' in data[i]:
                    nc_ids = [x - 1 for x in data[i]['depen_ids']]
                
                # last response
                response = []
                if 'last_response' in data[i]:
                    last_response = data[i]['last_response']
                elif 'last_auto_response' not in data[i]:
                    last_response = None
                else:
                    last_response = data[i]['last_auto_response']
                if last_response is not None:
                    response.append(tokenizer.cls_token_id)
                    # Following ConvDR
                    response.extend(tokenizer.convert_tokens_to_ids(["<response>"]))
                    response.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(last_response)))
                    response.append(tokenizer.sep_token_id)

                # oracle query
                oq = tokenizer.encode(data[i]['oracle_query'], add_special_tokens=True,max_length=args.max_query_length)   

                example = CCDExample(
                    turn_id,
                    actual_context,
                    response,
                    nc_ids,
                    oq,
                    query
                )
                self.examples.append(example)
    

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def merge_with(self, dataset):
        if isinstance(dataset, CCDDataset):
            self.examples.extend(dataset.examples)

        
    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_turn_id": [],
                "bt_actual_seq": [],
                "bt_actual_seq_mask": [],
                "bt_necessary_seq": [],
                "bt_necessary_seq_mask": [],
                "bt_oq": [],
                "bt_oq_mask": [],
                "bt_ac_match_ids": [],
                "bt_nc_match_ids": []
            }

            for example in batch:
                example = example[0]
                actual_seq = []
                necessary_seq = []
                ac_match_ids = []
                nc_match_ids = []
   
                actual_seq.extend(example.query)
                necessary_seq.extend(example.query)
            
                for i, q in enumerate(example.actual_context):
                    if i in example.nc_ids:
                        ac_match_ids.append(len(actual_seq))
                    actual_seq.extend(q)
                for nc_id in example.nc_ids:
                    nc_match_ids.append(len(necessary_seq))
                    necessary_seq.extend(example.actual_context[nc_id])
                
                # deal response
                ac_match_ids.append(len(actual_seq))
                nc_match_ids.append(len(necessary_seq))
                actual_seq.extend(example.response)
                necessary_seq.extend(example.response)
               
                if args.dataset == 'cast':
                    assert len(ac_match_ids) == len(nc_match_ids)

                # padding
                actual_seq, actual_seq_mask = pad_seq_ids_with_mask(actual_seq, max_length=args.max_concat_length)
                necessary_seq, necessary_seq_mask = pad_seq_ids_with_mask(necessary_seq, max_length=args.max_concat_length)
                oq, oq_mask = pad_seq_ids_with_mask(example.oq, max_length=args.max_query_length)

                
                collated_dict["bt_turn_id"].append(example.turn_id)
                collated_dict["bt_actual_seq"].append(actual_seq)
                collated_dict["bt_actual_seq_mask"].append(actual_seq_mask)

                collated_dict["bt_necessary_seq"].append(necessary_seq)
                collated_dict["bt_necessary_seq_mask"].append(necessary_seq_mask)
                collated_dict["bt_oq"].append(oq)
                collated_dict["bt_oq_mask"].append(oq_mask)
                collated_dict["bt_ac_match_ids"].append(ac_match_ids)
                collated_dict["bt_nc_match_ids"].append(nc_match_ids)

            
            # change to tensor
            for key in collated_dict:
                if key not in  ['bt_turn_id', 'bt_ac_match_ids', 'bt_nc_match_ids']:
                    collated_dict[key] = torch.tensor(collated_dict[key],
                                                        dtype=torch.long)
             
            return collated_dict

        return collate_fn

'''
auxillary functions
'''

def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask


def conversation_data_aug_v1(train_data, max_sample_ratio):
    examples = train_data.examples
    ratio = max_sample_ratio
    new_examples = []
    for example in tqdm(examples):
        nc_ids = example.nc_ids
        nc_ids_set = set(nc_ids)
        candidates = set(range(len(example.actual_context))) - nc_ids_set
        if len(candidates) == 0:
            continue
        for l in range(1, len(candidates)):
            aug_samples = list(combinations(candidates, l))
            if len(aug_samples) > ratio:
                aug_samples = random.sample(aug_samples, ratio)
            
            for sample in aug_samples:
                new_ac_ids = sorted(list(nc_ids) + list(sample))
                new_actual_context = []
                new_nc_ids = []
                for i, ac_id in enumerate(new_ac_ids):
                    new_actual_context.append(example.actual_context[ac_id])
                    if ac_id in nc_ids_set:
                        new_nc_ids.append(i)
                new_example = CCDExample(example.turn_id, 
                                     new_actual_context, 
                                     example.response, 
                                     new_nc_ids, 
                                     example.oq, 
                                     example.query)

                new_examples.append(new_example)


    train_data.examples.extend(new_examples)
    print('add {} augmented samples'.format(len(new_examples)))

    return train_data



def TDL_measurer(s):
    ts_ac_plus_q = set()
    ts_oracle_q = set()
    for x in s.actual_context:
        ts_ac_plus_q = ts_ac_plus_q.union(set(x))
    ts_ac_plus_q = ts_ac_plus_q.union(set(s.query))
    ts_ac_plus_q = ts_ac_plus_q.union(set(s.response))
    ts_oracle_q = set(s.oq)

    tdl = len(ts_ac_plus_q.union(ts_oracle_q).difference(ts_ac_plus_q & ts_oracle_q))
    return tdl

def ACL_measurer(s):
    return len(s.actual_context)


def MPS_measurer(train_data, ndcg_path):
    with open(ndcg_path, 'r') as f:
        data = f.readlines()
    d = {}
    for line in data:
        line = line.strip().split('\t')
        d[line[0]] = float(d[line[1]])
    
    mps_list = []
    for example in train_data.examples:
        mps_list.append(d[example.turn_id])

    return mps_list


def curriculum_split(train_data):
    # define R-level dataset list
    R = 5
    L = []
    examples = train_data.examples
    for example in examples:
        tdl = TDL_measurer(example)
        L.append((example, tdl))
    L = sorted(L, key=lambda x : x[1])
    step = int(len(L) / R)

    res_datasets = []
    for i in range(0, len(L), step):
        dataset = CCDDataset(None, [], None)    # blank dataset
        dataset.examples = L[i : i + step]
        res_datasets.append(dataset)
    
    return res_datasets