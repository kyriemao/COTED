from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')


import csv
import argparse
from models import load_model
from my_utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from ds_lib import CCDDataset, ConvDRDataset
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import torch
import numpy as np

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. output the result
'''



def build_faiss_index(args):
    logger.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index


def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    merged_candidate_matrix = None

    for block_id in range(args.passage_block_num):
        logger.info("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage__emb_p__data_obj_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage__embid_p__data_obj_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding2id = pickle.load(handle)
        except:
            break
        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))
        index.add(passage_embedding)

        # ann search
        tb = time.time()
        D, I = index.search(query_embeddings, topN)
        elapse = time.time() - tb
        logger.info({
            'time cost': elapse,
            'query num': query_embeddings.shape[0],
            'time cost per query': elapse / query_embeddings.shape[0]
        })

        candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
        D = D.tolist()
        candidate_id_matrix = candidate_id_matrix.tolist()
        candidate_matrix = []

        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]

        index.reset()
        del passage_embedding
        del passage_embedding2id

        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
        
        # merge
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                         candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1

    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logger.info(merged_I)

    return merged_D, merged_I





def get_test_query_embedding(args):
    # load model
    set_seed(args)
    config, tokenizer, model = load_model(args.test_model_path)
    mdoel = model.to(args.device)

    # test dataset/dataloader
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("Buidling test dataset...")
    if args.model_type == 'ANCE':
        if args.test_my_way or args.test_depen:
            test_dataset = CCDDataset(tokenizer, [args.test_file], args)
        else:
            test_dataset = ConvDRDataset(tokenizer, [args.test_file], args)
    else:
        raise ValueError("{} has not been implemented".format(args.model_type))
    test_loader = DataLoader(test_dataset, 
                            batch_size = args.eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args))

    logger.info("Generating query embeddings for testing...")
    model.zero_grad()

    embeddings = []
    embedding2id = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            model.eval()
            batch_turn_id = batch['bt_turn_id']
            query_embs = model(batch['bt_actual_seq'].to(args.device), batch['bt_actual_seq_mask'].to(args.device))
            query_embs = query_embs.detach().cpu().numpy()
            embeddings.append(query_embs)
            embedding2id.extend(batch_turn_id)

    embeddings = np.concatenate(embeddings, axis = 0)
    torch.cuda.empty_cache()

    return embeddings, embedding2id


def output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    offset2pid,
                    args):
    
    qids_to_ranked_candidate_passages = {}
    topN = args.top_n

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
    # query
    qid2query = {}
    with open(args.test_file, 'r') as f:
        data = f.readlines()
    for record in data:
        record = json.loads(record.strip())
        qid2query[record['turn_id']] = record['query']
    
    # all passages
    all_passages = load_collection(oj(args.collection_dir, "collection.tsv"))

    # write to file
    logger.info('begin to write the output...')

    output_file = oj(args.output_dir, 'res.jsonl')
    output_trec_file = oj(args.output_dir, 'res.trec')
    with open(output_file, "w") as f, open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            query = qid2query[qid]
            for i in range(topN):
                pid, score = passages[i]
                passage = all_passages[pid]

                f.write(
                        json.dumps({
                            "query": query,
                            "doc": passage,
                            "query_id": str(qid),
                            "doc_id": str(pid),
                            "rank": i,
                            "retrieval_score": score,
                        }) + "\n")
                
                g.write(
                        str(qid) + " Q0 " + str(pid) + " " + str(i + 1) +
                        " " + str(-i - 1 + 200) + " ance\n")
    
    logger.info("output file write ok at {}".format(args.output_dir))



def do_test():
    args = get_args()
    set_seed(args) 
    
    index = build_faiss_index(args)
    if not args.cross_validate:
        # args.test_model_path = args.test_model_path + '/epoch - test epoch'
        query_embeddings, query_embedding2id = get_test_query_embedding(args)
    else:
        base_test_file = args.test_file
        base_model_path = args.test_model_path
        NUM_FOLD = 5
        
        total_query_embeddings = []
        total_query_embedding2id = []
        for i in range(NUM_FOLD):
            args.test_file = base_test_file + '.{}'.format(i)
            args.test_model_path = base_model_path + '/fold_{}/epoch-{}'.format(i, args.test_epoch)

            query_embeddings, query_embedding2id = get_test_query_embedding(args)
            total_query_embeddings.append(query_embeddings)
            total_query_embedding2id.extend(query_embedding2id)

        total_query_embeddings = np.concatenate(total_query_embeddings, axis = 0)
        query_embeddings = total_query_embeddings
        query_embedding2id = total_query_embedding2id
        args.test_file = base_test_file


    # score_mat: score matrix, test_query_num * (topn * block_num)
    # pid_mat: corresponding passage ids
    passge_embeddings_dir = oj(args.collection_dir, 'passage_embeddings')
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                     args,
                                                     passge_embeddings_dir, 
                                                     index, 
                                                     query_embeddings, 
                                                     args.top_n) 

    with open(oj(args.collection_dir, "offset2pid.pickle"), "rb") as f:
        offset2pid = pickle.load(f)

    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    offset2pid,
                    args)
    
    logger.info("test finish!")
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cast', help="the dataset name")
    parser.add_argument("--test_model_path", 
                        type=str, 
                        help="The model checkpoint.")
    parser.add_argument("--model_type", 
                        type=str, required=True, 
                        help="The model type used for testing.")
    parser.add_argument("--test_file",
                        type=str,
                        help="The test dataset.")

    parser.add_argument("--max_query_length",
                        default=64,
                        type=int,
                        help="Max input query length after tokenization."
                        "This option is for single query input.")
                        
    parser.add_argument("--max_doc_length",
                        default=256,
                        type=int,
                        help="Max doc length")
    parser.add_argument("--max_concat_length",
                        default=512,
                        type=int,
                        help="Max concat length")
                        
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=4,
                        type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization.")

    parser.add_argument("--collection_dir", type=str, help="Dir path to collection")

    parser.add_argument( "--passage_block_num",
                        type=int,
                        required=True,
                        help="the number of passage block_num")
    parser.add_argument( "--output_dir",
                        type=str,
                        help="TREC-style run file, to be evaluated by the trec_eval tool.")

    parser.add_argument("--top_n",
                        default=100,
                        type=int,
                        help="Number of retrieved documents for each query.")
    parser.add_argument("--n_gpu",
                        default=1,
                        type=int,
                        help="num gpu")
    parser.add_argument("--use_gpu",
                        action='store_true',
                        help="Whether to use GPU for Faiss.")
    parser.add_argument("--use_rank",
                    action='store_true',
                    help="whether add rank data into the dataset")
    parser.add_argument("--use_response_type",
                        type=str,
                        required=True,
                        help="the respone type to use, [default, auto, manual, no]")
    parser.add_argument("--use_context_type",
                        type=str,
                        default='org',
                        help="the context type to use, [query_first, query_last]")
    parser.add_argument("--cross_validate",
                        action='store_true',
                        help="Set when doing cross validation.")
    parser.add_argument("--test_epoch",
                        type=str,
                        default='final')

    


    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    check_dir_exist_or_build([args.output_dir])

    return args






if __name__ == '__main__':
    do_test()
