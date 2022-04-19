from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')


import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from models import load_model
from my_utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, get_optimizer
from ds_lib import CCDDataset, curriculum_split, conversation_data_aug_v1



def save_model(args, model, epoch):
    output_dir = oj(args.model_output_dir, 'epoch-{}'.format(epoch))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    args.tokenizer.save_pretrained(output_dir)
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Epoch {}, Save checkpoint at {}".format(epoch, output_dir))


def merge_batch(b1, b2):
    res = {}
    res['bt_turn_id'] = b1['bt_turn_id'] + b2['bt_turn_id']
    res['bt_actual_seq'] = torch.cat((b1['bt_actual_seq'], b2['bt_actual_seq']), dim=0)
    res['bt_actual_seq_mask'] = torch.cat((b1['bt_actual_seq_mask'], b2['bt_actual_seq_mask']), dim=0)
    res['bt_necessary_seq'] = torch.cat((b1['bt_necessary_seq'], b2['bt_necessary_seq']), dim=0)
    res['bt_necessary_seq_mask'] = torch.cat((b1['bt_necessary_seq_mask'], b2['bt_necessary_seq_mask']), dim=0)
    res['bt_oq'] = torch.cat((b1['bt_oq'], b2['bt_oq']), dim=0)
    res['bt_oq_mask'] = torch.cat((b1['bt_oq_mask'], b2['bt_oq_mask']), dim=0)
    res['bt_ac_match_ids'] = b1['bt_ac_match_ids'] + b2['bt_ac_match_ids']
    res['bt_nc_match_ids'] = b1['bt_nc_match_ids'] + b2['bt_nc_match_ids']
    return res



def two_step_multi_task_training(args, train_dataset, teacher_model, student_model, fold_id='single'):
    # conversationa data augmentation
    if args.data_aug_ratio > 0:
        train_dataset = conversation_data_aug_v1(train_dataset, args.data_aug_ratio)

    num_train_epochs = args.num_train_epochs

    # curriculum learning, split into R buckets
    dataset_list = curriculum_split(train_dataset)

    total_training_steps, pre_steps = 0, 0
    for dataset in dataset_list:
        total_training_steps += len(dataset) + pre_steps
        pre_steps += len(dataset)
    total_training_steps += (num_train_epochs - len(dataset_list)) * pre_steps
    total_training_steps /= args.train_batch_size

    optimizer_kd = get_optimizer(args, student_model, weight_decay=args.weight_decay)
    optimizer_den = get_optimizer(args, student_model, weight_decay=args.weight_decay)
    scheduler_kd = get_linear_schedule_with_warmup(optimizer_kd, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps)
    scheduler_den = get_linear_schedule_with_warmup(optimizer_den, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps)

    kd_loss_func = nn.MSELoss()
    device = args.device
    pre_dataset = None

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))

    global_step = 0
    epoch_iterator = trange(num_train_epochs, desc="Epoch")
    for epoch in epoch_iterator:
        if epoch >= len(dataset_list):
            train_dataset = pre_dataset
            pre_dataset = None
        else:
            train_dataset = dataset_list[epoch]

        if pre_dataset:
            train_loader = DataLoader(train_dataset, batch_size = int(args.train_batch_size / 2), shuffle=True , collate_fn=train_dataset.get_collate_fn(args))
            pre_loader_iter = iter(DataLoader(pre_dataset, batch_size = int(args.train_batch_size / 2), shuffle=True , collate_fn=train_dataset.get_collate_fn(args)))
        else:
            train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle=True , collate_fn=train_dataset.get_collate_fn(args))
            pre_loader_iter = None

        for batch in tqdm(train_loader,  desc="Step"):
            student_model.train()
            teacher_model.eval()
            
            # curriculum sampling
            if epoch < len(dataset_list) and pre_loader_iter:
                # pair training
                pre_batch = next(pre_loader_iter)
                batch = merge_batch(batch, pre_batch)

            bt_actual_seq = batch['bt_actual_seq'].to(device)
            bt_actual_seq_mask = batch['bt_actual_seq_mask'].to(device)
            if args.nc_mimic_loss_weight > 0 or args.add_denoising_loss:
                bt_necessary_seq = batch['bt_necessary_seq'].to(device)
                bt_necessary_seq_mask = batch['bt_necessary_seq_mask'].to(device)

            actual_query_emb = student_model(bt_actual_seq, bt_actual_seq_mask)
            with torch.no_grad():
                # freeze teacher's parameters
                teacher_embs = teacher_model(batch['bt_oq'].to(device), batch['bt_oq_mask'].to(device)).detach()

            # kd loss
            loss_mim = kd_loss_func(actual_query_emb, teacher_embs)
            if args.nc_mimic_loss_weight > 0:
                necessary_query_emb = student_model(bt_necessary_seq, bt_necessary_seq_mask)
                loss_mim += args.nc_mimic_loss_weight * kd_loss_func(necessary_query_emb, teacher_embs) 
            loss_mim.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer_kd.step()
            scheduler_kd.step()
            student_model.zero_grad()

            # denoising loss
            if args.add_denoising_loss and len(batch['bt_ac_match_ids']) > 0:
                _, ac_embs = student_model(bt_actual_seq, bt_actual_seq_mask, term_match_ids = batch['bt_ac_match_ids'])
                _, nc_embs = student_model(bt_necessary_seq, bt_necessary_seq_mask, term_match_ids = batch['bt_nc_match_ids'])

                loss_den = 0
                for i in range(len(ac_embs)):
                    loss_den += kd_loss_func(ac_embs[i], nc_embs[i].detach())
                loss_den /= len(ac_embs)
                loss_den *= args.denoising_weight
                loss_den.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                optimizer_den.step()
                scheduler_den.step()
                student_model.zero_grad()

            global_step += 1

        # save the student model, count from 1
        epoch += 1
        if args.save_epochs > 0 and epoch > 0 and (epoch) % args.save_epochs == 0:
            save_model(args, student_model, epoch)

        if epoch < len(dataset_list): 
            if not pre_dataset:
                pre_dataset = train_dataset
            else:
                pre_dataset.merge_with(train_dataset) 

    # always save the final model
    save_model(args, student_model, 'final')
    args.tb_writer.close()




def do_train():
    args = get_args()
    set_seed(args)

    # load the teacher model
    config, tokenizer, teacher_model = load_model(
                        pretrained_checkpoint_path=args.teacher_model_path)

    args.tokenizer = tokenizer
    teacher_model = teacher_model.to(args.device)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    NUM_FOLD = 5
    base_output_dir = args.model_output_dir
    base_log_dir = args.log_dir
    for i in range(NUM_FOLD):
        # build training data
        # remove fold i for evaulation and use the other folds for training
        train_files = []
        for j in range(NUM_FOLD):
            if j != i:
                suffix = '.{}'.format(j)    # e.g. turn.txt.1
                train_files.append(args.train_file + suffix)

        train_dataset = CCDDataset(tokenizer, train_files, args)
        train_loader = DataLoader(train_dataset, 
                                batch_size = args.train_batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))
        
        args.model_output_dir = base_output_dir + '/fold_{}'.format(i)
        args.log_dir = base_log_dir + '/fold_{}'.format(i)
        check_dir_exist_or_build([args.model_output_dir, args.log_dir])
        args.tb_writer = SummaryWriter(log_dir=args.log_dir)    # tensorboard writer

        set_seed(args)
        config, tokenizer, student_model = load_model(
                    pretrained_checkpoint_path=args.student_model_path)
        
        student_model = student_model.to(args.device)

        two_step_multi_task_training(args, train_dataset, teacher_model, student_model, fold_id = i)


        # del model and flush
        del student_model
        torch.cuda.empty_cache()







def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cast', help="the dataset name")
    parser.add_argument("--teacher_model_path", type=str, help="The teacher model path")
    parser.add_argument("--student_model_path", type=str, help="The student model path")
    parser.add_argument("--teacher_model_type", 
                        type=str, default="ANCE", 
                        help="The teacher model type used for training, ANCE by default")
    parser.add_argument("--num_negatives", 
                        type=int, 
                        default=9,
                        help="Number of negative documents per query."
    )
    parser.add_argument("--use_rank", 
                        action='store_true', 
    )
    parser.add_argument("--train_file",
                        type=str,
                        help="The test dataset.")
    parser.add_argument( "--log_dir", 
                        type=str,
                        help="Directory for tensorboard logging.")
    parser.add_argument("--max_query_length",
                        default=64,
                        type=int,
                        help="Max input query length after tokenization."
                        "This option is for single query input.")                     
    parser.add_argument("--max_doc_length",
                        default=512,
                        type=int,
                        help="Max doc length")
    parser.add_argument("--max_concat_length",
                        default=512,
                        type=int,
                        help="Max concat length. 512 for CAsT 2020.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        required=True,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--use_data_percent",
                        default=1.0,
                        type=float,
                        help="the percent of the used training samples")
    parser.add_argument("--learning_rate",
                        default=5e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")                
    parser.add_argument("--per_gpu_train_batch_size",
                        default=4,
                        type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument("--n_gpu",
                        default=1,
                        type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument('--save_epochs',
                        type=int,
                        default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization.")
    parser.add_argument( "--model_output_dir",
                        type=str,
                        help="model store address.")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--use_response_type",
                        type=str,
                        required=True,
                        help="the respone type to use, [default, auto, manual, or no]")
    parser.add_argument("--load_student_model_from_checkpoint",
                        action='store_true',
                        help="whether to load student model checkpoint to continue train")

    # my method properties
   
    parser.add_argument("--data_aug_ratio",
                        type=int,
                        default=0,
                        help="data augmentation ratio")
    parser.add_argument("--use_curriculum_training",
                        action='store_true',
                        help="adopt curriculum training")
    parser.add_argument("--add_denoising_loss",
                        action='store_true',
                        help="use denoising loss")
    parser.add_argument("--nc_mimic_loss_weight",
                        type=float,
                        default=0.0)
    parser.add_argument("--denoising_weight",
                        type=float,
                        default=1.0)
    



    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.tb_writer = SummaryWriter(log_dir=args.log_dir)    # tensorboard writer

    if os.path.exists(args.model_output_dir) and os.listdir(
        args.model_output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.model_output_dir))

    check_dir_exist_or_build([args.model_output_dir, args.log_dir])

    return args




if __name__ == '__main__':
    do_train()
