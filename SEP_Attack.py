import torch.multiprocessing as mp

import os

import argparse
import os
import numpy as np
import json

np.random.seed(1234)
import pickle
import dataloader
import criteria
import random

random.seed(0)
import csv
import math
import pdb

from SEP_utils import Ensemble,VictimModel

from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig


# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(2147483647)



# tf.disable_v2_behavior()
from collections import defaultdict
import torch
import torch.nn as nn


import warnings

warnings.filterwarnings("ignore", category=UserWarning)


np.random.seed(0)
random.seed(0)





def get_change_ls(x1, x2):
    change_ls = []
    for idx in range(len(x1)):
        if x1[idx] != x2[idx]:
            change_ls.append(idx)
    return change_ls


def get_important_order(text_ls, ensemble, true_label, words_perturb):
    wi_score = []
    wi_ids = []
    for i in range(0, len(words_perturb), 256):

        j = 0
        texts = []
        idxs = []
        while j < 256 and i + j < len(words_perturb):
            new_text = text_ls[:]
            new_text[words_perturb[i + j][0]] = ""
            texts.append(" ".join(new_text))
            idxs.append(words_perturb[i + j][0])
            j += 1
        if len(texts) > 0:
            imps = ensemble(texts, true_label)
            wi_score.extend(imps)
            wi_ids.extend(idxs)

    return wi_score, wi_ids


def find_max_synonym(synonyms, x_t, idx, ensemble, orig_label):
    stexts = []
    sloss = None
    for s in synonyms:
        x_t_tmp = x_t[:]
        x_t_tmp[idx] = s
        stexts.append(" ".join(x_t_tmp))
    if len(stexts) > 0:
        sloss = ensemble(stexts, orig_label)
        min_s_idx = torch.argmin(sloss)
        min_s = synonyms[min_s_idx]
        return min_s
    else:
        return x_t[idx]


def optimize(
    text_ls,
    x_t,
    ensemble,
    orig_label,
    words_perturb,
    max_change_num,
    att_succ_num,
    synonyms_dict,
):
    change_ls = get_change_ls(text_ls, x_t)

    wi_score, wi_ids = get_important_order(x_t, ensemble, orig_label, words_perturb)


    wi_score = 0-torch.tensor(wi_score)

    while torch.max(wi_score) - torch.min(wi_score) < 1:
        wi_score *= 10.0

    wi_distribution = torch.softmax(wi_score, dim=0).numpy()
    wi_distribution /= wi_distribution.sum()


    perturb_word_idx_list = np.random.choice(
        wi_ids, max_change_num, replace=False, p=wi_distribution
    )


    change_num = len(change_ls)

    for idx in perturb_word_idx_list:
        if idx not in change_ls:
            change_num += 1

        if change_num > max_change_num:
            break
        x_t[idx] = find_max_synonym(
            synonyms_dict[text_ls[idx]], x_t, idx, ensemble, orig_label
        )


    change_ls = get_change_ls(x_t, text_ls)
    lcl = len(change_ls)
    for _ in range(lcl - att_succ_num):
        stexts = []
        for idx in change_ls:
            tmp_text = x_t[:]
            tmp_text[idx] = text_ls[idx]
            stexts.append(" ".join(tmp_text))
        # [batchsz,]
        sloss = ensemble(stexts, orig_label)
        min_idx_ = torch.argmin(sloss, dim=0)
        min_idx = change_ls[min_idx_]
        x_t[min_idx] = text_ls[min_idx]
        change_ls.remove(min_idx)



def early_stop(attack_scores, patience):
    if len(attack_scores) <= patience:
        return False
    try:
        for i in range(1, patience + 1):
            if attack_scores[-i] < attack_scores[-i - 1]:
                return False
        return True
    except Exception as e:
        pdb.set_trace()


def attack(
    top_k_words,
    text_ls,
    true_label,
    victim:VictimModel,
    word2idx,
    idx2word,
    cos_sim,
    ensemble,
    device,
    arg_k,
    text_idx,
    target_model,
    save_path,
    sim_score_window=15,
    batch_size=32,
    max_change_rate=0.2,
):

    alpha = 1  
    hash_qrs = {}
    att = victim([" ".join(text_ls)],true_label,hash_qrs)
    optim_step = {}

    # att = False
    if att:
        return None,None,None,None,None,None
    else:
        text_ls = text_ls[:256]
        orig_label = true_label
        
        len_text = len(text_ls)

        max_change_num = min(max(math.floor(len_text * max_change_rate), 3), len_text)
        att_succ_num = max(math.floor(len_text * 0.15), 1)

        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [
            word2idx[word] for idx, word in words_perturb if word in word2idx
        ]
        synonym_words = []

        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
        synonyms_all = []

        synonyms_dict = defaultdict(list)

        new_words_perturb = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms and len(synonyms) > 1:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms
                    new_words_perturb.append(idx)

        max_change_num = min(max_change_num, len(words_perturb))
        att_succ_num = min(att_succ_num, len(words_perturb))

        log_stack = []
        log_stack_flag = []
        x_t = text_ls[:]
        optim_step["optim"] = []

        
        ensemble.init_weights()

        for _ in range(ensemble.len_weights):
            ensemble.next_weights()
            torch.cuda.empty_cache()
            T = 10
            patience = 5
            attack_scores = []

            for t in range(T):
                optimize(
                    text_ls,
                    x_t,
                    ensemble,
                    orig_label,
                    words_perturb,
                    max_change_num,
                    att_succ_num,
                    synonyms_dict,
                )

                asn_text = x_t[:]
                loss, loss_unique = ensemble(
                    [" ".join(asn_text)], orig_label, unique=True
                )
                loss = loss[0].cpu()


                attack_scores.append(loss)
                if " ".join(asn_text) not in log_stack_flag:
                    log_stack_flag.append(" ".join(asn_text))
                    loss_unique = loss_unique[0]
                    log_stack.append((asn_text, loss_unique))
                    optim_step["optim"].append(
                        (" ".join(asn_text), float(loss_unique.detach().cpu().item()))
                    )
                if early_stop(attack_scores, patience):
                    break
            patience -= 1
            if patience < 3:
                patience = 3
        log_stack = sorted(log_stack, key=lambda x: x[-1])

        all_attack = 0
        all_qrs = 0
        ten_attack = 0
        ten_qrs = min(len(log_stack),10)
        for asn_text,score in log_stack:
            if victim([" ".join(asn_text)],orig_label,hash_qrs):
                all_attack = 1
                all_qrs+=1
                if all_qrs<=10:
                    ten_attack = 1
                    ten_qrs = all_qrs
                    break
                break
            all_qrs+=1
        
        log_stack = log_stack[:50]
        
        search_ns = [max(1,len(text_ls)*0.01),max(1,len(text_ls)*0.02),max(1,len(text_ls)*0.025),max(1,len(text_ls)*0.03),max(1,len(text_ls)*0.05),max(1,len(text_ls)*0.1),max(1,len(text_ls)*0.15),max(1,len(text_ls)*0.2),max(1,len(text_ls)*0.25),max(1,len(text_ls)*0.3)]
        # search_ns = [1,2,3,4,5,6,7,8,9]
        search_ns = [max(1,math.floor(i)) for i in search_ns]
        search_attack = [0,0,0,0,0,0,0,0,0,0]
        search_qrs = [0,0,0,0,0,0,0,0,0,0]
        
        synonyms = {}
        optim_step_ns = {}
        optim_step_ns['search'] = {}
        for ms_idx,ms in enumerate(search_ns):
            log_stack_sub = []
            for asn_text,score in log_stack:

                change_ls = []
                for w1,w2,w_idx in zip(text_ls,asn_text,range(len(text_ls))):
                    if w1 != w2:
                        change_ls.append((w1,w_idx))
                search_space = random.sample(change_ls,min(ms,len(change_ls)))

                stexts = []
                for w,w_idx in search_space:
                    if w not in synonyms:
                        res = list(zip(*(cos_sim[word2idx[w]])))
                        synonyms[w] = []
                        for rs in res[1]:
                            synonyms[w].append(idx2word[rs])

                    x_t_tmp = asn_text[:]
                    x_t_tmp[w_idx] = synonyms[w][1]
                    stexts.append(" ".join(x_t_tmp))

                if len(stexts) > 0:
                    sloss, sloss_unique = ensemble(stexts, orig_label, unique=True)
                    # sloss_unique [len(text_ls),1]
                    sloss_unique = sloss_unique.squeeze(dim=-1).cpu().tolist()
                    # scores_cur[sn_indx].extend(sloss_unique)
                    log_stack_sub.append((asn_text,np.mean(sloss_unique)))

                # if len(log_stack[ms_idx]) == 10:
                #     break
            log_stack_sub = sorted(log_stack_sub,key=lambda x:x[-1])
            q = 0
            optim_step_ns['search'][ms] = log_stack_sub
            for asn_text,score in log_stack_sub[:10]:
                if victim([" ".join(asn_text)],orig_label,hash_qrs):
                    search_attack[ms_idx] = 1
                    q +=1
                    search_qrs[ms_idx] = q
                    break
                q+=1
                search_qrs[ms_idx] = q
                

        return all_attack,all_qrs,ten_attack,ten_qrs,search_attack,search_qrs


def generating_random_weights(weight_size, weight_nums=100):
    weights = torch.empty(weight_nums, weight_size).uniform_(
        0, 1
    )  
    weights = weights / torch.norm(weights, dim=1, keepdim=True)  
    return weights


def main():
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Which dataset to attack."
    )
    
    parser.add_argument(
        "--target_model",
        type=str,
        required=True,
        help="Target models for text classification: fasttext, charcnn, word level lstm "
        "For NLI: InferSent, ESIM, bert-base-uncased",
    )
    parser.add_argument(
        "--word_embeddings_path",
        type=str,
        default="",
        help="path to the word embeddings for the target model",
    )
    parser.add_argument(
        "--counter_fitting_embeddings_path",
        type=str,
        default="counter-fitted-vectors.txt",
        help="path to the counter-fitting embeddings we used to find synonyms",
    )
    parser.add_argument(
        "--counter_fitting_cos_sim_path",
        type=str,
        default="",
        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings",
    )


    parser.add_argument(
        "--synonym_num", default=50, type=int, help="Number of synonyms to extract"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size to get prediction"
    )
    parser.add_argument(
        "--data_size", default=1000, type=int, help="Data size to create adversaries"
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="max sequence length for BERT target model",
    )
    parser.add_argument(
        "--target_dataset", default="imdb", type=str, help="Dataset Name"
    )
    parser.add_argument("--fuzz", default=0, type=int, help="Word Pruning Value")
    parser.add_argument("--top_k_words", default=1000000, type=int, help="Top K Words")
    parser.add_argument("--allowed_qrs", default=1000000, type=int, help="Allowerd qrs")

    parser.add_argument("--k", default=0.9, type=float, help="k_margin")

    parser.add_argument("--bstart", default=0, type=int)

    parser.add_argument("--bend", default=100, type=int)
    
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    print("parser okk.")

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path, csvf=False)
    data = list(zip(texts, labels))
    data = data[: args.data_size]  # choose how many samples for adversary
    print("Data import finished!")


    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}
    sim_lis = []

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, "r") as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print(
            "Load pre-computed cosine similarity matrix from {}".format(
                args.counter_fitting_cos_sim_path
            )
        )
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        print("Start computing the cosine similarity matrix!")
        embeddings = []
        with open(args.counter_fitting_embeddings_path, "r") as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)

    print("Cos sim import finished!")

    # build the semantic similarity module
    # use = USE(args.USE_cache_path)

    stop_words_set = criteria.get_stopwords()
    print("Start attacking!")
    if "imdb" in args.dataset_path:
        models_name = {
            "roberta": "./ensemble/imdb/models/roberta-base-new",
            "bert": "./ensemble/imdb/models/bert-base-uncased-new",
            "albert": "./ensemble/imdb/models/albert-base-v2-new",
            "lstm_tradi": "./ensemble/imdb/models/lstm_tradition/model_state_dict.pt",
            "cnn_tradi": "./ensemble/imdb/models/cnn_tradition/model_state_dict.pt",
        }
    elif "ag" in args.dataset_path:
        models_name = {
            "roberta": "./ensemble/ag/models/roberta-base",
            "bert": "./ensemble/ag/models/bert-base-uncased",
            "albert": "./ensemble/ag/models/albert-base-v2",
            "lstm_tradi": "./ensemble/ag/models/lstm_tradition/model_state_dict.pt",
            "cnn_tradi": "./ensemble/ag/models/cnn_tradition/model_state_dict.pt",
        }
    elif "mr" in args.dataset_path:
        models_name = {
            "roberta": "./ensemble/mr/models/roberta-base-new",
            "bert": "./ensemble/mr/models/bert-base-uncased-new",
            "albert": "./ensemble/mr/models/albert-base-v2-new",
            "lstm_tradi": "./ensemble/mr/models/lstm_tradition/model_state_dict.pt",
            "cnn_tradi": "./ensemble/mr/models/cnn_tradition/model_state_dict.pt",
        }
    elif "yelp" in args.dataset_path:
        models_name = {
            "roberta": "./ensemble/yelp/models/roberta-base",
            "bert": "./ensemble/yelp/models/bert-base-uncased-new",
            "albert": "./ensemble/yelp/models/albert-base-v2-new-new",
            "lstm_tradi": "./ensemble/yelp/models/lstm_tradition/model_state_dict.pt",
            "cnn_tradi": "./ensemble/yelp/models/cnn_tradition/model_state_dict.pt",
        }
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    all = 0
    all_attack_g = 0
    all_qrs_g = 0
    ten_attack_g = 0
    ten_qrs_g = 0
    search_attack_g = [0,0,0,0,0,0,0,0,0,0]
    search_qrs_g = [0,0,0,0,0,0,0,0,0,0]

    import logging

    bstart = args.bstart
    bend = args.bend
    print(f"bstart: {bstart} bend:{bend}")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(filename=f"{save_path}/{bstart}.log", level=logging.INFO)
    victim_models = {args.target_model:models_name[args.target_model]}
    victim = VictimModel(args.target_dataset,victim_models,device,1)
    models_name.pop(args.target_model)
    # print(models_name)
    data = data[bstart:bend]
    atta_num = 0
    ensemble = Ensemble(models_name, device, args.k)
    for idx, (text, true_label) in enumerate(data):
        print(idx)
        print(len(text))
        print(" ".join(text)[:20])
        all_attack,all_qrs,ten_attack,ten_qrs,search_attack,search_qrs = attack(
            args.top_k_words,
            text_ls=text,
            true_label=true_label,
            victim=victim,
            word2idx=word2idx,
            idx2word=idx2word,
            cos_sim=sim_lis,
            ensemble=ensemble,
            device=device,
            arg_k=args.k,
            text_idx=idx + bstart,
            sim_score_window=args.sim_score_window,
            batch_size=args.batch_size,
            target_model = args.target_model,
            save_path=save_path
        )
        if all_attack is not None:
            all+=1
            all_attack_g += all_attack
            all_qrs_g += all_qrs
            ten_attack_g += ten_attack
            ten_qrs_g += ten_qrs
            for i in range(len(search_attack_g)):
                search_attack_g[i] += search_attack[i]
                search_qrs_g[i] += search_qrs[i]
            print(f"attacking {bstart+idx} all:{all} all succ:{all_attack_g} all qrs:{all_qrs_g} ten succ:{ten_attack_g} ten qrs:{ten_qrs_g} {'||'.join([str(g) for g in search_attack_g])} {'||'.join([str(g) for g in search_qrs_g])}")
            logging.info(f"attacking {bstart+idx} all:{all} all succ:{all_attack_g} all qrs:{all_qrs_g} ten succ:{ten_attack_g} ten qrs:{ten_qrs_g} {'||'.join([str(g) for g in search_attack_g])} {'||'.join([str(g) for g in search_qrs_g])}")
            print("\n\r")
            atta_num +=1

if __name__ == "__main__":
    main()


"""


python3 SEP_Attack.py \
        --dataset_path data/ag  \ choice:, data/mr, data/imdb, data/yelp
        --word_embeddings_path glove.6B.200d.txt \
        --target_model cnn_tradi \ choice: bert,lstm,cnn
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset ag \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --k 1\
        --save_path ensemble/ag/cnn/ \
        --bstart 0 \
        --bend 500

        

        

        

        

"""
