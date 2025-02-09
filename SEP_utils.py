from typing import Any
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe

import torch.nn.functional as F
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel
from collections import OrderedDict
import numpy as np
import ensemble



class TradiTokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        self.glove = GloVe(
            name="6B", cache="<path-to-find-glove>", dim=300
        )

    def __call__(self, texts) -> Any:
        xs = []
        for text in texts:
            t = self.tokenizer(text)
            x = [self.glove.stoi[w] for w in t if w in self.glove.stoi]
            x = x[:256]
            xs.append(torch.tensor(x))

        x_pad = pad_sequence(xs, batch_first=True)

        return x_pad


class my_config:
    max_length = 128  
    batch_size = 512  
    embedding_size = 300  
    num_layers = 1  
    dropout = 0.5  
    output_size = 4  
    lr = 0.001  
    epochs = 20  
    glove = None
    patience = 10
    warm_up_steps = 3
    num_filters = 100
    filter_sizes = [3, 4, 5]


import math

class VictimModel:
    def __init__(self,target_dataset, models_name, device, k):

            self.models_name = models_name
            self.tokenizers = OrderedDict()
            self.models = OrderedDict()
            self.device = device
            self.target_dataset = target_dataset
            self._parse_models()
            self.k = k
            print(self.models)
    
    def _parse_models(self):
        if "ag" == self.target_dataset:
            config = my_config()
            for k, v in self.models_name.items():
                if k in ["lstm_tradi", "cnn_tradi"]:
                    self.tokenizers[k] = TradiTokenizer()
                    # todo: 继续写
                    config.glove = self.tokenizers[k].glove
                    if k == "lstm_tradi":
                        self.models[k] = ensemble.ag.LSTMTradi(config)
                        # print(v)
                        # print(self.models[k])
                        self.models[k].load_state_dict(torch.load(v))
                        self.models[k] = self.models[k].to(self.device)
                    else:
                        self.models[k] = ensemble.ag.TextCNNTradi(config)
                        self.models[k].load_state_dict(torch.load(v))
                        self.models[k] = self.models[k].to(self.device)
                    continue

                self.tokenizers[k] = AutoTokenizer.from_pretrained(v)
                self.models[k] = AutoModelForSequenceClassification.from_pretrained(
                    v
                ).to(self.device)
        elif "imdb" == self.target_dataset or "mr" == self.target_dataset or "yelp" == self.target_dataset:
            config = my_config()
            config.output_size = 2
            for k, v in self.models_name.items():
                if k in ["lstm_tradi", "cnn_tradi"]:
                    self.tokenizers[k] = TradiTokenizer()
                    # todo: 继续写
                    config.glove = self.tokenizers[k].glove
                    if k == "lstm_tradi":
                        self.models[k] = ensemble.imdb.LSTMTradi(config)
                        # print(v)
                        # print(self.models[k])
                        self.models[k].load_state_dict(torch.load(v))
                        self.models[k] = self.models[k].to(self.device)
                    else:
                        self.models[k] = ensemble.imdb.TextCNNTradi(config)
                        self.models[k].load_state_dict(torch.load(v))
                        self.models[k] = self.models[k].to(self.device)
                    continue
                print(v)
                self.tokenizers[k] = AutoTokenizer.from_pretrained(v)
                self.models[k] = AutoModelForSequenceClassification.from_pretrained(
                    v
                ).to(self.device)  
                
                
    def __call__(self, texts, orig_label,hash_qrs) -> Any:
        """
        ad_text: List[str]
        """
        scores_ls = []
        k_margin = (
            torch.tensor([-self.k] * len(texts)).unsqueeze(dim=-1).to(self.device)
        )
        tradi_tk = None
        for idx, (mn, model) in enumerate(self.models.items()):
            if mn in ["lstm_tradi", "cnn_tradi"]:
                # todo: continue
                if tradi_tk is None:
                    tks = self.tokenizers[mn](texts)
                    tradi_tk = tks.to(self.device)
                model.eval()
                with torch.no_grad():
                    logits = model(tradi_tk)
                    
            else:
                tks = self.tokenizers[mn](
                    texts, return_tensors="pt", truncation=True, padding=True
                )
                for k, v in tks.items():
                    tks[k] = tks[k].to(self.device)
                model.eval()
                # logits [len(ad_text),output_dim]
                with torch.no_grad():
                    logits = model(**tks).logits
                    

        pred_label = torch.argmax(logits,dim=-1).squeeze()
        
        return pred_label!=orig_label


class Ensemble:
    def __init__(self, models_name, device, k):
        self.models_name = models_name
        self.tokenizers = OrderedDict()
        self.models = OrderedDict()
        self.device = device
        self._parse_models()
        self.k = k

    def test_acc(self, data):
        for k, model in self.models.items():

            acc = 0
            all = 0
            model.eval()
            # pdb.set_trace()
            print(k)
            for idx, (text, true_label) in enumerate(data):
                all += 1
                tk = self.tokenizers[k](
                    " ".join(text), return_tensors="pt", truncation=True, max_length=512
                )
                for tk_key in tk.keys():
                    tk[tk_key] = tk[tk_key].to(self.device)
                if k in ["lstm", "textcnn"]:
                    logits = model(tk["input_ids"])
                else:
                    logits = model(**tk).logits
                # pdb.set_trace()
                if true_label == torch.argmax(logits):
                    acc += 1
            # self.weights.append(acc / all)
            print(k, acc / all)

    def init_weights(self):
        self.weights_zoo = self._generating_weights(
            len(self.models_name), max_nums=100, keep_nums=len(self.models_name)
        ).to(self.device)
        self.weights = None
        self.cur_w = -1
        self.len_weights = self.weights_zoo.size()[0]

    def next_weights(self):
        self.cur_w += 1
        self.weights = self.weights_zoo[self.cur_w]
        self.weights
        return self.weights

    # 行列式点阵过程
    def _fast_map_dpp(self, kernel_matrix, max_length):
        cis = np.zeros((max_length, kernel_matrix.shape[0]))
        di2s = np.copy(np.diag(kernel_matrix))
        selected = np.argmax(di2s)
        selected_items = [selected]
        while len(selected_items) < max_length:
            idx = len(selected_items) - 1
            ci_optimal = cis[:idx, selected]
            di_optimal = math.sqrt(di2s[selected])
            elements = kernel_matrix[selected, :]
            eis = (elements - ci_optimal @ cis[:idx, :]) / di_optimal
            cis[idx, :] = eis
            di2s -= np.square(eis)
            di2s[selected] = -np.inf
            selected = np.argmax(di2s)
            selected_items.append(selected)
        return selected_items

    def _generating_random_weights(self, weight_size, max_nums=100):
        weights = torch.empty(max_nums, weight_size).uniform_(
            0, 1
        )  
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  
        return weights

    def _generating_weights(self, weight_size, max_nums, keep_nums):

        weights = self._generating_random_weights(weight_size, max_nums=max_nums)
        kernel = weights @ weights.T
        kernel = kernel.numpy()
        selected_items = self._fast_map_dpp(kernel, keep_nums)

        return weights[selected_items, :]

    def _parse_models(self):
        if "ag" in self.models_name["albert"]:
            
            config = my_config()
            for k, v in self.models_name.items():
                if k in ["lstm", "textcnn"]:
                    self.tokenizers[k] = AutoTokenizer.from_pretrained(
                        self.models_name["bert"]
                    )
                    if k == "lstm":
                        print(v)
                        self.models[k] = ensemble.ag.LSTMClassifier(len(self.tokenizers[k]), 256, 4)
                        self.models[k].load_state_dict(torch.load(v))
                        self.models[k] = self.models[k].to(self.device)
                    else:
                        self.models[k] = ensemble.ag.TextCNN(
                            len(self.tokenizers[k]), 256, 100, [3, 4, 5], 4
                        )
                        self.models[k].load_state_dict(torch.load(v))
                        self.models[k] = self.models[k].to(self.device)

                    continue
                elif k in ["lstm_tradi", "cnn_tradi"]:
                    self.tokenizers[k] = TradiTokenizer()
                    # todo: 继续写
                    config.glove = self.tokenizers[k].glove
                    if k == "lstm_tradi":
                        self.models[k] = ensemble.ag.LSTMTradi(config)
                        # print(v)
                        # print(self.models[k])
                        self.models[k].load_state_dict(torch.load(v,map_location=torch.device('cpu')))
                        self.models[k] = self.models[k].to(self.device)
                    else:
                        self.models[k] = ensemble.ag.TextCNNTradi(config)
                        self.models[k].load_state_dict(torch.load(v,map_location=torch.device('cpu')))
                        self.models[k] = self.models[k].to(self.device)
                    continue

                self.tokenizers[k] = AutoTokenizer.from_pretrained(v)
                self.models[k] = AutoModelForSequenceClassification.from_pretrained(
                    v
                ).to(self.device)
        elif "imdb" in self.models_name["albert"] or "mr" in self.models_name["albert"] or "yelp" in self.models_name["albert"]:
            config = my_config()
            config.max_length = 256
            if "mr" in self.models_name["albert"]:
                config.max_length = 128
            config.output_size = 2
            print(config.output_size)
            for k, v in self.models_name.items():
                if k in ["lstm_tradi", "cnn_tradi"]:
                    self.tokenizers[k] = TradiTokenizer()
                    # todo: 继续写
                    config.glove = self.tokenizers[k].glove
                    print(v)
                    if k == "lstm_tradi":
                        self.models[k] = ensemble.imdb.LSTMTradi(config)
                        # print(v)
                        # print(self.models[k])
                        self.models[k].load_state_dict(torch.load(v,map_location=torch.device('cpu')))
                        self.models[k] = self.models[k].to(self.device)
                    else:
                        self.models[k] = ensemble.imdb.TextCNNTradi(config)
                        print("报错的地方",v)
                        self.models[k].load_state_dict(torch.load(v,map_location=torch.device('cpu')))
                        self.models[k] = self.models[k].to(self.device)
                    continue
                print(v)
                self.tokenizers[k] = AutoTokenizer.from_pretrained(v)
                self.models[k] = AutoModelForSequenceClassification.from_pretrained(
                v
                ).to(self.device)

    def __call__(self, texts, orig_label, flag=False, unique=False) -> Any:
        """
        ad_text: List[str]
        """
        if self.weights is None:
            exit(0)

        scores_ls = []
        k_margin = (
            torch.tensor([-self.k] * len(texts)).unsqueeze(dim=-1).to(self.device)
        )
        tradi_tk = None
        for idx, (mn, model) in enumerate(self.models.items()):
            if mn in ["lstm", "textcnn"]:
                tks = self.tokenizers[mn](
                    texts, return_tensors="pt", truncation=True, padding=True
                )
                for k, v in tks.items():
                    tks[k] = tks[k].to(self.device)
                model.eval()
                # logits [len(ad_text),output_dim]
                with torch.no_grad():
                    logits = model(tks["input_ids"])
                    logits = torch.softmax(logits, -1)
                    onehot_logits = torch.zeros_like(logits)
                    onehot_logits[:, orig_label] = logits[:, orig_label]
                    other_logits = logits - onehot_logits
                    best_other_logits = torch.max(
                        other_logits, dim=-1
                    ).values.unsqueeze(-1)
                    ori_logits = logits[:, orig_label].unsqueeze(dim=-1)
                    # [bsz,1]
                    loss = torch.max(
                        torch.cat([ori_logits - best_other_logits, k_margin], dim=-1),
                        dim=-1,
                    ).values.unsqueeze(dim=-1)
                    scores_ls.append(loss)
            elif mn in ["lstm_tradi", "cnn_tradi"]:
                # todo: continue
                if tradi_tk is None:
                    tks = self.tokenizers[mn](texts)
                    tradi_tk = tks.to(self.device)
                model.eval()
                with torch.no_grad():
                    logits = model(tradi_tk)
                    logits = torch.softmax(logits, -1)
                    onehot_logits = torch.zeros_like(logits)
                    onehot_logits[:, orig_label] = logits[:, orig_label]
                    other_logits = logits - onehot_logits
                    best_other_logits = torch.max(
                        other_logits, dim=-1
                    ).values.unsqueeze(-1)
                    ori_logits = logits[:, orig_label].unsqueeze(dim=-1)
                    # [bsz,1]
                    loss = torch.max(
                        torch.cat([ori_logits - best_other_logits, k_margin], dim=-1),
                        dim=-1,
                    ).values.unsqueeze(dim=-1)
                    scores_ls.append(loss)
            else:
                tks = self.tokenizers[mn](
                    texts, return_tensors="pt", truncation=True, padding=True
                )
                for k, v in tks.items():
                    tks[k] = tks[k].to(self.device)
                model.eval()
                # logits [len(ad_text),output_dim]
                with torch.no_grad():
                    logits = model(**tks).logits
                    logits = torch.softmax(logits, -1)
                    onehot_logits = torch.zeros_like(logits)
                    onehot_logits[:, orig_label] = logits[:, orig_label]
                    other_logits = logits - onehot_logits
                    best_other_logits = torch.max(
                        other_logits, dim=-1
                    ).values.unsqueeze(-1)
                    ori_logits = logits[:, orig_label].unsqueeze(dim=-1)
                    # [bsz,1]
                    loss = torch.max(
                        torch.cat([ori_logits - best_other_logits, k_margin], dim=-1),
                        dim=-1,
                    ).values.unsqueeze(dim=-1)
                    scores_ls.append(loss)

        scores_ori = torch.cat(scores_ls, dim=-1).to(self.device)
        # [batchsz,]

        scores = scores_ori @ self.weights.T
        if unique:
            weights = torch.ones(self.weights.size()) / self.weights.size()[-1]
            weights = weights.to(self.device)
            scores2 = scores_ori @ weights.T

            return scores, scores2
        if not flag:
            return scores
        else:
            return scores, scores_ori
