import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import ir_datasets
import random
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import logging
import itertools
import torch.nn.functional as F
import argparse
import more_itertools
import numpy as np
import os
from scipy import stats

torch.manual_seed(0)
logger = ir_datasets.log.easy()

def position_encoding_init(max_pos, emb_dim):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(max_pos)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class BertQppModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.emb_dim = 768
        self.max_pos = 1000
        self.position_enc = torch.nn.Embedding(self.max_pos, self.emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(self.max_pos, self.emb_dim)
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = torch.nn.LSTM(input_size=self.emb_dim, hidden_size=self.bert.config.hidden_size,
                                  num_layers=1, bias=True, batch_first=False, dropout=0.2)
        self.utility = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 100),
        )
        self.utility_pair = torch.nn.Sequential(
            torch.nn.Linear(200, 10),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, pos_list, input_ids, attention_mask, token_type_ids):
        res = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state  # [BATCH, LEN, DIM]
        res = res[:, 0]  # get CLS token rep [BATCH, DIM]
        # combine with position
        res = res + self.position_enc(torch.tensor([pos for pos in pos_list], dtype=torch.long))  # [BATCH, DIM]
        res = res.unsqueeze(1)  # [BATCH, 1, DIM]
        lstm_output = self.lstm(res)[0].squeeze(1)  # [BATCH, DIM]
        # return self.utility(lstm_output).reshape(input_ids.shape[0])
        return self.utility(lstm_output)

class PairedInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid = l[0].strip()
            self.q_init = l[1].strip()
            self.q_rlm = l[2].strip()
            self.class_label = int(l[3].strip())
        else:
            self.qid = l[0].strip()
            self.q_init = l[1].strip()
            self.q_rlm = l[2].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='/store/index/msmarco-passage.pisa')
    parser.add_argument('--train-query', default='/store/adaptive_feedback/exp_trecdl/bert_pair_train.input')
    parser.add_argument('--test-query', default='/store/adaptive_feedback/exp_trecdl/bert_pair_test.input')
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--train-its', default=1000, type=int)
    parser.add_argument('--chunk-per-query', default=20, type=int)
    parser.add_argument('--docs-per-query', default=10, type=int)
    parser.add_argument('--outfile', default='/store/adaptive_feedback/exp_trecdl/bert_pair.res')
    parser.add_argument('--skip-utility', action='store_true')
    parser.add_argument('--skip-norel', action='store_true')
    args = parser.parse_args()

    index = PisaIndex(args.index)
    bm25 = index.bm25()
    dataset_train = ir_datasets.load('msmarco-passage/train')
    # load train queries
    train_qlist = []
    with open(args.train_query) as f:
        content = f.readlines()
    for line in content:
        instance = PairedInstance(line)
        train_qlist.append(instance)
    print('Total train queries : ', len(train_qlist))
    # load test queries
    test_qlist = []
    with open(args.test_query) as f:
        content = f.readlines()
    for line in content:
        instance = PairedInstance(line)
        test_qlist.append(instance)
    print('Total test queries : ', len(test_qlist))

    def _build_input_init(queries, opt):
        while True:
            for query in queries:
                print('\nOriginal query : ', query.q_init)
                res = [r.docno for r in bm25.search(query.q_init).itertuples(index=False)]  # retrieve a list of documents; store docids
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < args.chunk_per_query:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        texts = {d.doc_id: d.text for d in dataset_train.docs.lookup(list(curr_window)).values()}
                        win_count += 1
                        if opt == 'train':
                            yield query.q_init, [texts[did] for did in list(curr_window)], query.class_label, pos_hit
                        else:
                            yield query.q_init, [texts[did] for did in list(curr_window)], pos_hit

    def _build_input_rlm(queries, opt):
        while True:
            for query in queries:
                print('\nFeedback query : ', query.q_rlm)
                res = [r.docno for r in bm25.search(query.q_rlm).itertuples(index=False)]  # retrieve a list of documents; store docids
                res_window = list(more_itertools.windowed(res, n=args.batch_size, step=args.batch_size))
                win_count = 0
                for curr_window in res_window:
                    if win_count < args.chunk_per_query:
                        pos_hit = [win_count * args.batch_size + i for i in range(0, args.batch_size)]
                        texts = {d.doc_id: d.text for d in dataset_train.docs.lookup(list(curr_window)).values()}
                        win_count += 1
                        if opt == 'train':
                            yield query.q_rlm, [texts[did] for did in list(curr_window)], query.class_label, pos_hit
                        else:
                            yield query.q_rlm, [texts[did] for did in list(curr_window)], pos_hit

    # training
    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertQppModel()
    train_init_iter = _build_input_init(train_qlist, 'train')
    train_rlm_iter = _build_input_rlm(train_qlist, 'train')
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    mse = torch.nn.MSELoss()
    suffixes = []
    model_no = 0
    u_loss = 0
    utl_loss = []
    if args.skip_utility:
        suffixes.append('noutil')
    if args.skip_norel:
        suffixes.append('norel')
    suffixes.append('{}')
    model_name = f'models/model-{"-".join(suffixes)}.pt'
    with logger.pbar_raw(total=args.train_its, ncols=200) as pbar:
        # train the model
        # for train_i in range(len(queries_train) * args.chunk_per_query):
        for train_i in range(args.train_its):
            query_init, docs_init, label_init, poshit_init = next(train_init_iter)
            inputs_train_init = tokeniser([query_init for _ in docs_init], [doc for doc in docs_init], padding=True,
                                     truncation='only_second',
                                     return_tensors='pt')
            query_rlm, docs_rlm, label_rlm, poshit_rlm = next(train_rlm_iter)
            inputs_train_rlm = tokeniser([query_rlm for _ in docs_rlm], [doc for doc in docs_rlm], padding=True,
                                          truncation='only_second',
                                          return_tensors='pt')
            utility_init = model(poshit_init, **{k: v for k, v in inputs_train_init.items()})
            # print('UTILITY-1 : ', utility_init)
            utility_rlm = model(poshit_rlm, **{k: v for k, v in inputs_train_rlm.items()})
            # print('UTILITY-2 : ', utility_rlm)
            utility_merge = torch.cat((utility_init, utility_rlm), dim=-1)
            # print('merge : ', utility_merge)
            utility_merge = model.utility_pair(utility_merge)
            print('sig : ', utility_merge)
            u_loss += mse(utility_merge, torch.tensor([label_init], dtype=torch.float32))
            print('LOSS : ', u_loss)
            u_loss.backward()
            optim.step()
            optim.zero_grad()
            if u_loss.cpu().detach().item() != 0:
                utl_loss.append(u_loss.cpu().detach().item())
                print('UTL LoSS : ', utl_loss)
            u_loss = 0
            pbar.set_postfix(
                {'avg_utl_loss': sum(utl_loss) / len(utl_loss),
                 'recent_utl_loss': sum(utl_loss[-100:]) / len(utl_loss[-100:])})
            pbar.update(1)
            if train_i % 2000 == 0:
                model_no = train_i
                print(pbar)
                torch.save(model.state_dict(), model_name.format(train_i))

    # evaluation
    adaptive_file = open(args.outfile, 'a')
    model.load_state_dict(torch.load('./models/model-'+model_no+'.pt'))
    model.eval()
    test_init_iter = _build_input_init(test_qlist, 'test')
    test_rlm_iter = _build_input_rlm(test_qlist, 'test')
    pred_out = ''
    log = 0
    for test_i in range(len(test_qlist) * args.chunk_per_query):
        log += 1
        query_init, docs_init, label_init, poshit_init = next(test_init_iter)
        inputs_test_init = tokeniser([query_init for _ in docs_init], [doc for doc in docs_init], padding=True,
                                    truncation='only_second',
                                    return_tensors='pt')
        query_rlm, docs_rlm, label_rlm, poshit_rlm = next(test_rlm_iter)
        inputs_test_rlm = tokeniser([query_rlm for _ in docs_rlm], [doc for doc in docs_rlm], padding=True,
                                    truncation='only_second',
                                    return_tensors='pt')
        utility_init = model(poshit_init, **{k: v for k, v in inputs_test_init.items()})
        utility_rlm = model(poshit_rlm, **{k: v for k, v in inputs_test_rlm.items()})
        utility_merge = torch.cat((utility_init, utility_rlm), dim=-1)
        utility_merge = model.utility_pair(utility_merge)
        print('sig : ', utility_merge)
        predicted = utility_merge.data.numpy()
        for x in test_qlist:
            if predicted[c] >= 0.5:
                pred_out += str(x.qid) + '\t' + str(predicted[c] + '\t' + '1\n')
            else:
                pred_out += str(x.qid) + '\t' + str(predicted[c] + '\t' + '0\n')
    adaptive_file.write(pred_out)

    # measure accuracy
    gt_file = np.genfromtxt('/store/adaptive_feedback/exp_trecdl/gt/dl20_pair.gt', delimiter='\t')
    actual = gt_file[:, 2:]
    predict_file = np.genfromtxt('/store/adaptive_feedback/exp_trecdl/trecdl20.pred', delimiter='\t')
    predict = predict_file[:, 2:]
    score = accuracy_score(actual, predict)
    print('Accuracy : ', round(score, 4))

    # t-test
    file = np.genfromtxt('/home/suchana/Desktop/foo', delimiter='\t')
    diff1 = file[:, 0]
    diff2 = file[:, 1]
    # print(stats.ttest_ind(diff1, diff2)) # individual test
    print(stats.ttest_rel(diff1, diff2))  # paired test

if __name__ == '__main__':
    main()