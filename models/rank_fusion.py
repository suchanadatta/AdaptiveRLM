import os
import argparse
import csv
import pandas as pd
import operator

class QueryInstance:
    def __init__(self, sigmoid, init_df, rlm_df):
        self.sigmoid = sigmoid
        self.init_df = init_df
        self.rlm_df = rlm_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-list', default='/home/suchana/Desktop/rank_list_1')
    parser.add_argument('--rlm-list', default='/home/suchana/Desktop/rank_list_2')
    parser.add_argument('--sig-pred', default='/home/suchana/Desktop/pred')
    parser.add_argument('--fusion-list', default='/home/suchana/Desktop/fusion')
    args = parser.parse_args()

    def prepend_line(file_name, line):
        """ Insert given string as a new line at the beginning of a file """
        dummy_file = file_name + '.dum'
        with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
            write_obj.write(line + '\n')
            for line in read_obj:
                write_obj.write(line)
        os.remove(file_name)
        os.rename(dummy_file, file_name)

    # append this line at the beginning of the res file to create panda series
    # append_line = 'qid\tq0\tdocid\trank\tscore\trunid'
    # prepend_line(args.init_list, append_line)
    # prepend_line(args.rlm_list, append_line)

    # read init and rlm res files as panda series
    init_df = pd.read_csv(args.init_list, delimiter='\t')
    rlm_df = pd.read_csv(args.rlm_list, delimiter='\t')

    # create qid,sigmoid prediction dict
    qid_sig_dict = {}
    file_read = csv.reader(open(args.sig_pred), delimiter='\t')
    for line in file_read:
        qid_sig_dict[line[0]] = line[1]

    # create data instances keyed by qid
    instance_obj_list = {}
    for qid, pred in qid_sig_dict.items():
        sliced_init = init_df.loc[(init_df['qid'] == int(qid)), ['qid', 'q0', 'docid', 'rank', 'score', 'runid']]
        sliced_rlm = rlm_df.loc[(rlm_df['qid'] == int(qid)), ['qid', 'q0', 'docid', 'rank', 'score', 'runid']]
        init_docid = list(sliced_init['docid'])
        rlm_docid = list(sliced_rlm['docid'])
        instance_obj_list[qid] = QueryInstance(float(pred), init_docid, rlm_docid)

    # for each query; do for each docid; do (1-\theta(Q)) * L(Q) + \theta(Q) * L(Q_E)
    # where \theta(Q) = sigmoid value
    adaptive_file = open(args.fusion_list, 'a')
    res_buff = ''
    for qid, instance in instance_obj_list.items():
        per_doc_rank = {}
        union_docid = set(instance.init_df + instance.rlm_df)
        for docid in union_docid:
            fuse_rank = 0
            if docid in instance.init_df:
                fuse_rank += (1-instance.sigmoid) / (instance.init_df.index(docid) + 1)
            else: fuse_rank += (1-instance.sigmoid) / 1500
            if docid in instance.rlm_df:
                fuse_rank += instance.sigmoid / (instance.rlm_df.index(docid) + 1)
            else: fuse_rank += instance.sigmoid / 1500
            per_doc_rank[docid] = round(fuse_rank, 5)
        # print(per_doc_rank)
        sorted_rank = dict( sorted(per_doc_rank.items(), key=operator.itemgetter(1),reverse=True))
        # print(sorted_rank)
        rank = 1
        cutoff = 100
        for docid, score in sorted_rank.items():
            if rank <= cutoff:
                res_buff += str(qid) + '\tQ0\t' + str(docid) + '\t' + str(rank) + '\t' + str(score) + '\trunid\n'
                rank+= 1
        adaptive_file.write(res_buff)
        res_buff = ''

if __name__ == '__main__':
    main()
