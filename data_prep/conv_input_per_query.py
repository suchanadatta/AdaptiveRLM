# input file contains histogram for top retrieved n documents for each m train queries
# this program creates m different histogram files associated with each query

import csv
import os

DATADIR = '/store/causalIR/model-aware-qpp/'

full_hist_file = open(DATADIR + 'histograms/trec8_withidf_bin30_t100.hist', 'r')
read_hist_file = full_hist_file.readlines()
out_file_name = ''
per_query_hist_file = ''
qid = ''
flag = 1

for line in read_hist_file:
    parts = line.split(' ')
    print('PARTS : ', parts[0])
    if qid == '' or parts[0] == qid:
        qid = parts[0]
        out_file_name = DATADIR + 'exp_hist_with_idf/input_100/test_input/' + qid + '.hist'
        if flag == 1:
            per_query_hist_file = per_query_hist_file + line
            flag = 0
        else:
            per_query_hist_file = per_query_hist_file + line
    elif parts[0] != qid:
        qid = parts[0]
        with open(out_file_name, 'w') as out:
            out.write(per_query_hist_file)
            out.close()
            per_query_hist_file = ''
            per_query_hist_file = per_query_hist_file + line
with open(out_file_name, 'w') as out:
    out.write(per_query_hist_file)
    out.close()


