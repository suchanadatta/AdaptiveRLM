import csv

qrel_file = open('/store/collection/ms-marco/trec_DL/pass_2020.qrels', 'r')
qrels = csv.reader(qrel_file, delimiter='\t')
qrel_stat = open('trecdl-20_bertpl.gt', 'w')

qid = ""
count = 0
for line in qrels:
    if qid == "" or line[0] == qid:
        qid = line[0]
        if int(line[3]) > 0:
            count = count + 1
    elif line[0] != qid:
        qrel_stat.writelines(qid + '\t' + str(count) + '\n')
        count = 0
        qid = line[0]
        if int(line[3]) > 0:
            count = count + 1
        # qrel_stat.writelines(line[0] + '\t' + count)
        # count = count + 1
qrel_stat.writelines(qid + '\t' + str(count))
qrel_stat.close()