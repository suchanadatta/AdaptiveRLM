# prepare input for model-aware-qpp
# inputs = two query IDs (say q1 (original query), q2 (expanded query))
# output = binary class label - 1/0 (1 -> q1>q2; 0 -> q1<q2)
# output format - q1 \t q2 \t class_label

import sys

if len(sys.argv) < 4:
    print('Needs 3 arguments - <original qid - AP file>'
          ' <expanded query - AP file> <output file path>')
    exit(0)

arg_qid_ap_file = sys.argv[1]
arg_expand_ap_file = sys.argv[2]
arg_res_file_path = sys.argv[3]

res_file = open(arg_res_file_path + 'test_input.pairs', 'w')

temp_dict = {}
# qid_ap_dict = {}
# expand_ap_dict = {}

def make_qid_ap_dict(ap_file):
    fp = open(ap_file)
    for line in fp.readlines():
        parts = line.rstrip().split('\t')
        temp_dict[parts[0]] = parts[1]

make_qid_ap_dict(arg_qid_ap_file)
qid_ap_dict = temp_dict.copy()
print('initial : ', qid_ap_dict)
make_qid_ap_dict(arg_expand_ap_file)
expand_ap_dict = temp_dict.copy()
print('expanded : ', expand_ap_dict)

for qid, ap in qid_ap_dict.items():
    if float(ap) > float(expand_ap_dict[qid]):
        res_file.writelines(qid + '\t' + ap + '\t' + qid + '\t' + expand_ap_dict[qid] + '\t1\n')
    else:
        res_file.writelines(qid + '\t' + ap + '\t' + qid + '\t' + expand_ap_dict[qid] + '\t0\n')
res_file.close()
