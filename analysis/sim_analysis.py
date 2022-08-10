import numpy as np
import csv
import itertools

gt_class_0 = []
gt_class_1 = []
paired_gt_file = open('/store/adaptive_feedback/sample_run_trec/gt/train_input.pairs', 'r')
paired_gt = csv.reader(paired_gt_file, delimiter='\t')
for line in paired_gt:
    if line[2] == str(0):
        gt_class_0.append(str(line[1]))
    else:
        gt_class_1.append(str(line[1]))
print('class-0 instances : ', len(gt_class_0))
print('class-1 instances : ', len(gt_class_1))

merge_layer_dict = {}
merge_layer_vec = open('/store/adaptive_feedback/sample_run_trec/analysis/submodel_epoch-1.train', 'r')
merged_layer = csv.reader(merge_layer_vec, delimiter=' ')
for line in merged_layer :
    merge_layer_dict[str(line[0])] = np.genfromtxt(line[2:], delimiter=" ")
# print('merged layer dict : ', merge_layer_dict)

pos = 0
cosim_0_0 = 0.0
while pos < 2:
    print(merge_layer_dict.get(str(gt_class_0[pos])))
    pos+=1
    # print(merge_layer_dict[gt_class_0[pos]])