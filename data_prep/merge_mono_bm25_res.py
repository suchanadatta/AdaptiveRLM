import csv
import pandas as pd

orig_df = pd.read_csv('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/optimize.trecdl.monot5.sorted', delimiter='\t')
rerank_df = pd.read_csv('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/dl_monot5-base_d1000.res.reranked.sorted', delimiter='\t')
merge_df = pd.read_csv('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/merge', delimiter='\t')
rm_dup = csv.reader(open('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/merge.norm.sort'), delimiter='\t')

uniq_qids = list(set(orig_df['qid']))
print(uniq_qids)

for id in uniq_qids:
    sliced_df = orig_df.loc[(orig_df['qid'] == id), ['qid', 'docid', 'q0', 'rank', 'score', 'run']]
    print(sliced_df)

    qid_score_list = sliced_df['score']
    print(qid_score_list)

    qid_max_score = max(qid_score_list)
    qid_min_score = min(qid_score_list)
    print(qid_max_score)
    print(qid_min_score)

    sliced_df['norm'] = [(x - qid_min_score) / (qid_max_score - qid_min_score) for x in qid_score_list]
    print(sliced_df)

    sliced_df.to_csv('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/dl_monot5-base_d1000.res.norm', header=None, index=None, sep='\t', mode='a')

for id in uniq_qids:
    sliced_df = rerank_df.loc[(rerank_df['qid'] == id), ['qid', 'docid', 'q0', 'rank', 'score', 'run']]
    print(sliced_df)

    qid_score_list = sliced_df['score']
    print(qid_score_list)

    qid_max_score = max(qid_score_list)
    qid_min_score = min(qid_score_list)
    print(qid_max_score)
    print(qid_min_score)

    sliced_df['norm'] = [(x - qid_min_score) / (qid_max_score - qid_min_score) for x in qid_score_list]
    print(sliced_df)

    sliced_df.to_csv('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/dl_monot5-base_d1000.res.reranked.sorted.norm', header=None, index=None, sep='\t', mode='a')

for id in uniq_qids:
    sliced_df = merge_df.loc[(merge_df['qid'] == id), ['qid', 'docid', 'score', 'run']]
    print(sliced_df)

    qid_score_list = sliced_df['score']
    print(qid_score_list)

    qid_max_score = max(qid_score_list)
    qid_min_score = min(qid_score_list)
    print(qid_max_score)
    print(qid_min_score)

    sliced_df['norm'] = [(x - qid_min_score) / (qid_max_score - qid_min_score) for x in qid_score_list]
    print(sliced_df)

    sliced_df.to_csv('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/merge.norm', header=None, index=None, sep='\t', mode='a')

# remove duplicates
f = open('/store/adaptive_feedback/exp_trecdl/optimize_monot5_rlm/rerank_rlm_with_t5/merge.norm.sort.nodup', 'w')
qid = ''
docid = ''
for line in rm_dup:
    if qid == '' or line[0] != qid:
        f.writelines(line[0] + '\tQ0\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\n')
        qid = line[0]
        docid = line[1]
    elif line[0] == qid and line[1] == docid:
        continue
    elif line[0] == qid and line[1] != docid:
        f.writelines(line[0] + '\tQ0\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\n')
        qid = line[0]
        docid = line[1]





