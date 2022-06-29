from pyterrier_pisa import PisaIndex
import csv
csv.field_size_limit(100000000) # to increase csv limit

# # any iterator
def iter_docs():
  # read from tsv file
  col_file = open('/store/collection/trec678rb/trec678_not_analyzed.dump', 'r')
  col_read = csv.reader(col_file, delimiter='\t')
  for line in col_read:
      print(line[0])
      yield {'docno': line[0], 'text': line[1]}
index = PisaIndex('/store/index/trecrb_not_analyzed.pisa', overwrite=True)
index.index(iter_docs())

# test retrieval
# index = PisaIndex('/store/index/trecrb_not_analyzed.pisa')
# bm25 = index.bm25(k1=1.2, b=0.4)
# res = [r.docno for r in bm25.search('human').itertuples(index=True)]
# # texts = {d.doc_id: d.text for d in dataset_train.docs.lookup(list(curr_window)).values()}
# print(res)
