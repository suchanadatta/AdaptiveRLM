#
# MS MARCO data to single file transformer
#
# Input:  Directory of raw msmarco corpus data, optional: "doStem" as 2nd argument if words should be stemmed
# Output: single file in ../data/msmarco_corpus.txt
#         contents: id1 text text text
#                   id2 text text
#         (text contains no newlines, id can contain special chars except whitespaces)
#

import os
import timeit
import sys
import re
import codecs
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 3:
    print('Needs 2 arguments - 1. the msmarco data directory path! 2. doStem')
    exit(0)

cleanTextRegex = re.compile('[^a-zA-Z]')
cleanHtmlRegex = re.compile('<[^<]+?>')

docCount = 0
stemmer = PorterStemmer()
doStem = len(sys.argv) == 3 and sys.argv[2] == 'doStem'

count = 0
start_time = timeit.default_timer()
outFileName = '/store/adaptive_feedback/word_vector/msmarco_unstem.txt'
if doStem:
    outFileName = '/store/adaptive_feedback/word_vector/msmarco_stemmed.txt'

with open(outFileName, 'w') as outputFile:
    with codecs.open(sys.argv[1], "r", "iso-8859-1") as f:
        contents = f.readlines()
        for line in contents:

            # ignore empty lines
            if line.isspace():
                continue
            new_line = line.replace('\t', ' ')
            # clean the html tags out
            parsed = cleanHtmlRegex.sub(' ', new_line)
            # clean non text characters
            parsed = cleanTextRegex.sub(' ', parsed)
            # clean whitespaces + lower words + concat again
            wordList = []
            for w in parsed.split(' '):
                if w:
                    if doStem:
                        cleaned = stemmer.stem(w.lower().strip())
                    else:
                        cleaned = w.lower().strip()
                    wordList.append(cleaned)
            outputText = ' '.join(wordList)
            outputText = remove_stopwords(outputText)
            count = count + 1
            if count % 10 == 0:
                print('Completed ', count, ' files, time:', timeit.default_timer() - start_time)
            # write single line output
            outputFile.write(outputText)
            outputFile.write('\n')
        outputFile.flush()

print('\n-------\n', 'Completed all ', count-1, ' files, time: ', timeit.default_timer() - start_time)
