"""
Created on Apr 20, 2016

@author: brandon
"""


import pickle

from tools.mmr import MMR
from ga.chromo import SummaryChromo


def sum_with_chromo(summarizer, raw_sents, vec_sents, n_words=100):
    # raw/vec_sents should be the entire dict

    for doc_id in raw_sents.keys():
        print("\n\n" + doc_id)

        summary = summarizer.summarize(raw_sents[doc_id],
                                       vec_sents[doc_id],
                                       n_words)
        
#         imp_scores = summarizer.most_recent_imp_scores
#         i = 0
#         for s in raw_sents[doc_id]:
#             print(str(imp_scores[i])[2:-2] + "\t" + " ".join(s))
#             i += 1

        # Write summary to file
        with open("../Output/Summaries/TLDR137222937/" + doc_id + "t.TLDR137222937", "w") as fp:
            for line in summary:
                fp.write(" ".join(line) + "\n")


def sum_with_mmr(raw_sents, n_words=100):
    for doc_id in raw_sents.keys():
        temp_sents = raw_sents[doc_id]
        sents = []
        for s in temp_sents:
            sents.append(" ".join(s))

        M = MMR(sents)
        summary = M.summarize(words=100, l=.5)
        
        # Write summary to file:
        with open("../Output/Summaries/MMR/" + doc_id+"t.MMR", "w") as fp:
            for line in summary:
                fp.write(" ".join(line) + "\n")

# Summarize ALL meta_docs:
with open("../Input/tldr_1-40.p", "rb") as fp:
    raw_sents, vec_sents = pickle.load(fp)
# with open("../Input/tldr_42-1050.p", "rb") as fp:
#     raw_sents, vec_sents = pickle.load(fp)
    
# pickle_name = "../Input/tldr137222937.p"
# with open(pickle_name, "rb") as fp:
#     sumr = pickle.load(fp)

# Test random chromo
# sumr = SummaryChromo()


# sum_with_chromo(sumr, raw_sents, vec_sents, n_words=100)


sum_with_mmr(raw_sents, 100)
