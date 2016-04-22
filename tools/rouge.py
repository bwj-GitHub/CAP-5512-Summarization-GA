"""
Created on Apr 13, 2016

@author: brandon
"""


import nltk


# Rouge-2 EXCEPT that bigram counts are all Binary
def rouge2_bin(candidate_bigrams, ref_bigrams):
    # ref_bigrams is a list of sets of bigrams for each ref.
    # candidate bigrams is a set of all bigrams in the candidate

    numerator = 0
    denom = 0
    for S in ref_bigrams:
        for bg in S:
            if bg in candidate_bigrams:
                numerator += 1
            denom += 1
            
    numerator += 0.0
    denom += 0.0
    return numerator/denom


if __name__ == "__main__":
    c1 = nltk.word_tokenize("there was a cat that wore a hat")
    c2 = nltk.word_tokenize("something about a cat")
    bc1 = set(nltk.bigrams(c1))
    bc2 = set(nltk.bigrams(c2))
    
    sum1 = nltk.word_tokenize("there was once a cat in a hat")
    bsum1 = set(nltk.bigrams(sum1))
    sum2 = nltk.word_tokenize("a while ago, there was a cat that wore a hat")
    bsum2 = set(nltk.bigrams(sum2))
    bsums = [bsum1, bsum2]
    print(rouge2_bin(bc1, bsums))
    print(rouge2_bin(bc2, bsums))
