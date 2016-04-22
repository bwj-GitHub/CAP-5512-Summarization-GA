"""
Created on Apr 13, 2016

@author: brandon
"""


import json

from gensim.models.word2vec import Word2Vec

from tools.tokenization import Tokenizer


def model_yelp_dataset(filename, size=200):
    doc_it = yelp_review_text_iterator(filename)
    # FIXME: using list is not efficient, but generators are not allowed :(
    return Word2Vec(sentences=list(doc_it), size=size, window=5,
                    min_count=5, iter=5, workers=2)


def model_generic_dataset(filename, size=100):
    doc_it = generic_labled_doc_iterator(filename)
    return Word2Vec(sentences=list(doc_it), size=size, window=8,
                    min_count=5, iter=10, workers=2)


def yelp_review_text_iterator(filename):
    tkr = Tokenizer(replace_names=True)
    # one document per line
    i = -1;
    with open(filename, "r", encoding="UTF8") as fp:
        for line in fp:
            j = json.loads(line)
            text = tkr.tokenize(j["text"])
            i += 1
            # Print progress every 250k docs
            if i % 250000 == 0:
                print(str(i) + " docs processed...")
            yield text


def generic_labled_doc_iterator(filename, labeled=False):
    tkr = Tokenizer(replace_names=True)
    # one document per line
    i = -1;
    with open(filename, "r", encoding="UTF8") as fp:
        for line in fp:
            if labeled:
                temp = line.split("\t")
                text = tkr.tokenize(temp[1].strip())
            else:
                text = tkr.tokenize(line)
            i += 1
            # Print progress every 1k docs
            if i % 1000 == 0:
                print(str(i) + " docs processed...")
            yield text


trained=True
if __name__ == "__main__":

    if not trained:
        print("Training...")
        model = model_generic_dataset(
                "../Input/w2v_md.txt", 200)
        model.save("../Input/Models/news_w2v_200.p")

        print("Finished Training!")
    else:
#         model = Word2Vec.load("../Input/Models/news_w2v_200.p")
        model = Word2Vec.load("../Input/Models/news_w2v_50.p")

    print(model.similarity("majority", "minority"))
    print(model.similarity("majority", "yesterday"))
    print(model.similarity("airport", "traveling"))
    print()
    print(model.similarity("arrested", "alleged"))
    print(model.similarity("arrested", "majority"))
    print(model.similarity("arrested", "minority"))
    print()
    print(model.similarity("thursday", "friday"))
    print(model.similarity("friday", "plans"))
    print(model.similarity("friday", "failed"))
    print()
    print(model.similarity("fled", "lost"))
    print(model.similarity("fled", "meeting"))
    print(len(model.seeded_vector(0)))

    print("Finished!")