"""
Created on Mar 16, 2016

@author: brandon
"""


import json

from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, DocvecsArray

from tools.tokenization import Tokenizer, SentenceSegmenter
from tools.file_tools import read_file_as_string, write_file


def model_generic_dataset(filename, size=30):
    doc_it = generic_labled_doc_iterator(filename)
    return Doc2Vec(doc_it, size=size, negative=5,
                   window=16, min_count=5, workers=2)


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
            if i % 1000 == 0:
                print(str(i) + " docs processed...")
            yield TaggedDocument(text, [i])


def get_sims(model, target, docs, n=5):
    tvec = model.infer_vector(target)
    print(tvec)
    sims = []
    for doc in docs:
        dvec = model.infer_vector(doc)
        sims.append(model.docvecs.similarity(tvec, dvec, True))


pp = False
trained=False

if __name__ == "__main__":

    if not pp:
        # read and sentence segment megadoc:
        SS = SentenceSegmenter()
        T = Tokenizer()
        fs = read_file_as_string("../Input/megadoc.txt")
        sents = SS.segment(fs)
        for i in range(len(sents)):
            sents[i] = " ".join(T.tokenize(sents[i]))
        write_file(sents, "../Input/megadoc_pp.txt")

    if trained:
        model = Doc2Vec.load("../Input/md_d2v_100.p")
    else:
        model = model_generic_dataset("../Input/megadoc_pp.txt", 100)
        model.save("../Input/md_d2v_100.p")
        print("Finished Training!")

    # Test similarity:
    a = "the prime minister fled the country this past thursday".split()
    b = "the prime minister fled the country this past saturday".split()
    c = "the prime minister fled the country this past wednesday".split()
    d = "our prime minister fled the country this past thursday".split()
    S=10
    a_vec = model.infer_vector(a, S)
    b_vec = model.infer_vector(b, S)
    c_vec = model.infer_vector(c, S)
    d_vec = model.infer_vector(d, S)
    print("a: b, c, d")
    sb = model.docvecs.similarity(a_vec, b_vec, True)
    sc = model.docvecs.similarity(a_vec, c_vec, True)
    sd = model.docvecs.similarity(a_vec, d_vec, True)
    with open("quick_out.txt", "w") as fp:
        fp.write(str(sb) + "\n")
        fp.write(str(sc) + "\n")
        fp.write(str(sd) + "\n")
#     b = model.infer_vector(docs[0][0], steps=S)
#     b_2 = model.infer_vector(docs[1][0], steps=S)  # b and b_2 should be closer than b and c
#     c = model.infer_vector(docs[555][0], steps=S)
#     c_2 = model.infer_vector(docs[556][0], steps=S)
#     print(model.docvecs.similarity(b, b_2, True))
#     print(model.docvecs.similarity(b, c, True))
#     print(model.docvecs.similarity(b_2, c, True))
#     print("to c2")
#     print(model.docvecs.similarity(c, c_2, True))
#     print(model.docvecs.similarity(b, c_2, True))
