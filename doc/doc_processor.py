"""
Created on Apr 13, 2016

@author: brandon

Create metadocs for summarization
"""


import os
import pickle

from nltk.util import bigrams

from tools.file_tools import quick_walk, read_file, write_file
from tools.tokenization import Tokenizer


# Parent Directory Locations:
DOCS = "/Users/brandon/Documents/School/Spring_2016/CAP_6640/Final_Project//"
HUMAN_SUMMARIES = "../Input/Human Summaries/"

do_source = False
do_human = True

# Create Meta Source Documents: ##############################################

if do_source:
    # Create Meta-docs of Source Documents:
    meta_docs=[]
    w2v_metadoc = []
    
    # Get the names of all sub-directories in root_dir:
    dirs = list(os.walk(DOCS))[0][1]
    print(dirs)
    
    # Read files in each directory to form meta-docs
    for _dir in dirs:
        name = _dir
        lines = []
        fnames = quick_walk(DOCS + _dir)
        for file in fnames:
            f_lines = read_file(file, ignore_markup=True)
            lines.extend(f_lines)
            w2v_metadoc.extend(lines)
        meta_docs.append((_dir, lines))
    
    # Write metadocs
    for MD in meta_docs:
        write_file(MD[1], "../Metadocs/"+MD[0]+".txt")
    print("Finished Creating Meta Documents!")

    # Write Metadoc for w2v (all metadocs concatenated)
    write_file(w2v_metadoc, "../w2v_md.txt")

# Extract N-Grams (1-3) from Human Summaries: #################################

if do_human:
    T = Tokenizer()
#     meta_docs = {}
#     fnames = quick_walk(HUMAN_SUMMARIES)
#     for f_name in fnames:
#         s_name = f_name.rpartition("/")[2][0:6]
#         if s_name not in meta_docs.keys():
#             meta_docs[s_name] = []
#         meta_docs[s_name].extend(read_file(f_name))
#     
#     for f in meta_docs.keys():
#         write_file(meta_docs[f], "../Input/Summary Metadocs/" + f + ".txt")
# 
#     print("Finished Creating Summary Meta Documents")
    grams = {}
    fnames = quick_walk(HUMAN_SUMMARIES)
    for fname in fnames:
        s_name = fname.rpartition("/")[2][0:6].lower()
        if s_name not in grams.keys():
            grams[s_name] = []  # each list item corresponds to a ref summary
        lines = read_file(fname)
        doc_text = " ".join(lines)
        doc_tokens = T.tokenize(doc_text)
        grams[s_name].append(set(bigrams(doc_tokens)))
    
    # Dump the bigrams!:
    with open("../Input/ref_bigrams.p", "wb") as fp:
        pickle.dump(grams, fp, protocol=2)

    for k in grams.keys():
        print(k + " " + str(grams[k]))
