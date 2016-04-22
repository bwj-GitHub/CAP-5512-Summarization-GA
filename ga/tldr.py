# TLDR.py (Too Long, Didn't Read)
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Homework 4

# ***** Brandon's Seal of Approval (pending test) *****

import random
import pickle

import nltk

from ga.fitness_function import FitnessFunction
from ga.chromo import SummaryTools
from tools.rouge import rouge2_bin
from tools.file_tools import read_file_as_string
from tools.tokenization import Tokenizer, SentenceSegmenter

class TLDR(FitnessFunction):

    def __init__(self, parms, sents=(None, None)):
        super(FitnessFunction, self).__init__()
        self.T = Tokenizer()  # TODO: Select optimal Tokenizer paramaters?
        self.SS = SentenceSegmenter()
        self.ST = SummaryTools()  # TODO: Set correct paths?
        self.name = "Extractive Summarization Problem"
        self.parms = parms

        # Load reference bigrams:
        with open("../Input/ref_bigrams.p", "rb") as fp:
            self.ref_bigrams = pickle.load(fp)
        print("Loaded reference bigrams!")

        # Load Metadocs OR pickled sents:
        if sents[0] is None:
            self.raw_sents = {}
            self.vec_sents = {}
            for doc_id in self.parms.training_files:
                print("Loading doc " + str(doc_id))
                self.raw_sents[doc_id] = []
                self.vec_sents[doc_id] = []
                md_path = "../Input/Metadocs/" + doc_id + "t.txt"
                md = read_file_as_string(md_path)
                sents = self.SS.segment(md)
                for s in sents:
                    tokens = self.T.tokenize(s)
                    self.raw_sents[doc_id].append(tokens)
                print("Parameterizing " + doc_id)
                self.vec_sents[doc_id] = self.ST.parameterize_sentences(self.raw_sents[doc_id])
        else:  # Load sents from pickles
            self.raw_sents = sents[0]
            self.vec_sents = sents[1]
        print("TLDR initialized!")

    def do_raw_fitness(self, X):
        """ This calculates the average rouge score of a chromosome's 
                summary's fitness over several training sets
                
            This assumes that each training set's metdoc summary bigrams
                have already been computed
        """

        X.raw_fitness = 0

        # Get ROUGE-2 score for each Meta-Doc in params.training_files:
        for d in range(self.parms.tf_count):
            doc_id = self.parms.training_files[d]
            candidate_bigrams = self.get_candidate_bigrams(X, doc_id)
            ref_bigrams = self.ref_bigrams[doc_id]
            f = rouge2_bin(candidate_bigrams, ref_bigrams)
            X.raw_fitness += f
            X.fit_dict[doc_id] = f
            
            # print TEST
            """
            if X.raw_fitness > .079:
                print("Dude: " + str(X.raw_fitness))
                print(self.parms.tf_count)
                print(X.summarize(self.raw_sents[doc_id],
                                  self.vec_sents[doc_id],
                                  self.parms.summary_word_length))
                  """

        # Get average score of training set scores
        X.raw_fitness = X.raw_fitness / self.parms.tf_count
        

    def get_candidate_bigrams(self, X, doc_id):
        """ Get bigrams of summary for doc_id produced by X.  """

        raw_sents = self.raw_sents[doc_id]
        vec_sents = self.vec_sents[doc_id]
        wc = self.parms.summary_word_length
        # TODO: summarize returns list of sentences (each list of tokens)
        #  concat the lists...
        sums = X.summarize(raw_sents, vec_sents, wc)
        bgs = []
        for s in sums:
            bgs.extend(s)
        bgs = set(nltk.bigrams(bgs))
        return bgs

    def do_print_genes(self, X, output):
        output.write("   RawFitness\n")
        output.write("\n\n")
