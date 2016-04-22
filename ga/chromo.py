"""
Created on Apr 10, 2016

@author: brandon

"""


#from random import Random
import random
import cProfile

import numpy as np
import nltk
from gensim.models.word2vec import Word2Vec
from nltk.parse.stanford import StanfordDependencyParser
#from builtins import staticmethod


class SummaryChromo:

    def __init__(self, x_size=62, y_size=10, h_size=50, h2_size=50,
                 sparsity=.80):
        
        
        # GA attributes: #######################
        self.raw_fitness = -1
        self.scl_fitness = -1
        self.pro_fitness = -1
        self.fit_dict = {}  # Record fitness on each summarization task

        # Chromo attributes: ###################
        # Size parameters:
        self.x_size = x_size  # TODO: remove?
        self.y_size = y_size
        self.h_size = h_size
        self.x2_size = self.y_size
        self.y2_size = 1
        self.h2_size = h2_size
        self.sparsity=sparsity

        # RNN 1 (sentence level) parameters:
        self.Wxh = np.random.randn(self.h_size, self.x_size)
        self.Whh = np.random.randn(self.h_size, self.h_size)
        self.Why = np.random.randn(self.y_size, self.h_size)
        self.h = np.zeros((self.h_size,1))
        self.bh = np.random.randn(self.h_size,1)
        self.by = np.random.randn(self.y_size,1)

        # RNN 2 (document level) parameters:
        self.Wxh2 = np.random.randn(self.h2_size, self.x2_size)
        self.Whh2 = np.random.randn(self.h2_size, self.h2_size)
        self.Why2 = np.random.randn(self.y2_size, self.h2_size)
        self.h2 = np.zeros((self.h2_size,1))
        self.bh2 = np.random.randn(self.h2_size,1)
        self.by2 = np.random.randn(self.y2_size,1)

        # For convenience...
        self._parameters = [self.Wxh, self.Whh, self.Why,
                            self.bh, self.by,
                            self.Wxh2, self.Whh2, self.Why2,
                            self.bh2, self.by2]
        self._sparsify()
    
    def _update_params(self, params_list):
        self.Wxh = params_list[0]
        self.Whh = params_list[1]
        self.Why = params_list[2]
        self.bh = params_list[3]
        self.by = params_list[4]
        self.Wxh2 = params_list[5]
        self.Whh2 = params_list[6]
        self.Why2 = params_list[7]
        self.bh2 = params_list[8]
        self.by2 = params_list[9]

    # Summarization Methods: #################################################
    def summarize(self, raw_sents, vec_sents, n_words):
        """Return a summary of sents of at most n_words words.

        raw_sents is a list of strings; each string is a sentence.
        vec_sents is a list of sentence matrices; each sentence
        matrix is a list of token vector, 1 token vector for
        each sentence; each token vecotr is vector representing
        that token (including syntactic information, such as POS.
        """

        importance_scores = self.rate_sentences(vec_sents)
        inds = [i for i in range(len(raw_sents))]
        ranks = list(zip(importance_scores, inds))
        ranks.sort()
        ranks = list(zip(*ranks))[1]
        summary = []
        s_length = 0
        for i in ranks:
            s_length += len(raw_sents[i])
            if s_length < n_words:
                summary.append(raw_sents[i])
            else:
                break  # TODO: Try to find a shorter sentence?
        return summary

    def rate_sentences(self, sents):
        """Determine the importance of each sentence in sents.

        Each sentence in sents should be a list of word vectors; the
        word vectors for an ENTIRE sentence can be formed with the
        class method parameterize_sentence().
        """

        scores = []
        for sent in sents:
            # Evaluate tokens of sentence with inner RNN:
            for i in range(len(sent)):
                # NOTE: shape of sent[i] should be (x_size,1) NOT (x_size,);
                #  the latter will cause issues...
                y1 = self._token_step(sent[i])
            # NOTE: if rnn does its job, only last y1 is necessary
            
            # Evaluate the entire sentence with the outer RNN:
            scores.append(self._sentence_step(y1))

            # Reset hidden state of inner RNN:
            self.h = np.zeros((self.h_size,1))

        # Reset hidden state of outer RNN:
        self.h2 = np.zeros((self.h2_size,1))

        return scores

    def _token_step(self, x):
        """Run the inner RNN on vector x; return vector of length y_size."""

        # Update the hidden state:
        self.h = np.tanh(np.dot(self.Whh, self.h) + np.dot(self.Wxh, x))

        # Compute the output vector:
        y = np.dot(self.Why, self.h)
        return y

    def _sentence_step(self, x2):
        """Run the outter RNN on vector x2; return a single float."""

        # Update the (second) hidden state:
        self.h2 = np.tanh(np.dot(self.Whh2, self.h2) + np.dot(self.Wxh2, x2))

        # Compute the output vector:
        y2 = np.dot(self.Why2, self.h2)
        return y2

    # GA Methods: ############################################################

    def _sparsify(self):
        """Randomly set parameter values to 0 with prob. self.sparsity."""

        for i in range(len(self._parameters)):
            self._parameters[i] = self.dropoff(self._parameters[i],
                                               self.sparsity)

    def _reset_fitness(self):
        self.raw_fitness = -1
        self.scl_fitness = -1
        self.pro_fitness = -1
        self.fit_dict = {}

    def do_mutation(self):
        # Mutation Type 1 (Add Delta):
        for i in range(len(self._parameters)):
            self._parameters[i] = self.add_delta(self._parameters[i],
                                                 self.sparsity)
        self._update_params(self._parameters)
        self._reset_fitness()

    @staticmethod
    def add_delta(W, p=.5):
        """Return matrix W with a small delta added to some values."""

        delta = np.random.randn(*W.shape) * .1
        delta_mask = np.random.choice(a=[0,1], size=W.shape, p=[p, 1-p])
        delta *= delta_mask  # Set most deltas to 0
        return W + delta

    @staticmethod
    def dropoff(W, p=.5):
        """Randomly set values in W to 0 with p probability each."""

        DO = np.random.choice(a=[0,1], size=W.shape, p=[p, 1-p])
        return W*DO

    @staticmethod
    def select_parent(population, parm_values):
        """ Select a parent for crossover """
        
        r_wheel = 0
        
        select_type = parm_values.select_type

        if select_type == 1:       # Proportional Selection
            
            randnum = random.random()
            
            for j in range(parm_values.pop_size):
                r_wheel = r_wheel + population[j].pro_fitness
                if randnum < r_wheel:
                    return j
                
        elif select_type == 2:     # Tournament Selection
            
            pool = []
            
            # Choose candidates
            for i in range(parm_values.pop_size):
                pool.append(random.randint(0, parm_values.pop_size-1))

            # Sort candidates by fitness proportionality
            for i in range(0, parm_values.tourney_size-1):
                for j in range(i+1, parm_values.tourney_size):

                    if parm_values.min_or_max == "max" and population[pool[i]].pro_fitness > population[pool[j]].pro_fitness:
                        temp = pool[i]
                        pool[i] = pool[j]
                        pool[j] = temp
                        
                    elif parm_values.min_or_max == "min" and population[pool[i]].pro_fitness < population[pool[j]].pro_fitnes:
                        temp = pool[i]
                        pool[i] = pool[j]
                        pool[j] = temp

            # Select best?
            best = parm_values.tourney_size - 1
            thresh = parm_values.tourney_thresh
            while best >= 0:
                randnum = random.random()
                if randnum < thresh:
                    return pool[best]
                else:
                    best = best - 1
                    
            return pool[0]
            
        
        elif select_type == 3:     # Random Selection
            
            randnum = random.random()
            j = int(randnum * parm_values.pop_size)
            return j
        
        else:
            print("ERROR - No selection method selected")
        
        return -1

    
    @staticmethod
    def clone(pnum, parent, child):
        
        SummaryChromo.copy_b2a(child, parent)
        child._reset_fitness()
        

    @staticmethod
    def mate_parents(p1, p2, c1, c2):
        
        """ See clone() method
        if p2 is None:
            temp = SummaryChromo()
            SummaryChromo.copy_b2a(temp, p1)
            return temp
        """

        # Init children:
        child1 = SummaryChromo()
        SummaryChromo.copy_b2a(child1, p1)
        child2 = SummaryChromo()
        SummaryChromo.copy_b2a(child2, p2)

        # Crossover parents:
        # NOTE: Assumes that size of parameter matrices is static...

        for i in range(len(p1._parameters)):
            w_shape = p1._parameters[i].shape
            prop = random.random()
            mask1 = np.random.choice(a=[0,1],
                                     size=w_shape,
                                     p=[prop, 1-prop])
            #mask2 = np.ones(*w_shape) - mask1
            mask2 = np.ones(w_shape) - mask1

            child1._parameters[i] = mask1 * p1._parameters[i] +\
                                    mask2 * p2._parameters[i]

            child2._parameters[i] = mask1 * p2._parameters[i] +\
                                    mask2 * p1._parameters[i]

            child1._update_params(child1._parameters)
            child2._update_params(child2._parameters)
        
                                    
        child1._reset_fitness()
        child2._reset_fitness()
        SummaryChromo.copy_b2a(c1, child1)
        SummaryChromo.copy_b2a(c2, child2)        
        #return child1, child2
        

    @staticmethod
    def copy_b2a(a, b):
        # NOTE: Assumes other attributes were set correctly during init
        for i in range(len(a._parameters)):
            a._parameters[i] = b._parameters[i].copy()
        a._update_params(a._parameters)
        
        a.raw_fitness = b.raw_fitness
        a.scl_fitness = b.scl_fitness
        a.pro_fitness = b.pro_fitness


class SummaryTools:

    _PRSR_NAME = "stanford-parser.jar"
    _MODELS_NAME = "stanford-parser-models.jar"

    def __init__(self, jar_dir="/Users/brandon/Code/External Packages/JARS/",
                 w2v_model_path="../Input/Models/news_w2v_50.p"):
        parser_path = jar_dir + SummaryTools._PRSR_NAME
        models_path = jar_dir + SummaryTools._MODELS_NAME
        self.SDP = StanfordDependencyParser(parser_path,
                                            models_path)
        self.model = Word2Vec.load(w2v_model_path)
        # Determine model vector size:
        self.v_size = len(self.model.seeded_vector(0))

    def parameterize_sentence(self, S):
        """Determine and return the word vectors for each token in S.

        Vector is the word2vec for sent[i] concatenated with parameters
        representing part-of-speech (POS).

        * S should already be tokenized and processed. *

        Additional Parameters:
        is_NN     is_NNS
        is_NNP    is_VB
        is_VBD    is_VBG
        is_DT     is_JJ

        For the sake of efficiency, does not get deps.
        """

        # 1) Tag S for POS:
        # TODO: Time me! It might be what is slowing us to a crawl...
        pos_tags = SummaryTools.get_pos_tags(S)

        word_vecs = []
        # 3) Lookup word2vecs (use 0 vectors if word does not exist):
        for i in range(len(S)):
            word_vecs.append(self.get_word_vec(S[i]))
            other_params = []

            # Determine additional params:
            check_pos = ["NN", "NNP", "NNS", "VB", "VBD", "VBG", "DT", "JJ"]
            for check in check_pos:
                other_params.append(1 if pos_tags[i] == check else 0)
            check_deps = ["root", "advc", "nsubj", "dobj"]

            # Additional parameters go here: #################################
            # TODO: Check if word was in quotations?
            # ################################################################

#             other_params.append(self.get_simmilarity(S[i], prev_root))
#             other_params.append(self.get_simmilarity(S[i], prev_sub))

            # Concatenate additional parameters to the word vector:
            # FIXME: other_params needs to be array
            other_params = np.array([[o] for o in other_params])
            wv = np.array([[v] for v in word_vecs[-1]])
            word_vecs[-1] = np.concatenate((wv, other_params),
                                           axis=0)
        return word_vecs

    def parameterize_sentences(self, S):
        S_params = []
        S_deps = self.get_rel_deps_all(S)
        for i in range(len(S)):
            s = S[i]
            S_params.append(self.parameterize_sentence(s))
            # concat params with corresponding deps
            for w in range(len(S_params[-1])):
                check_deps = ["root", "advc", "nsubj", "dobj"]
                deps = []
                for check in check_deps:
                    deps.append(1 if check in S_deps[i][w] else 0)
                wv = S_params[-1][w]
                deps = np.array([[d] for d in deps])
                S_params[-1][w] = np.concatenate((wv, deps), axis=0)
        return S_params

    def get_word_vec(self, word):
        try:
            return self.model[word]
        except KeyError:
            # word is not in model
            return np.zeros(shape=(self.v_size,))

    @staticmethod
    def get_pos_tags(S):
        # NOTE: use "nltk.download("maxent_treebank_pos_tagger")"
        #  (inside python) if missing resource to call nltk.pos_tag()
        return list(zip(*nltk.pos_tag(S)))[1]

    def get_rel_deps(self, S):
        # {index: ["ROOT", "ADVC", ...]}
        deps = {}
        for i in range(len(S)):
            deps[i] = set()

        parse = self.SDP.parse_sents([S])
        parse = list(parse)[0]  # 1st sentence
        DG = list(parse)[0]  # 1st DepGraph?
        for n in range(len(DG.nodes)):
            if DG.nodes[n]["word"] is None:
                continue  # TOP node (not a word)
            deps[n-1].add(DG.nodes[n]["rel"])
        return deps

    def get_rel_deps_all(self, S):
        deps = []

        parses = list(self.SDP.parse_sents(S))
        for i in range(len(S)):
            s_deps = {}
            for w in range(len(S[i])):
                s_deps[w] = set()
            parse = list(parses[i])  # ith sentence
            DG = parse[0]  # 1st DepGraph?
            for n in range(len(DG.nodes)):
                if DG.nodes[n]["word"] is None:
                    continue  # TOP node (not a word)
                # TODO: Add additional DepParse info?
                s_deps[n-1].add(DG.nodes[n]["rel"])
            deps.append(s_deps)
        return deps

    def get_simmilarity(self, w1, w2, v_size=50):
        try:
            return self.W2V_MODELS[v_size].similarity(w1, w2)
        except KeyError:
            return 0


def param2(S1, S2, S3):
    ST = SummaryTools()
    # TODO: Fix parameterize_sentence by parsing ALL sentences at once!
#     ST.SDP.parse_sents([S])
#     ST.SDP.parse_sents([S2])
#     ST.SDP.parse_sents([S3])
    ST.SDP.parse_sents([S1, S2, S3])


# Testing:
if __name__ == "__main__":
    ST = SummaryTools()
    SC = SummaryChromo()
    SC2 = SummaryChromo()
#     x = np.random.randn(50, 1)
#     y1 = SC._token_step(x)
#     y2 = SC._token_step(x)
#     print(y1)
#     print(y2)
#     print(cosine(y2, y2))
#     print("...")
#     s = nltk.word_tokenize("the cat went home")
#     S = ST.parameterize_sentence(s)
#     print(s)
# 
#     # Test Crossover!
    c1 = SummaryChromo()
    c2 = SummaryChromo()
    print(SC._parameters[7])
    print(SC2._parameters[7])
    print("#####")
    SummaryChromo.mate_parents(SC, SC2, c1, c2)
    print(c1._parameters[7])
    print(c2._parameters[7])

    # Profile performance:
#     S = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia's top political leaders, saying the meeting would not bring any progress in deadlocked negotiations to form a government."
#     S2 = "They have demanded a thorough investigation into their election complaints as a precondition for their cooperation in getting the national assembly moving and a new government formed."
#     S3 = "The King of Eastern New-Zealand was cited today as having said that he would oppose any opposition to his newly proposed bill."
#     T = Tokenizer()
#     S = T.tokenize(S)
#     S2 = T.tokenize(S2)
#     S3 = T.tokenize(S3)
#     cProfile.run("param2(S, S2, S3)")
#     
#     # Test new parameterize_sentences!
#     print("Test new para...")
#     PARMS = ST.parameterize_sentences([S, S2, S3])
