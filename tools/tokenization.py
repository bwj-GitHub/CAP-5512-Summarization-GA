# -*- coding: utf-8 -*-
"""
Created on October 2, 2015

@author: Brandon

"""


import sys
import re

import nltk.data
from nltk.stem.wordnet import WordNetLemmatizer

from tools.file_tools import read_file
from tools.stemmer import Stemmer


class Tokenizer(object):
    """A customizable tokenizer.

    TODO: Describe me more!
    """

    PUNCT = set(['.', ',', '?', '!', '@', '#', '$', '%', '^', ';', '/',
                  '&', '*', '(', ')', '-', '+', ':',
                  # '>', '<',  # Don't replace these, for now.
#                 '\'', "\"",  # Replace these?
                  ';', '`', '|', '[', ']'])
    DEF_DIR = "../Input/"  # Default directory for replacement files.

    # Special HTML Characters
    HTML_CODES = [
            ("&#33;", "!"),
            ("&#34;", "\""),
            ("&quot;", "\""),
            ("&#35;", "#"),
            ("&#36;", "$"),
            ("&#37;", "%"),
            ("&#38;", "&"),
            ("&amp;", "&"),
            ("&#39;", "'")
            ]


    def __init__(self, **kwargs):

        """Brief summary...

        TODO: Document me!
        TODO: Always ignore <> ?
        """
        
        # Function shortcut initializations:
        self._case = lambda x: x
        self._lemmatize = lambda x: x
        self._segment = lambda x: x  # TODO: Change segmentation...
        self._stem = lambda x: x
        
        # Location args:
        self.list_dir = kwargs.get("list_dir", "../Input/Lists/")
        self.user_replacements = kwargs.get("user_replacements", None)

        # Replacement args:
        self.replace_codes = kwargs.get("replace_codes", True)
        self.replace_companies = kwargs.get("replace_companies", False)
        self.replace_contractions = kwargs.get("replace_contractions", True)
        self.replace_names = kwargs.get("replace_names", False)
        self.replace_nums = kwargs.get("replace_nums", False)
        self.replace_prices = kwargs.get("replace_prices", False)
        self.replace_products = kwargs.get("replace_products", False)
        self.replace_times = kwargs.get("replace_times", False)

        # Other args:
        self.sentence_boundry_detection = kwargs.get("sents", False)
        self.lemmatize = kwargs.get("lemma", False)
        self.stem = kwargs.get("stem", None)  # Specify the language to use, or True to use English

        self.preserve_punct = kwargs.get("preserve_punct", False)
        self.separate_elipsis = kwargs.get("sep_elipsis", True)
        self.squash_punctuation = kwargs.get("squash_punct", False)  # TODO: Not implemented
        self.squash_elipsis = kwargs.get("squash_elipsis", False)  # TODO: Not implemented
        self.punct_spacing = kwargs.get("punct_spacing", True)

        self.short_abrevs = kwargs.get("abbrevs", False)  # Remove '.'s
        self.remove_timestamps = kwargs.get("remove_timestamps", False)
        self.preserve_case = kwargs.get("preserve_case", False)
        self._verbose = kwargs.get("verbose", False)
        
        # Set up function shortcuts:
        if self.preserve_case is False:
            self._case = str.lower
        if self.lemmatize:
            self._lemmatize = WordNetLemmatizer().lemmatize
        if self.sentence_boundry_detection:
            self._segment = SentenceSegmenter().segment
        if self.stem:
            self._stem = Stemmer().stemWord

        # Load replacement files:
        self.punct_regex_list, self.no_punct_regex_list, \
                self.punct_word_list, self.no_punct_word_list, \
                self.punct_words_list, \
                self.no_punct_words_list = self._load_replacements()

        self.codes, self.c_reps = zip(*Tokenizer.HTML_CODES)

        self._warned = False

    def tokenize(self, text, transform=None):
        """Split the string text into a list of tokens, return tokens.
        
        text: string to be split into tokens.
        transform: optional function to be applied to all tokens.
        """
        
        # A string is expected, if a list is provided warn the user
        # (once for a Tokenizer), join the list together, and
        # tokenize the new list
        if isinstance(text, list):
            if not self._warned:
                """
                print("WARNING: You are tokenizing a list, " +
                      "list items will be joined together " +
                      "and then tokenized.", file=sys.stderr)
                  """
                self._warned = True
            text = " ".join(text)
        
        # TODO: Allow for setting of this operation in init
        # Remove escape characters:
        text = text.replace("\\", "")
        
        # Separate sentences?
        if self.sentence_boundry_detection:
            text = self._segment(text, ' <sb> ')
        
        # Setup transform function:
        if transform is None:
            transform = lambda x: x

        # Detach any '<' or '>' from tokens:
        text = text.replace('<', ' <')
        text = text.replace('>', '> ')
        
        # Separate elipsis from other tokens:
        if self.separate_elipsis:
            text = text.replace('...', ' ... ')
        
        # Remove '.'s from abbreviations:
        
        
        # Replace left/right quotation marks with apostrophes:
        text = text.replace('\u2018', '\'')  # left single quotation mark
        text = text.replace('\u2019', '\'')  # right ...

        # Make any HTML encoding replacements:
        if self.replace_codes:
            for i in range(len(self.codes)):
                code = self.codes[i]
                rep = self.c_reps[i]
                text = text.replace(code, rep)

        # Add spacing between punctuation and tokens:
        if self.punct_spacing:
            reg = re.compile("[.,;()][A-Za-z]")
            strings = reg.findall(text)
            for string in strings:
                text = text.replace(string, string[0] + " " + string[1:])

        # Make punctuation sensitive replacements and tokenize:
        text = self._replace_punct_regex(text)
        text = self._replace_punct_words(text)
        tokens = text.split()
        tokens = self._replace_punct_word(tokens)

        # Remove or expand punctuation:
        if self.preserve_punct:
            tokens = self._expand_punctuation(tokens)
        else:
            for i in range(len(tokens)):
                tokens[i] = self._remove_punct(tokens[i])

        # Make any remaining replacements:
        tokens = self._replace_np_word(tokens)
        tokens = self._replace_np_words(tokens)
        final_tokens = []
        for i in range(len(tokens)):
            token = tokens[i]
            tokens[i] = self._replace_np_regex(token)
            tokens[i] = transform(tokens[i])
            tokens[i] = self._case(tokens[i])
            tokens[i] = self._stem(tokens[i])
            tokens[i] = self._lemmatize(tokens[i])  # Expects lowercase?
            if tokens[i] != "":
                final_tokens.append(tokens[i])

        if self._verbose:
            print(final_tokens)
        return final_tokens

    def _expand_punctuation(self, tokens):
        # TODO: Document me!
        new_tokens = []
        for token in tokens:
            toks = []
            pa = False  # "punctuation active": last token was PUNCT
            for char in token:
                if char in Tokenizer.PUNCT and pa is False:
                    toks.append(char)
                    pa = True
                elif char not in Tokenizer.PUNCT and pa is True:
                    toks.append(char)
                    pa = False
                elif toks == []:
                    toks.append(char)
                else:
                    toks[-1] += char

            new_tokens.extend(toks)
        return new_tokens

    def _remove_punct(self, token):
        for punct in Tokenizer.PUNCT:
            token = token.replace(punct, "")
        return token

    def _load_replacements(self):
        punct_regex = []
        np_regex = []
        punct_word = {}
        np_word = {}
        punct_words = {}
        np_words = {}
        
        # Determine replacement files:
        replacement_files = []
        if self.user_replacements is not None:
            replacement_files.append(str(self.list_dir +
                                         self.user_replacements))
        if self.replace_companies:
            replacement_files.append(self.list_dir + "company names.txt")
        if self.replace_contractions:
            replacement_files.append(self.list_dir + "contractions.txt")
        if self.replace_names:
            replacement_files.append(self.list_dir + "names.txt")
            
        # Add nums, prices, times, and product replacement regexes:
        punct_regex.extend(self._compile_replacement_regexes())

        # Parse replacement files:
        for fn in replacement_files:
            res = False
            ps = False  # Punctuation Sensitive
            #with open(fn, "r", encoding="UTF8") as fp:
            with open(fn, "r", encoding="UTF8") as fp:
                for line in fp:
                    if line.strip() == "# -regex, -punct" or \
                            line.strip() == "# -punct, -regex":
                        res = True
                        ps = True
                        continue
                    elif line.strip() == "# -regex":
                        res = True
                        continue
                    elif line.strip() == "# -punct":
                        ps = True
                        continue

                    toks = line.split("\t")
                    for i in range(len(toks)):
                        toks[i] = toks[i].strip()

                    if len(toks) == 1:  # Must be names list:
                        np_word[toks[0]] = "<NAME>"
                    elif res and ps:
                        punct_regex.append((re.compile(toks[0]), toks[1]))
                    elif res:
                        np_regex.append((re.compile(toks[0]), toks[1]))
                    elif ps:
                        if toks[0].find(" ") != -1:
                            punct_words[toks[0]] = toks[1]
                        else:
                            punct_word[toks[0]] = toks[1]
                    else:
                        if toks[0].find(" ") != -1:
                            np_words[toks[0]] = toks[1]
                        else:
                            np_word[toks[0]] = toks[1]

        return punct_regex, np_regex, punct_word, np_word, \
                punct_words, np_words
                
    def _read_replacement_file(self, filename):
        lines = []
        
        # Check for the file in the directory specified in
        # self.replaced_lists:
        try:
            lines = read_file(str(self.replaced_lists + filename))
            return lines
        except FileNotFoundError:
            pass
        
        # Check for the file in "../Input/":
        try:
            lines = read_file(str("../Input/" + filename))
            return lines
        except FileNotFoundError:
            pass
        
        # Check for the file in the current directory:
        try:
            lines = read_file(filename)
            return lines
        except FileNotFoundError:
            pass
        
        # Check for the Input/ directory in this directory:
        try:
            lines = read_file(str("Input/" + filename))
            return lines
        except FileNotFoundError:
            pass
        
        return lines

    def _compile_replacement_regexes(self):
        replacements = []
        
        if self.replace_times:
            replacements.append((re.compile(
                    "(([0]?[1-9])|([12][0-9]))(([ :]([0-6][0-9][ -]??" +
                    "(([AaPp][.]?[Mm][.]?)|())))|([ -]?[AaPp][.]?[Mm][.]?))"),
                    " <time> "))
        if self.remove_timestamps:
            replacements.append((re.compile(
                    "\[[0-9]{1,2}:[0-9]{1,2}:?[0-9]{0,2}[ ]?" +
                    "[AaPp]?[.]?[Mm]?[.]?\]"),
                    " <timestamp> "))
        if self.replace_prices:
            replacements.append((re.compile(
                    "([$£][ ]?[0-9]+[.]?[0-9]{0,3})|" +
                    "([0-9]+[.]?[0-9]{0,3}[ ]?[$£])"),
                    " <price> "))
        if self.replace_nums:
            replacements.extend([
                (re.compile("[0-9]" ), " <SD> "),
                (re.compile("[0-9]{2}"), " <dd> "),
                (re.compile("[0-9]{3,6}"), " <MED_NUM> "),
                (re.compile("[0-9]{7,11}"), " <LARGE_NUM> "),
                (re.compile("[0-9]{12,}"), " <EX_LARGE_NUM> ")
            ])

        return replacements
            
    def _replace_punct_regex(self, text):
        for pair in self.punct_regex_list:
            reg = pair[0]
            rep = pair[1]
            strings = reg.findall(text)
            for string in strings:
                if isinstance(string, tuple):
                    # Find the first non-empty group
                    for g in string:
                        if g != "":
                            text = text.replace(g, rep)
                else:
                    text = text.replace(string, rep)                
        return text

    def _replace_np_regex(self, token):
        for pair in self.no_punct_regex_list:
            reg = pair[0]
            rep = pair[1]
            if reg.match(token):
                token = rep               
        return token

    def _replace_punct_word(self, tokens):
        for i in range(len(tokens)):
            token = tokens[i].lower()
            if self.punct_word_list.get(token):
                tokens[i] = self.punct_word_list[token]
        # Fix any tokens containing spaces:
        temp = " ".join(tokens)
        tokens = temp.split()
        return tokens

    def _replace_np_word(self, tokens):
        # Note: Ignores case!
        for i in range(len(tokens)):
            token = tokens[i].lower()
            if self.no_punct_word_list.get(token):
                tokens[i] = self.no_punct_word_list[token]
        return tokens

    def _replace_punct_words(self, text):
        for word in self.punct_words_list.keys():
            while text.find(word) != -1:
                text = text.replace(word, self.punct_words_list[word])
        return text

    def _replace_np_words(self, tokens):
        for word in self.no_punct_words_list.keys():
            for t in range(len(tokens)):
                token = tokens[t]
                if token == word:
                    tokens[t] = self.no_punct_words_list[word]
                if tokens[t].find(" ") != -1:
                    new_toks = tokens[t].split()
                    tokens[t] = new_toks[0]
                    for nt in range(1, len(new_toks)):
                        tokens.insert(t+1, new_toks[nt])
        return tokens

    def tokenize_nums(self, text, transform=None):
        # The order of the items in match_and_replace matters!
        tokens = text.split()
        for t in range(len(tokens)):
            tokens[t] = tokens[t].replace(".", "")
            tokens[t] = tokens[t].replace("'", "")
            tokens[t] = tokens[t].replace(",", "")
            tokens[t] = tokens[t].replace(">", "")
            tokens[t] = tokens[t].replace("<", "")
            tokens[t] = tokens[t].replace(";", "")
            tokens[t] = tokens[t].replace("&", "")
            tokens[t] = tokens[t].replace("$", "")
            tokens[t] = tokens[t].replace("!", "")
            tokens[t] = tokens[t].replace("@", "")
            tokens[t] = tokens[t].replace("#", "")
            tokens[t] = tokens[t].replace("(", "")
            tokens[t] = tokens[t].replace(")", "")
            tokens[t] = tokens[t].replace(":", "")
            tokens[t] = tokens[t].replace("?", "")
            tokens[t] = tokens[t].lower()
        ## TEmporarily override
        # TODO: Remove these comment outs
        return tokens

    def _form_name_regex(self, string):
        regex = ""
        # TODO: do stuff
        for character in string:
            if character.is_lower() is True:
                regex.append(character)
            else:
                regex.append(str("[" + character + character.lower() + "]"))
        return re.compile(regex)

    @staticmethod
    def is_product(string):
        if re.search("[0-9]", string) and re.search("[a-zA-Z]", string):
            # If a string contains both letters and numbers,
            # it is probably a product
#             if re.search("[aeiouAEIOU]", string):
            if re.match(".*[aeiou].*", string) and string.find("x") == -1:
                return False
            elif string.lower() == "3g" or string.lower() == "4g" or\
                    string.lower() == "3d" or string.lower() == "2d" or\
                    string.lower() == "mp3" or string.lower() == "mp4":
                return False
            elif re.match("[0-9]*[.]*[0-9]+[kK]$", string):
                return False
            elif string.find("/") != -1 or string.find("\\") != -1 or\
                    string.find("=") != -1 or\
                    string.lower().find("wpa") != -1 or\
                    string.lower().find("isp") != -1 or\
                    string.lower().find("html") != -1 or\
                    string.lower().find("hdmi") != -1:
                return False
            elif re.match("[0-9]+[GgKkMm][Bb][.]?$", string):
                return False
            elif re.match("[0-9]+[Tt][Hh][.]?$", string):
                return False
            elif re.match("[0-9]+[Ss][Tt][.]?$", string):
                return False
            elif re.match("[0-9]+[RrNn][Dd][.]?$", string):
                return False
            elif re.match("[0-9]+[AaPp][Mm][.]?$", string):
                return False
            elif re.match("[Ii][Pp].+[.]?$", string):
                return False
            elif re.match("[0-9]+[Hh][Zz][.]?$", string):
                return False
            elif re.match("[0-9]+[Vv][.]?$", string):
                return False
            elif re.match("[0-9]+[MmCc][Mm][.]?$", string):
                return False
            else:
#                 print(string)
                return True

    def is_company(self, string):
        """Determine if string is a company name, return True if it is.

        Since company names can have more than one word in them, so
        too can string.

        It is the calling method's job to send appropriately
        matched words.
        I.E.: is_company("Bank of America is") will return False
            because of the extra token "is".

        The recommended way of dealing with this would be to send phrases
        of sizes ranging from one up to a certain amount, such as 3.
        I.E: is_company("Bank") >> False,
             is_comapny("Bank of") >> False,
             is_company("Bank of America") >> True
        """

        if string in self.companies or string.lower() in self.companies:
            return True

    def is_name(self, string):
        """Determine if string is a name, return True if it is."""

        if string in self.names or string.lower() in self.names:
            return True
        
    @staticmethod
    def is_punctuation(string):
        """Return True if the string is non empty and contains
        only punctuation.
        """

        return len(re.findall("[a-zA-Z]", string)) == 0


class SentenceSegmenter:
    """Segments strings into sentences."""

    def __init__(self, **kwargs):
        self._sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

    def segment(self, string, join_symbol=None):

        # '[word],  ' (two spaces) is a potential sentence boundary
        # TODO: ... and ... and ... and  try to break up run on sentences...
        # Correct for mis-placed periods (ie: He said okay .So we went."):
        string = string.replace("...", "<elipsis>")
        string = string.replace(".", ". ")  # TODO: Don't do this!
        string = string.replace(".  ", ". ")  # Correcting previous!
        string = string.replace("<elipsis>", "...")
        # Correct for some abbreviated months:
        string = string.replace("Jan.", "January")
        string = string.replace("Feb.", "February")
        string = string.replace("Mar.", "Mar")
        string = string.replace("Apr.", "April")

        string = string.replace("Jun.", "June")
        string = string.replace("Jul.", "July")
        string = string.replace("Aug.", "August")
        
        string = string.replace("Sept.", "September")
        string = string.replace("Oct.", "October")
        string = string.replace("Nov.", "November")
        string = string.replace("Dec.", "December")

        # Split into sentences using NLTK
        sents = self._sent_detector.tokenize(string)

        # Correct for extraneous punctuation:
        sents = [s for s in sents if not Tokenizer.is_punctuation(s)]

        # Split on 2+ consecutive spaces:
        temp = []
        for sent in list(sents):
            temp.extend(self._multi_space_segment(sent))
        sents = temp

        if join_symbol:
            sents = join_symbol.join(sents)
        return sents
    
    def _multi_space_segment(self, text):
        tss = text.find("  ")
        if tss != -1:
            segments = re.split("[ ]{2,}", text)
            return segments
        else:
            return [text]  # segment expects a list...


# Tests:
if __name__ == "__main__":
    # Tokenize Sentences (join around <sb>)
    SS = SentenceSegmenter()
    T = Tokenizer(preserve_case=False, preserve_punct=True,
                  replace_names=True, replace_contractions=True,
                  lemma=True,
                  sents=True,
                  remove_timestamps=True)
    
    s = "Hi John, how are you?? I haven't seen you in a while. cats can't fly"
    s2 = SS.segment(s, ' <sb> ')
    print(s2)
    s3 = "What is up my man  It is great to see ya!"
    s = "[08:34:36 AM] Hi, my name is Karnakar. How may I help you?[08:35:58 AM] Derek: I have a "
    print(SS.segment(s3))

    # Tokenize the tokenized sentences
    print(T.tokenize(s))
    
    # Segment stuff:
    s1 = "The monitoring ended Sept. 30."
    print(SS.segment(s1))
