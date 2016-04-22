'''
Created on Jan 6, 2016

@author: brandon
'''


import snowballstemmer
    
    
class Stemmer:

    def __init__(self, language='english'):
        self.stemmer = snowballstemmer.stemmer(language);
        
    def stemWord(self, word):
        return self.stemmer.stemWord(word)
    
    def stemSentence(self, words):
        words = words.split()
        a = self.stemmer.stemWords(words)
        return " ".join(self.stemmer.stemWords(words))
    
    def stemFile(self, in_file, out_file=None):
        if out_file is None:
            out_file = in_file
        
        stemmed_lines = []
        with open(in_file, 'r', encoding='utf8') as fp:
            for line in fp:
                stemmed_lines.append(self.stem_words(line))
                
        with open(out_file, 'w', encoding='utf8') as fp:
            for line in stemmed_lines:
                fp.write(line + '\n')

if __name__ == '__main__':
    stemmer = Stemmer('spanish')
    stemmer.stem_file('../Input/Tests/subject_spanish.txt',
                      '../Output/Stemmed/subject_spanish-names.txt')
    print("Finished!")
    
        