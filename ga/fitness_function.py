# Fitness_Function.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Homework 4

from __future__ import print_function

#import ga.chromo

class FitnessFunction(object):
    
    name = ""
    
    def __init__(self):
        print("Setting up Fitness Function....", end="")
        
    def do_raw_fitness(self, X):
        print("Executing FF Raw Fitness")
        
    def do_print_genes(self, X, output):
        print("Executing FF Gene Output")