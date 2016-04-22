# Number_Match.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Created April 11, 2016
# Homework 4

import math

from ga.fitness_function import FitnessFunction
import ga.chromo

class NumberMatch(FitnessFunction):
    
    test_value = []
    
    def __init__(self, parms):
        super(FitnessFunction, self).__init__()
        name = "Number Match Problem"
        self.parm_values = parms
        
        input_reader = open(self.parm_values.data_input_file_name, "r")
        for i in range(self.parm_values.num_genes):
            self.test_value[i] = int(input_reader.read())
        input.close()
        
    def do_raw_fitness(self, X):
        difference = 0
        """
        for j in range(self.parm_values.num_genes):
            difference = math.fabs(X.get_int_gene_value(j) - test_value[j])
            X.raw_fitness = X.raw_fitness + difference
        """
        
    def do_print_genes(self, X, output):
        output.write("   RawFitness")
        output.write("\n\n")