# OneMax.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Created April 11, 2016
# Homework 4

from ga.fitness_function import FitnessFunction
import ga.chromo
import ga.parameters

class OneMax(FitnessFunction):
    
    def __init__(self, parms):
        super(FitnessFunction, self).__init__()
        self.name = "OneMax Problem"
        self.parm_values = parms
        
    def do_raw_fitness(self, X):
        X.raw_fitness = 0
        """
        for i in range(self.parm_values.num_genes * self.parm_values.gene_size):
            if X.chromo[i] == '1':
                X.raw_fitness += 1
        """
        
    def do_print_genes(self, X, output):
        output.write("   RawFitness\n")
        output.write("\n\n")