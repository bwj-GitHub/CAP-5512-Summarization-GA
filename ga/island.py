# Island.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Homework 4

import time
import math
import random
import pickle


from ga.parameters import Parameters
from ga.fitness_function import FitnessFunction
from ga.tldr import TLDR
from ga.onemax import OneMax
from ga.number_match import NumberMatch
from ga.chromo import SummaryChromo, SummaryTools
import ga.hwrite as hwrite

class Island(object):

    def __init__(self, parms, file_name, tldr, r_seed):
        """ Constructor """

        # Start timing the GA
        self.start_time = time.clock()
        
        # Get params file name
        self.parm_values = self.set_parameters(parms, file_name)
        self.run_seed = r_seed
        
        # Set up recording structures
        self.gen_best = []
        self.gen_avg = []
        self.run_best = []
        for i in range(self.parm_values.num_runs):
            self.gen_best.append([])
            self.gen_avg.append([])
            self.run_best.append([])

        # Write parameters to summary output file
        id_str = self.get_id_str()
        summary_file_name = "../Output/" + self.parm_values.exp_id + id_str + "_summary.txt"
        self.summary_output = open(summary_file_name, "w")
        self.parm_values.output_parameters(self.summary_output)

        # Set up Fitness Statistics matrix
        self.fitness_stats = []
        for i in range(2):
            temp = []
            for j in range(self.parm_values.generations):
                temp.append(0)
                                
            self.fitness_stats.append(temp)

        # Set up arrays to record every generation's best/avg
        for i in range(self.parm_values.num_runs):
            for j in range(self.parm_values.generations):
                self.gen_best[i].append(-1)
                self.gen_avg[i].append(-1)
            
        # Calculate number of elites
        self.num_elites = int(self.parm_values.pop_size * self.parm_values.elitism_rate)
        if self.num_elites == 0 and self.parm_values.elitism_rate > 0:
            self.num_elites = 1
        self.elites = []
        for i in range(self.num_elites):
            self.elites.append(SummaryChromo())

        # Problem specific setup
        self.problem = tldr
        print(self.problem.name)
        
        # Initialize stuff
        self.best_of_gen_chromo = SummaryChromo()
        self.best_of_run_chromo = SummaryChromo()
        self.best_over_all_chromo = SummaryChromo()
        self.best_of_run_r = -1
        self.best_of_run_g = -1
        
        if self.parm_values.min_or_max == "max":
            self.default_best = 0.0
            self.default_worst = 999999999999999999999.0
        else:
            self.default_best = 999999999999999999999.0
            self.default_worst = 0.0
        
        self.best_over_all_chromo.raw_fitness = self.default_best
        
        self.r = 0
        
    
    #Key for codes:
    #0 represents uniform distribution
    #1 represents logarithmic uniform distribution
    #2 represents normal distribution    
    def set_parameters(self, parms, file_name):
        """ Sets the parameters that this local GA will use """

        # JR should change this method
        #mut_rate = Island.generate_mut_rate(1)
        #xover_rate = Island.generate_xover_rate(0)
        #pop_size = Island.generate_pop_size()
        
        # Default - just takes parms from file
        #return Parameters(file_name)
        
        return parms
    
    
    def get_id_str(self):
        """ Returns a string that uniquely identifies this island """
        ret_str = ""
        ret_str += "_r" + str(self.run_seed) + "_"
        ret_str += "m0." + str(int(100*self.parm_values.mutation_rate)) + "_"
        ret_str += "x0." + str(int(100*self.parm_values.xover_rate)) + "_"
        ret_str += "p" + str(int(self.parm_values.pop_size)) + "_"
        ret_str += "s0." + str(int(100*self.parm_values.sparsity)) + "_"
        return ret_str
        
    def start_run(self):
        """ Initializes stuff to begin a run """
        
        self.r += 1        
        
        self.best_of_run_chromo.raw_fitness = self.default_best
        
        # Initialize first generation
        self.member = []
        self.child = []
        for i in range(self.parm_values.pop_size):
            self.member.append(SummaryChromo())
            self.child.append(SummaryChromo())
            
        self.g = -1
     
     
    def get_best_of_gen(self):
        """ Returns best elite to migrate to other islands """        
        return self.best_of_gen_chromo        
            
            
    def run_next_generation(self, migrants=[]):
        """ Executes one generation of local GA """
        
        self.g += 1
        
        self.sum_pro_fitness = 0
        self.sum_scl_fitness = 0
        self.sum_raw_fitness = 0
        self.sum_raw_fitness2 = 0
        self.best_of_gen_chromo.raw_fitness = self.default_best

        for m in range(len(migrants)):
            worst_of_gen_chromo = self.member[0]            
            for i in range(self.parm_values.pop_size):
                if i > 0:
                    if self.parm_values.min_or_max == "max":
                        if self.member[i].raw_fitness < worst_of_gen_chromo.raw_fitness:
                            worst_of_gen_chromo = self.member[i]
                    elif self.parm_values.min_or_max == "min":
                        if self.member[i].raw_fitness > worst_of_gen_chromo.raw_fitness:
                            worst_of_gen_chromo = self.member[i]

            self.member.remove(worst_of_gen_chromo)
            self.member.append(migrants[m])

        # Test fitness of each member
        for i in range(self.parm_values.pop_size):

            # BJ: If a member changed, it's fitness would be set to -1;
            #     don't re-evaluate members that haven't changed.
#             self.member[i]._reset_fitness()

            if (self.member[i] == -1):
                self.problem.do_raw_fitness(self.member[i])

            self.sum_raw_fitness += self.member[i].raw_fitness
            self.sum_raw_fitness2 += self.member[i].raw_fitness * self.member[i].raw_fitness

            # Update best chromosomes
            if self.parm_values.min_or_max == "max":
                if self.member[i].raw_fitness > self.best_of_gen_chromo.raw_fitness:
                    SummaryChromo.copy_b2a(self.best_of_gen_chromo, self.member[i])
                    self.best_of_gen_r = self.r
                    self.best_of_gen_g = self.g
                if self.member[i].raw_fitness > self.best_of_run_chromo.raw_fitness:
                    SummaryChromo.copy_b2a(self.best_of_run_chromo, self.member[i])
                    self.best_of_run_r = self.r
                    self.best_of_run_g = self.g
                if self.member[i].raw_fitness > self.best_over_all_chromo.raw_fitness:
                    SummaryChromo.copy_b2a(self.best_over_all_chromo, self.member[i])
                    self.best_over_all_r = self.r
                    self.best_over_all_g = self.g                        
            else:
                if self.member[i].raw_fitness < self.best_of_gen_chromo.raw_fitness:
                    SummaryChromo.copy_b2a(self.best_of_gen_chromo, self.member[i])
                    self.best_of_gen_r = self.r
                    self.best_of_gen_g = self.g
                if self.member[i].raw_fitness < self.best_of_run_chromo.raw_fitness:
                    SummaryChromo.copy_b2a(self.best_of_run_chromo, self.member[i])
                    self.best_of_run_r = self.r
                    self.best_of_run_g = self.g
                if self.member[i].raw_fitness < self.best_over_all_chromo.raw_fitness:
                    SummaryChromo.copy_b2a(self.best_over_all_chromo, self.member[i])
                    self.best_over_all_r = self.r
                    self.best_over_all_g = self.g
                    
        # Accumulate fitness statistics
        self.fitness_stats[0][self.g] += self.sum_raw_fitness / self.parm_values.pop_size
        self.fitness_stats[1][self.g] += self.best_of_gen_chromo.raw_fitness
        
        self.average_raw_fitness = self.sum_raw_fitness / self.parm_values.pop_size
        self.stddev_raw_fitness = math.sqrt(math.fabs(self.sum_raw_fitness2 - self.sum_raw_fitness*self.sum_raw_fitness/self.parm_values.pop_size)/(self.parm_values.pop_size-1))
        
        print(str(self.r) + "\t" + str(self.g) + "\t" + str(self.best_of_gen_chromo.raw_fitness) + "\t" + str(self.average_raw_fitness) + "\t" + str(self.stddev_raw_fitness))
        
        # Output generation statistics to summary file
        self.summary_output.write(" R ")
        hwrite.right(self.r, 3, self.summary_output)
        self.summary_output.write(" G ")
        hwrite.right(self.g, 3, self.summary_output)
        hwrite.right_places(float(self.best_of_gen_chromo.raw_fitness), 7, 3, self.summary_output)
        hwrite.right_places(self.average_raw_fitness, 11, 3, self.summary_output)
        hwrite.right_places(self.stddev_raw_fitness, 11, 3, self.summary_output)
        self.summary_output.write("\n")
                                       
        
        """ SCALE FITNESS OF EACH MEMBER AND SUM """
        
        if self.parm_values.scale_type == 0:       # No change
            for i in range(self.parm_values.pop_size):
                self.member[i].scl_fitness = self.member[i].raw_fitness + 0.000001
                self.sum_scl_fitness += self.member[i].scl_fitness
                
        elif self.parm_values.scale_type == 1:     # Invert fitness
            for i in range(self.parm_values.pop_size):
                self.member[i].scl_fitness = 1 / (self.member[i].raw_fitness + 0.000001)
                self.sum_scl_fitness += self.member[i].scl_fitness
                
        elif self.parm_values.scale_type == 2:     # Scale by rank (max fitness)
            
            # Copy genetic data to temp array
            for i in range(self.parm_values.pop_size):
                self.member_index[i] = i
                self.member_fitness[i] = self.member[i].raw_fitness
                
            # Bubble sort the array by floating point number
            for i in range(self.parm_values.pop_size-1, 0, -1):
                for j in range(0, i):
                    if self.member_fitness[j] > self.member_fitness[j+1]:
                        self.t_member_index = self.member_index[j]
                        self.t_member_fitness = self.member_fitness[j]
                        self.member_index[j] = self.member_index[j+1]
                        self.member_fitness[j] = self.member_fitness[j+1]
                        self.member_index[j+1] = self.t_member_index
                        self.member_fitness[j+1] = self.t_member_fitness
                        
            # Copy ordered array to scale fitness fields
            for i in range(self.parm_values.pop_size):
                self.member[self.member_index[i]].scl_fitness = i
                self.sum_scl_fitness += self.member[self.member[i]].scl_fitness
            
        elif self.parm_values.scale_type == 3:     # Scale by rank (min fitness)
            
            # Copy genetic data to temp array
            for i in range(self.parm_values.pop_size):
                self.member_index[i] = i
                self.member_fitness[i] = self.member[i].raw_fitness
                
            # Bubble sort the array by floating point number
            for i in range(1, self.parm_values.pop_size):
                for j in range(self.parm_values.pop_size-1, i-1, -1):
                    if self.member_fitness[j-1] < self.member_fitness[j]:
                        self.t_member_index = self.member_index[j-1]
                        self.t_member_fitness = self.member_fitness[j-1]
                        self.member_index[j-1] = self.member_index[j]
                        self.member_fitness[j-1] = self.member_fitness[j]
                        self.member_index[j] = self.t_member_index
                        self.member_fitness[j] = self.t_member_fitness
                        
            # Copy ordered array to scale fitness fields
            for i in range(self.parm_values.pop_size):
                self.member[self.member_index[i]].scl_fitness = i
                self.sum_scl_fitness += self.member[self.member[i]].scl_fitness
            
        else:
            print("ERROR - No scaling method selected")
            
            
        """ PROPORTIONALIZE SCALED FITNESS FOR EACH MEMBER AND SUM """
        
        for i in range(self.parm_values.pop_size):
            self.member[i].pro_fitness = self.member[i].scl_fitness / self.sum_scl_fitness
            self.sum_pro_fitness += self.member[i].pro_fitness


        """ CROSSOVER AND CREATE NEXT GENERATION """

        parent1 = -1
        parent2 = -1

        # Save the elites
        self.member.sort(key=lambda M: M.raw_fitness)
#         self.sort_members()
        self.get_elites()
        
        # Assume always two offspring per mating
        for i in range(0, self.parm_values.pop_size, 2):
            
            # Leave room for the elites
            if (i + self.num_elites) >= self.parm_values.pop_size:
                break
            
            # Select two parents
            parent1 = SummaryChromo.select_parent(self.member, self.parm_values)
            parent2 = SummaryChromo.select_parent(self.member, self.parm_values)
            while parent2 == parent1:
                parent2 = SummaryChromo.select_parent(self.member, self.parm_values)

            # Crossover two parents to creat two children
            randnum = random.random()
            if randnum < self.parm_values.xover_rate:
                SummaryChromo.mate_parents(self.member[parent1], self.member[parent2], self.child[i], self.child[i+1])
            else:
                SummaryChromo.clone(parent1, self.member[parent1], self.child[i])
                SummaryChromo.clone(parent2, self.member[parent2], self.child[i+1])

        # Mutate children
        randnum = random.random()
        for i in range(self.parm_values.pop_size):
            if randnum < self.parm_values.mutation_rate:
                self.child[i].do_mutation()
            
        # Swap children with last generation
        for i in range(self.parm_values.pop_size):
            SummaryChromo.copy_b2a(self.member[i], self.child[i])

        # Add the elites back in
        if self.g == self.parm_values.generations-1:
            self.insert_elites(True)
        else:
            self.insert_elites(False)
            
        
    def finish_run(self):
            """ Wraps up stuff to finish a run """
            
            hwrite.left(self.best_of_run_r, 4, self.summary_output)
            hwrite.right(self.best_of_run_g, 4, self.summary_output)
            self.problem.do_print_genes(self.best_of_run_chromo, self.summary_output)
            print(str(self.r) + "\t" + "B" + "\t" + str(float(self.best_of_run_chromo.raw_fitness)))
            print()    
            
			# Pickle best Chromo
            f_name = ("../Output/Best Chromos/chromo" +
                      "-f" + str(self.best_of_gen_chromo.raw_fitness) +
                      "-p" + str(self.parm_values.pop_size) +
                      "-m" + str(self.parm_values.mutation_rate) +
                      "-x" + str(self.parm_values.xover_rate) +
                      ".p")
            with open(f_name, "wb") as fp:
                pickle.dump(self.best_of_gen_chromo, fp, protocol=2)
        
    def shut_down(self):
        """ Code for when this local GA is done with all runs """

        # Output fitness statistics matrix
        self.summary_output.write("Gen | AvgFit | StdDev-Avg | BestFit | StdDev-Best\n")

        for i in range(self.parm_values.generations):
            hwrite.left(i, 15, self.summary_output)

            # Print avg of avg
            hwrite.left_places(self.fitness_stats[0][i]/self.parm_values.num_runs, 20, 4, self.summary_output)

            # Print std dev of avg of avg
            stddev_avg = 0
            avg_avg = self.fitness_stats[0][i]/self.parm_values.num_runs
            for r in range(self.parm_values.num_runs):
                stddev_avg += (self.gen_avg[r][i] - avg_avg) * (self.gen_avg[r][i] - avg_avg)
            stddev_avg = math.sqrt(stddev_avg / (self.parm_values.num_runs-1))
            hwrite.left_places(stddev_avg, 20, 4, self.summary_output)

            # Print avg of best
            hwrite.left_places(self.fitness_stats[1][i] / self.parm_values.num_runs, 20, 4, self.summary_output)

            # Print std dev of avg of best
            stddev_best = 0
            avg_best = self.fitness_stats[1][i] / self.parm_values.num_runs
            for r in range(self.parm_values.num_runs):
                stddev_best += (self.gen_best[r][i] - avg_best) * (self.gen_best[r][i] - avg_best)
            stddev_best = math.sqrt(stddev_best / self.parm_values.num_runs)
            hwrite.left_places(stddev_best, 20, 4, self.summary_output)
            
            self.summary_output.write("\n")
        
        self.summary_output.write("\n")
        self.summary_output.close()
        
        print()
        print("Start: " + str(self.start_time))
        end_time = time.clock()
        print("End: " + str(end_time))
        
        
        
    """ ELITISM CODE """
    
        
    def swap(self, i, j):
        """ Swaps contents of two SummaryChromo instances """        

        temp = SummaryChromo()
        SummaryChromo.copy_b2a(temp, self.member[i])
        SummaryChromo.copy_b2a(self.member[i], self.member[j])
        SummaryChromo.copy_b2a(self.member[j], temp)
        
        
    def get_elites(self):
        """ Copies best individuals to elites list """

        if self.parm_values.min_or_max == "min":
            for i in range(self.num_elites):
                SummaryChromo.copy_b2a(self.elites[i], self.member[i])
                
        elif self.parm_values.min_or_max == "max":
            for i in range(self.num_elites):
                SummaryChromo.copy_b2a(self.elites[i], self.member[self.parm_values.pop_size-i-1])
                
    
    def insert_elites(self, do_print):
        """ Adds unaltered elites back into the population """
        
        if do_print:
            print("Elites:")
        
        for i in range(self.num_elites):
            SummaryChromo.copy_b2a(self.member[self.parm_values.pop_size-i-1], self.elites[i])
