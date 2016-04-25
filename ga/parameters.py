# Parameters.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Homework 4

class Parameters(object):
    
    def __init__(self, file_name):
        """ Read parameter values from file """
        
        param_input = open(file_name, "r")

        # exp_id
        param_input.read(30)
        self.exp_id = param_input.readline().strip()
        
        # problem_type
        param_input.read(30)
        self.problem_type = param_input.readline().strip()
        
        # data_input_file_name
        param_input.read(30)
        self.data_input_file_name = param_input.readline().strip()
        
        # num_runs
        param_input.read(30)
        self.num_runs = int(param_input.readline())
        
        # generations
        param_input.read(30)
        self.generations = int(param_input.readline())
        
        # pop_size
        param_input.read(30)
        self.pop_size = int(param_input.readline())
        
        # select_type
        param_input.read(30)
        self.select_type = int(param_input.readline())
        
        # scale_type
        param_input.read(30)
        self.scale_type = int(param_input.readline())
        
        # xover_type
        param_input.read(30)
        self.xover_type = int(param_input.readline())
        
        # xover_rate
        param_input.read(30)
        self.xover_rate = float(param_input.readline())
        
        # mutation_type
        param_input.read(30)
        self.mutation_type = int(param_input.readline())
        
        # mutation_rate
        param_input.read(30)
        self.mutation_rate = float(param_input.readline())
        
        # seed
        param_input.read(30)
        self.seed = int(param_input.readline())
        
        # num_genes
        param_input.read(30)
        self.num_genes = int(param_input.readline())
        
        # gene_size
        # TODO: Remove me?
        param_input.read(30)
        self.gene_size = int(param_input.readline())
        
        # tourney_size
        param_input.read(30)
        self.tourney_size = int(param_input.readline())
        
        # tourney_thresh
        param_input.read(30)
        self.tourney_thresh = float(param_input.readline())
        
        # elitism_rate
        param_input.read(30)
        self.elitism_rate = float(param_input.readline())

        # training_files
        param_input.read(30)
        self.training_files = param_input.readline().split()
        self.tf_count = len(self.training_files)
        print("Files used for training: " + str(self.training_files))

        # summary_word_length
        param_input.read(30)
        self.summary_word_length = int(str(param_input.readline()))
        
        # sparsity
        param_input.read(30)
        self.sparsity = float(param_input.readline())
        
        # num_islands
        param_input.read(30)
        self.num_islands = int(param_input.readline())
        
        # xover_dist_type
        param_input.read(30)
        self.xover_dist_type = float(param_input.readline())
        
        # mut_dist_type
        param_input.read(30)
        self.mut_dist_type = float(param_input.readline())

        param_input.close()

        # min_or_max
        if self.scale_type == 0 or self.scale_type == 2:
            self.min_or_max = "max"
        else:
            self.min_or_max = "min"
            
        self.param_dict = {}
        self.param_dict["Experiment ID"] = self.exp_id
        self.param_dict["Problem Type"] = self.problem_type
        self.param_dict["Data Input File Name"] = self.data_input_file_name
        self.param_dict["Number of Runs"] = self.num_runs
        self.param_dict["Generations per Run"] = self.generations
        self.param_dict["Population Size"] = self.pop_size
        self.param_dict["Selection Method"] = self.select_type
        self.param_dict["Fitness Scaling Type"] = self.scale_type
        self.param_dict["Crossover Type"] = self.xover_type
        self.param_dict["Crossover Rate"] = self.xover_rate
        self.param_dict["Mutation Type"] = self.mutation_type
        self.param_dict["Mutation Rate"] = self.mutation_rate
        self.param_dict["Random Number Seed"] = self.seed
        self.param_dict["Number of Genes/Points"] = self.num_genes
        self.param_dict["Size of Genes"] = self.gene_size
        self.param_dict["Min or Max Fitness"] = self.min_or_max 
        self.param_dict["Sparsity"] = self.sparsity
        self.param_dict["Number of Islands"] = self.num_islands
        self.param_dict["Crossover Distribution Type"] = self.xover_dist_type
        self.param_dict["Mutation Distribution Type"] = self.mut_dist_type
            
            
    def get_parameter_dictionary(self):
        return self.param_dict

    def set_parameter_dictionary(self, changed_dict):
        self.param_dict = changed_dict
        self.reset_parameters()
        
    def reset_parameters(self):
        self.exp_id = self.param_dict["Experiment ID"]
        self.problem_type = self.param_dict["Problem Type"]
        self.data_input_file_name = self.param_dict["Data Input File Name"]
        self.num_runs = self.param_dict["Number of Runs"]
        self.generations = self.param_dict["Generations per Run"]
        self.pop_size = self.param_dict["Population Size"]
        self.select_type = self.param_dict["Selection Method"]
        self.scale_type = self.param_dict["Fitness Scaling Type"]
        self.xover_type = self.param_dict["Crossover Type"]
        self.xover_rate = self.param_dict["Crossover Rate"]
        self.mutation_type = self.param_dict["Mutation Type"]
        self.mutation_rate = self.param_dict["Mutation Rate"]
        self.seed = self.param_dict["Random Number Seed"]
        self.num_genes = self.param_dict["Number of Genes/Points"]
        self.gene_size = self.param_dict["Size of Genes"]
        self.min_or_max = self.param_dict["Min or Max Fitness"]    
        self.sparsity = self.param_dict["Sparsity"]
        self.num_islands = self.param_dict["Number of Islands"]
        self.xover_dist_type = self.param_dict["Crossover Distribution Type"]
        self.mut_dist_type = self.param_dict["Mutation Distribution Type"]
            
            
    def output_parameters(self, summary_output):
        """ Write parameter values to file """        
        
        summary_output.write("Experiment ID: " + self.exp_id + "\n")
        summary_output.write("Problem Type: " + self.problem_type + "\n")
        
        summary_output.write("Data Input File Name: " + self.data_input_file_name + "\n")
        
        summary_output.write("Number of Runs: " + str(self.num_runs) + "\n")
        summary_output.write("Generations per Run: " + str(self.generations) + "\n")
        summary_output.write("Population Size: " + str(self.pop_size) + "\n")
        
        summary_output.write("Selection Method: " + str(self.select_type) + "\n")
        summary_output.write("Fitness Scaling Type: " + str(self.scale_type) + "\n")
        summary_output.write("Min or Max Fitness: " + str(self.min_or_max) + "\n")
        
        summary_output.write("Crossover Type: " + str(self.xover_type) + "\n")
        summary_output.write("Crossover Rate: " + str(self.xover_rate) + "\n")
        summary_output.write("Mutation Type: " + str(self.mutation_type) + "\n")
        summary_output.write("Mutation Rate: " + str(self.mutation_rate) + "\n")
        
        summary_output.write("Random Number Seed: " + str(self.seed) + "\n")
        summary_output.write("Number of Genes/Points: " + str(self.num_genes) + "\n")
        summary_output.write("Size of Genes:" + str(self.gene_size) + "\n")
        
        summary_output.write("Tournament Size: " + str(self.tourney_size) + "\n")
        summary_output.write("Tournament Threshold: " + str(self.tourney_thresh) + "\n")
        summary_output.write("Elitism Rate: " + str(self.elitism_rate) + "\n")
        summary_output.write("Training Files: " + str(self.training_files) + "\n")
        
        summary_output.write("Sparsity: " + str(self.sparsity) + "\n")
        summary_output.write("Number of Islands: " + str(self.num_islands) + "\n")
        summary_output.write("Crossover Distribution Type: " + str(self.xover_dist_type) + "\n")
        summary_output.write("Mutation Distribution Type: " + str(self.mut_dist_type) + "\n")
        
        
        summary_output.write("\n\n")