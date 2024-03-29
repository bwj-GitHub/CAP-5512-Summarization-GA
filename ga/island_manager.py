# Island_Manager.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Homework 4

import random
import copy
import pickle

from ga.island import Island
from ga.parameters import Parameters
from ga.tldr import TLDR

from tools.file_tools import fopen

pop_size_min = 40
pop_size_max = 40

num_islands = 1
num_runs = 30
num_generations = 100
grid_w = 0
grid_h = 0
do_migrate = False

# Override parameter files (set to None to use params file values):
MUT_RATE_OVERIDE = 0.05
XOVER_RATE_OVERIDE = 0.8
POP_SIZE_OVERIDE = 40
SPARSITY_OVERIDE = 0.8

PARAM_FILE = "../Input/tldr.params"

def generate_neighbors(grid_h, grid_w):
    neighbors = []
    for isl in range(num_islands):
        neighborSet = []
        #first row
        if isl < grid_w:
            neighborSet.append(isl + grid_w*(grid_h - 1))
            neighborSet.append(isl + grid_w)

        #last row
        elif isl >= grid_h*grid_w - grid_w:
            neighborSet.append(isl - grid_w*(grid_h - 1))
            neighborSet.append(isl - grid_w)

        else:
            neighborSet.append(isl + grid_w)
            neighborSet.append(isl - grid_w)
            
        #first column
        if isl % grid_w == 0:
            neighborSet.append(isl + grid_w - 1)
            neighborSet.append(isl + 1)

        #last column
        elif isl % grid_w == grid_w - 1:
            neighborSet.append(isl - grid_w + 1)
            neighborSet.append(isl - 1)
            
        else:
            neighborSet.append(isl+1)
            neighborSet.append(isl-1)
        neighbors.append(neighborSet)

    return neighbors

#Key for distribution types:
#0 represents uniform distribution
#1 represents logarithmic uniform distribution (min exponent of -4)
#2 represents normal distribution
def generate_mut_rate(distrib_type):
    if MUT_RATE_OVERIDE is not None:
        return MUT_RATE_OVERIDE
    elif distrib_type == 0:
        return random.uniform(0, 1)
    elif distrib_type == 1:
        exponent = float(random.randint(-4, -1))
        factor = float(random.randint(1, 9))
        return (10.0**exponent) * factor

def generate_xover_rate(distrib_type):
    if XOVER_RATE_OVERIDE is not None:
        return XOVER_RATE_OVERIDE
    elif distrib_type == 0:
        return random.uniform(0, 1)
    elif distrib_type == 1:
        exponent = float(random.randint(-4, -1))
        factor = float(random.randint(1, 9))
        return (10.0**exponent) * factor

def generate_pop_size():
    if POP_SIZE_OVERIDE is not None:
        return POP_SIZE_OVERIDE
    else:
        return random.randint(pop_size_min, pop_size_max)

def generate_sparsity():
    if SPARSITY_OVERIDE is not None:
        return SPARSITY_OVERIDE
    else:
        return random.random()


def pickle_tldr_vecs(params_file="../Input/tldr.params", save_as="../Input/tldr1-27.p"):
    params = Parameters(params_file)
    tldr = TLDR(params)
    to_pik = (tldr.raw_sents, tldr.vec_sents)
    with fopen(save_as, "wb") as fp:
        pickle.dump(to_pik, fp, protocol=2)


def main():

    islands = []
    neighbors = []

    # Load params file
    #file_name = "../Input/tldr.params"
    file_name = PARAM_FILE
    params_object = Parameters(file_name)
    parms = params_object.get_parameter_dictionary()
    print("Loaded params file: " + file_name)
    
    # Create shared TLDR object
    #tldr = TLDR(params_object)
    my_pickle_file = "../Input/" + str(parms["Data Input File Name"])
    with fopen(my_pickle_file, "rb") as fp:
        vecs = pickle.load(fp)
    tldr = TLDR(params_object, sents=vecs)

    # Load/Create run seed to keep all island summary files together:
    try:
        with open("../Input/run_seed.txt", "r") as fp:
            r_seed = int(fp.read())
    except IOError:
        r_seed = 0
    print(r_seed)

    # Update run seed:
    with fopen("../Input/run_seed.txt", "w") as fp:
        fp.write(str(r_seed + 1))
    
    # Create number of islands
    num_islands = parms["Number of Islands"]
    num_runs = parms["Number of Runs"]
    num_generations = parms["Generations per Run"]
    
    # Initialize local GAs
    for n in range(num_islands):
        parms["Mutation Rate"] = generate_mut_rate(parms["Mutation Distribution Type"])
        parms["Crossover Rate"] = generate_xover_rate(parms["Crossover Distribution Type"])
        parms["Population Size"] = generate_pop_size()
        parms["Sparsity"] = generate_sparsity()

        # Deep copy stuff
        params_temp_object = copy.deepcopy(params_object)
        params_temp_object.set_parameter_dictionary(parms)        
        islands.append(Island(params_temp_object, file_name, tldr, r_seed))
        
    # Establish neighbors
    # TODO: Allow for any number of islands...
    if do_migrate == True and num_islands == 4:
        neighbors.append([1, 2, 3])
        neighbors.append([0, 2, 3])
        neighbors.append([0, 1, 3])
        neighbors.append([0, 1, 2])
    elif do_migrate and num_islands == 12:
        neighbors = generate_neighbors(4, 3)
            
    elif do_migrate == True and num_islands == 20:
        neighbors = generate_neighbors(5, 4)
            
    elif do_migrate == True and num_islands == 50:
        neighbors = generate_neighbors(5, 10)
        
    elif do_migrate == True:
        print("Using invalid number of islands!")
        return

    # Number of times this set of islands will run        
    for i in range(num_runs):

        # Start runs
        for n in range(num_islands):
            print("Starting Island #" + str(n))
            islands[n].start_run()

        # Run generations round-robin style
        migratingIndvs = {}
        for j in range(num_generations):
            
            # Migrate
            if do_migrate == True and j > 0:
                bestIndvs = []
                for n in range(num_islands):
                    bestIndvs.append(islands[n].get_best_of_gen())
                #Dictionary in which the key indicates the island index
                #The entry will be a list of individuals migrating to that island
                migratingIndvs = {}

                for n in range(len(bestIndvs)):
                    #Populate the lists of migrating individuals with empty lists
                    migratingIndvs[n] = []

                for n in range(len(bestIndvs)):
                    #Determine where each best individual goes
                    destinationIndex = random.choice(neighbors[n])
                    temp_migrants = migratingIndvs[destinationIndex]
                    temp_migrants.append(bestIndvs[n])
                    migratingIndvs[destinationIndex] = temp_migrants                

            # Run the next generation
            print("Gen #" + str(j))
            for n in range(num_islands):
                if do_migrate == True and j > 0:
                    migrants = []                
                    for migrant in migratingIndvs[n]:
                        migrants.append(migrant)
                    islands[n].run_next_generation(migratingIndvs[n])
                else:
                    islands[n].run_next_generation()

        # Finish runs
        for n in range(num_islands):
            islands[n].finish_run()

    # Everyone dies
    for n in range(num_islands):
        islands[n].shut_down()

if __name__ == "__main__":
    print("Extractive Summarization Problem - TLDR GA Initialization")
#     pickle_tldr_vecs()
    main()
