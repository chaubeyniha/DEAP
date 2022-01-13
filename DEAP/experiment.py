# understand lambda, map, list, role of arrays
from tqdm import tqdm
import sys, os
sys.path.insert(0, 'evoman')
import numpy as np
import random
from evoman.environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools
import json
from datetime import datetime

# Starting pop size = 100
pop_size = 100

# Starting gen size = 20
gens = 20

# Population increase value should stay at 1 IF and only IF you are testing EA 1
# else set to 0
pop_increase_value = 1

# Enemy group 1
enemy_group1 = [2, 5, 7]
# Enemy group 2
enemy_group2 = [3, 6, 7]


# This should not be changed
n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name= 'test',
                  enemies=enemy_group1,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  multiplemode="yes",
                  logs="off")

# number of weights for multilayer with 10 hidden neurons
n_weight = (env.get_num_sensors()+1)* n_hidden_neurons + (n_hidden_neurons+1)*5

# create individual
# inherit from Numpy allows individuals to use properties from Numpy
# Since we want to maximize we use positive weight (single - objective)
creator.create('FitnessMax', base.Fitness, weights = (1.0, 1.0))
creator.create('Individual', np.ndarray, fitness = creator.FitnessMax)

# toolbox for EA
toolbox = base.Toolbox()

toolbox.register('attribute_init', np.random.normal)
# make a random number of individuals by calling toolbox.individual()
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute_init, n = n_weight)
# Calling toolbox.population() will readily return a complete population in a lis
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

ind1 = toolbox.individual() # can print as base class representation (array)
# to check if fitness is valid = print ind1.fitness.valid

# simulatin takes env.play is played with individual x (weight as an array) --> returns fitness
def simulation(env,x):
    fitness,_player,_enemy,_time = env.play(pcont= np.array(x))
    return fitness

# run simulation with env & x (returns fitness) --> just run the simulation
def evaluate(x):
    return simulation(env,x)

toolbox.register("evaluate", evaluate)

# if is_ea_1_currently_test:
toolbox.register("mate", tools.cxTwoPoint)
# else:
#     toolbox.register("mate", tools.cxOnePoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def generation_loop(fitnesses, mate_prob, mut_prob, old_population, pop_increase_value=1):

    # Set fitnesses to individuals
    # assignment has to be list or tuple because because fitness values is are a list
    for ind, fit in zip(old_population, fitnesses):
        ind.fitness.values = [fit]
    # select next generation of offspring
    offspring = toolbox.select(individuals=old_population, k=len(old_population)+pop_increase_value)
    # clone the selected individuals, independent instance
    # map returns a generator which is then listed
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < mate_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mut_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Re-evaluate individuals after crossover & mutation
    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    # fitnesses = map(toolbox.evaluate, invalid_ind)
    # for ind, fit in zip(invalid_ind, fitnesses):
    #     ind.fitness.values = [fit]

    # The population is entirely replaced by the offspring
    old_population[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    # fits = [ind.fitness.values[0] for ind in old_population]

    # best = pop[np.argmin([toolbox.evaluate(x) for ind in pop])]
    pop = old_population
    return pop

def main():
    # list of individuals
    pop = toolbox.population(n=pop_size)
    mate_prob = 0.5  # probability at which two individuals are crossed
    mut_prob = 0.2  # probability at which an individual is mutated

    # evaluate fitness for population
    # same order in fitnesses and pop

    # Define the values we will keep track of
    avg_fitness_per_gen = []
    max_fitness_per_gen = []
    weight_delta_per_gen = []
    best_individual = None

    # loop the number of generations
    for g in tqdm(range(gens)):
        fitnesses = list(map(toolbox.evaluate, pop))
        for i in range(len(fitnesses)):
            if best_individual is None:
                best_individual = [fitnesses[i], pop[i]]
            elif best_individual[0] <= fitnesses[i]:
                best_individual = [fitnesses[i], pop[i]]
        pop = generation_loop(fitnesses, mate_prob, mut_prob, pop, pop_increase_value)
        current_generation_weights = np.zeros((len(pop), len(pop)))

        # Keep track of below values
        avg_fitness_per_gen.extend([np.mean(fitnesses)])
        max_fitness_per_gen.extend([max(fitnesses)])
        for individual1 in range(len(pop)):
            for individual2 in range(len(pop)):
                current_generation_weights[individual1,individual2] = np.average(np.subtract(pop[individual1], pop[individual2]))
        weight_delta_per_gen.extend([current_generation_weights])
        # print(np.average(np.subtract(pop[0], pop[1])))
    print(weight_delta_per_gen)
    return avg_fitness_per_gen, max_fitness_per_gen, weight_delta_per_gen, best_individual


# We need to do 4 big tests in total:
# Two EAs for Two enemy groups consisting for 2-3 enemies.
# Each person in team tests 1 EA for 1 enemy group. Does this 10 times.
#
avg_fit_all = {}
max_fit_all = {}
weight_delta_all = []
avg_run_time = {}
best_ind_all = None
# 10 runs
# tqdm is a visual thing
for i in tqdm(range(1, 11)):
    start = datetime.timestamp(datetime.now())
    avg_fit, max_fit, weight_delta, best_ind = main()
    end = datetime.timestamp(datetime.now())
    avg_fit_all[i] = avg_fit
    max_fit_all[i] = max_fit
    weight_delta_all.extend(weight_delta)
    avg_run_time[i] = end-start
    if best_ind_all is None:
        best_ind_all = best_ind
    elif best_ind_all[0] < best_ind[0]:
        best_ind_all = best_ind

#EA_1_Run_Niha_67.8528528.txt

with open('test/avg_fitness_all.json', 'w') as fp:
    json.dump(avg_fit_all, fp, indent=4)

with open('test/max_fitness_all.json', 'w') as fp:
    json.dump(max_fit_all, fp, indent=4)

for weight in range(len(weight_delta_all)):
    # weight_delta_all[weight].tofile(f'test/weight_delta_{weight}.txt')
    np.save(f'test/weight_delta_{weight+1}.txt', weight_delta_all[weight])

with open('test/avg_run_time_all.json', 'w') as fp:
    json.dump(avg_run_time, fp, indent=4)

np.savetxt(f'test/best_individual_{best_ind_all[0]}.txt', best_ind_all[1])

# print("  Max %s" % max(fits))
# print("  Avg %s" % mean) --> mean = sum(fits) / length

# iterate main several times (20)
# create a 2 dimensional list in excel, repeat main and then append to the list so you can compare populations
# first time = randomly generated population, then use the population best one is selected in next population,
# worst 3 deleted and then next population baseed on this
# end iteration = generation is reached (20) or when avg fitness of the population
