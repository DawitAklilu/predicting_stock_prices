import random
import math
import heapq
import copy
import sys

mpow = math.pow
log = math.log

GENE_OPTION = ['x', 'pow(x, 2)', 'pow(x, 3)'] + [str(x) for x in range(1, 20)]
OPERATIONS = ['+', '-', '*', '/']
# hold five best solutions in the hall of fame in descending order or best first
HALL_OF_FAME = [-float('inf')] * 5 

class Individual():
    def __init__(self, genes: list=[], start_len: int=20, options: list=GENE_OPTION, operations: list=OPERATIONS, mutation_rate: float=0.1):
        self.genes = genes
        if start_len < 1:
            raise Exception('Start length must be greater than one')
        self.start_len = start_len
        self.options = options
        self.operations = operations
        if not self.genes:
            self.create_random_genotype() 
            
    def __repr__(self) -> str:
        return ''.join(self.genes) 

    # create a new genotype
    def create_random_genotype(self) -> None:
        self.genes.append(random.choice(self.options))  # first parameter
        for _ in range(random.randint(0, self.start_len)):
            self.genes.append(random.choice(self.operations))  # add operation
            self.genes.append(random.choice(self.options))     # add parameter

    # mutate existing genotype
    def mutate(self) -> None:
        pass

    # combine two parents together
    def recombine(self) -> None:
        pass
        
####
# 2
####            
class Individual2():
    def __init__(self, fitness: callable, gene_params: list=[], terms: list=['1', 'x', 'pow(x, 2)', 'pow(x, 3)'], mutation_rate: float=0.1, crossover_rate: float=0.5, param_range: list=[(-10, 10)], mu: float=0, sigma: float=0.5):
        self.fitness_function = fitness
        self.genes = gene_params
        self.terms = terms
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.param_range = param_range
        if len(self.param_range) < len(terms):
            difference =  len(terms) - len(self.param_range)
            self.param_range.extend([self.param_range[-1]] * 3)
        print(self.param_range)
        self.mu = mu
        self.sigma = sigma
        if not self.genes:
            self.create_random_genotype()
        self.score = float('inf')
        self.calculate_fitness()
            
    def __repr__(self) -> str:
        return ' + '.join([f'{coeff}*{term}' for coeff, term in zip(self.genes, self.terms)]) 
        
    def __lt__(self, other) -> bool:
        """higher number is less fitness"""
        return self.score < other.score
        
    def calculate_fitness(self) -> None:
        self.score = self.fitness_function(self)
        
    # create a new genotype
    def create_random_genotype(self) -> None:
        # hardcoded
        self.genes = [random.uniform(self.param_range[i][0], self.param_range[i][1]) for i in range(len(self.terms))]
        self.calculate_fitness()

    # mutate existing genotype
    def mutate(self):
        """
        Return reference to self
        """
        for coeff_index in range(len(self.terms)):
            if random.random() < self.mutation_rate:
                self.genes[coeff_index] += random.gauss(self.mu, self.sigma)
        self.calculate_fitness()      
        return self

    # combine two parents together
    def recombine(self, other):
        """
        Return deepcopy of other
        """
        offspring = copy.deepcopy(other)
        for i, (offspring_gene, self_gene) in enumerate(zip(offspring.genes, self.genes)):
            if random.random() < offspring.crossover_rate:
                offspring.genes[i] = (self_gene + offspring_gene) / 2
        offspring.calculate_fitness()
        return offspring
        
        
class Fitness():
    def __init__(self, points: list) -> None:
        self.points = points
        
    def score(self, individual) -> float:
        #num_genes = int(len(individual.genes) / 2)
        #return sum([abs(eval(repr(individual)) - y) for x, y in enumerate(self.points)]) / math.log(num_genes) if num_genes else 1
        return sum([abs(eval(repr(individual)) - y) for x, y in enumerate(self.points, start=1)])
        
# elite selection
def selection_elite(population: list, fitness: callable, truncation_percent: float=0.1, elites: int=5) -> list:
    pass
    
# tournament selection
def selection_tournament(population: list, elite_number: int=2, tournament_size: int=3) -> list:
    if elite_number > len(population):
        raise Exception('selected elites must be less than population size')
    heapq.heapify(population)
    # get elite individuals
    elites = [heapq.heappop(population) for _ in range(elite_number)]
    # return elites to population for selection
    population.extend(elites)
    print(*elites)
    
    selected_population = elites
    total_fitness = 0
    while len(selected_population) < len(population):
        parent_1 = min(random.choices(population, k=tournament_size))
        parent_2 = min(random.choices(population, k=tournament_size))
        # recombine and mutate
        offspring = parent_1.recombine(parent_2).mutate()
        total_fitness += offspring.score
        selected_population.append(offspring)
    print(total_fitness / len(selected_population))
    return selected_population
    
def main():
    y_vals = [253.92, 260.79, 265.44, 276.2, 279.43]#, 272.23, 273.78, 272.29, 277.66]  #, 280.57]
    full_y = [253.92, 260.79, 265.44, 276.2, 279.43, 272.23]#, 273.78, 272.29, 277.66, 280.57]
    lower_bound, upper_bound = min(y_vals), max(y_vals)
    x_vals = [1, 2, 3, 4, 5, 6]
    fitness = Fitness(y_vals)
    population = [Individual2(fitness.score, param_range=[(lower_bound, upper_bound), (-5, 5)], sigma=2, mutation_rate=0.4, crossover_rate=0.5) for _ in range(100)]
    
    # get num iterations to run
    num_generations = 50
    if len(sys.argv) > 1:
        num_generations = int(sys.argv[1])
    for _ in range(num_generations):
        print('=========Iteration:', _)
        population = selection_tournament(population)
    for solution in sorted(population)[:3]:
        print('solution:', solution)
        print([eval(repr(solution)) for x in x_vals])
        print(full_y)
        print(fitness.score(solution))
    

if __name__ == '__main__':
    main()
