# standard library
import random
import math
import heapq
import copy
import sys

# 3rd party
import numpy as np

mpow = math.pow
log = math.log

GENE_OPTION = ['x', 'mpow(x, 2)', 'mpow(x, 3)'] + [str(x) for x in range(1, 20)]
OPERATIONS = ['+', '-', '*', '/']

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
    def __init__(self, fitness: callable, gene_params: list=[], terms: list=['1', 'x', 'mpow(x, 2)', 'mpow(x, 3)'], mutation_rate: float=0.1, crossover_rate: float=0.5, param_range: list=[(-10, 10)], mu: float=0, sigma: float=0.5):
        self.fitness_function = fitness
        self.genes = gene_params
        self.terms = terms
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.param_range = param_range
        # extend param range if not full
        if len(self.param_range) < len(terms):
            difference =  len(terms) - len(self.param_range)
            self.param_range.extend([self.param_range[-1]] * 3)
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
