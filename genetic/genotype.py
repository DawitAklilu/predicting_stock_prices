# standard library
import random
import math
import heapq
import copy
import sys
import statistics

# 3rd party
import numpy as np
import pandas as pd

# internal imports
from individuals import Individual, Individual2
from fitness import Fitness

mpow = math.pow
log = math.log

GENE_OPTION = ['x', 'pow(x, 2)', 'pow(x, 3)'] + [str(x) for x in range(1, 20)]
OPERATIONS = ['+', '-', '*', '/']
# hold five best solutions in the hall of fame in descending order or best first
HALL_OF_FAME = [-float('inf')] * 5 
    
# tournament selection
def selection_tournament(population: list, elite_number: int=2, tournament_size: int=3) -> list:
    """Select from the population using the popular tournament strategy"""
    if elite_number > len(population):
        raise Exception('selected elites must be less than population size')
    heapq.heapify(population)
    # get elite individuals
    elites = [heapq.heappop(population) for _ in range(elite_number)]
    # return elites to population for selection
    population.extend(elites)
    
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
    """Test if the functions and classes work together"""
    df = pd.read_csv('../data/GOOG_reversed.csv')
    df = df.iloc[500:520, :]
    print(df.head())
    window_size = 11
    percentage_off = []
    predicted_directions = []
    
    percentage_off_2 = []
    predicted_directions_2 = []
    percentage_off_3 = []
    predicted_directions_3 = []
    for window_end, window_start in enumerate(range(df.shape[0] - (window_size + 1)), start=window_size):
        y_vals = df['Close'].iloc[window_start:window_end]
        #x_vals = [i + 1 for i in range(len(y_vals))]
        x_future = window_size + 1
        y_actual = df['Close'].iloc[window_end + 1]
        lower_bound, upper_bound = min(y_vals), max(y_vals)
        one_third_sigma = (lower_bound - upper_bound) / 3
        
        fitness = Fitness(y_vals)
        population = [Individual2(fitness.score, param_range=[(lower_bound, upper_bound), (-one_third_sigma, one_third_sigma)], sigma=one_third_sigma, mutation_rate=0.4, crossover_rate=0.5) for _ in range(100)]
        
        # get num iterations to run
        num_generations = 10
        for _ in range(num_generations):
            population = selection_tournament(population, tournament_size=5)
            
        # show stats
        solution = min(population)
        x = x_future
        prediction = eval(repr(solution))
        print(f'--Window {window_start}--')
        percent_difference = abs(prediction - y_actual) / y_actual
        print(f'Guess: {prediction}, Actual: {y_actual}, Percent: {percent_difference}')
        previous_day = y_vals.iloc[-1]
        print(f'Previous: {previous_day}')
        did_increase = True if y_actual - previous_day > 0 else False
        predicted_increase = True if prediction - previous_day > 0 else False
        if did_increase == predicted_increase:
            print('Prediction direction correct')
            predicted_directions.append(True)
        else:
            print('Prediction direction wrong')
            predicted_directions.append(False)
        print(f'Solution fitness: {solution.score}')
        percentage_off.append(percent_difference)
    
        # try with two terms
        population = [Individual2(fitness.score, terms=['1', 'x', 'mpow(x, 2)'], param_range=[(lower_bound, upper_bound), (-one_third_sigma, one_third_sigma)], sigma=one_third_sigma, mutation_rate=0.4, crossover_rate=0.5) for _ in range(100)]
        
        # get num iterations to run
        num_generations = 10
        for _ in range(num_generations):
            population = selection_tournament(population, tournament_size=5)
            
        # show stats
        solution = min(population)
        x = x_future
        prediction = eval(repr(solution))
        print(f'--Window {window_start}--')
        percent_difference = abs(prediction - y_actual) / y_actual
        print(f'Guess: {prediction}, Actual: {y_actual}, Percent: {percent_difference}')
        previous_day = y_vals.iloc[-1]
        print(f'Previous: {previous_day}')
        did_increase = True if y_actual - previous_day > 0 else False
        predicted_increase = True if prediction - previous_day > 0 else False
        if did_increase == predicted_increase:
            print('Prediction direction correct')
            predicted_directions_2.append(True)
        else:
            print('Prediction direction wrong')
            predicted_directions_2.append(False)
        print(f'Solution fitness: {solution.score}')
        percentage_off_2.append(percent_difference)
        
        # try with one terms
        population = [Individual2(fitness.score, terms=['1', 'x'], param_range=[(lower_bound, upper_bound), (-one_third_sigma, one_third_sigma)], sigma=one_third_sigma, mutation_rate=0.4, crossover_rate=0.5) for _ in range(100)]
        
        # get num iterations to run
        num_generations = 10
        for _ in range(num_generations):
            population = selection_tournament(population, tournament_size=5)
            
        # show stats
        solution = min(population)
        x = x_future
        prediction = eval(repr(solution))
        print(f'--Window {window_start}--')
        percent_difference = abs(prediction - y_actual) / y_actual
        print(f'Guess: {prediction}, Actual: {y_actual}, Percent: {percent_difference}')
        previous_day = y_vals.iloc[-1]
        print(f'Previous: {previous_day}')
        did_increase = True if y_actual - previous_day > 0 else False
        predicted_increase = True if prediction - previous_day > 0 else False
        if did_increase == predicted_increase:
            print('Prediction direction correct')
            predicted_directions_3.append(True)
        else:
            print('Prediction direction wrong')
            predicted_directions_3.append(False)
        print(f'Solution fitness: {solution.score}')
        percentage_off_3.append(percent_difference)
        
        
    print(f'Mean difference: {statistics.mean(percentage_off)}')
    print(f'Median difference: {statistics.median(percentage_off)}')
    print(f'Standard deviation of differences: {statistics.stdev(percentage_off)}')
    print(predicted_directions)
    
    print(f'Mean difference: {statistics.mean(percentage_off_2)}')
    print(f'Median difference: {statistics.median(percentage_off_2)}')
    print(f'Standard deviation of differences: {statistics.stdev(percentage_off_2)}')
    print(predicted_directions_2)
    
    print(f'Mean difference: {statistics.mean(percentage_off_3)}')
    print(f'Median difference: {statistics.median(percentage_off_3)}')
    print(f'Standard deviation of differences: {statistics.stdev(percentage_off_3)}')
    print(predicted_directions_3)

if __name__ == '__main__':
    main()
