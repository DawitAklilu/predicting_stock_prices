# standard library
import random
import math
import heapq
import copy
import sys

# 3rd party
import numpy as np

# reference ahead of time for speedup
mpow = math.pow
log = math.log

class Fitness():
    def __init__(self, points: list) -> None:
        self.points = points
        
    def score(self, individual) -> float:
        #num_genes = int(len(individual.genes) / 2)
        #return sum([abs(eval(repr(individual)) - y) for x, y in enumerate(self.points)]) / math.log(num_genes) if num_genes else 1
        return sum([abs(eval(repr(individual)) - y) * (x ** 3) for x, y in enumerate(self.points, start=1)])
