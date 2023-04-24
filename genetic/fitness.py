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
        """Create a fitness object"""
        self.points = points
        
    def score(self, individual) -> float:
        """Score the individual using median absolute difference"""
        return sum([abs(eval(repr(individual)) - y) * (x ** 3) for x, y in enumerate(self.points, start=1)])
