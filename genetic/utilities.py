##############################
# Utility functions
##############################

# standard library
import typing

# 3rd party
import pandas as pd
import numpy as np


def get_indexes(data: iter, num_splits: int=3, num_days: int=39, min_index: int=1000) -> typing.List[tuple]:
    """Get a range of indexes depending on the data"""
    if len(data) < num_splits * num_days + min_index:
        raise Exception(f'not enough data to split')
    step = int((len(data) - min_index - num_days - 1) / (num_splits - 1))
    return [(x, x + num_days) for x in range(min_index, len(data), step)]

def compare_predictions(actual: list, predicted: list) -> list:
    """Count how many times the values match versus differ"""
    num_correct = num_incorrect = 0
    for real, prediction in zip(actual, predicted):
        if real == prediction:
            num_correct += 1
        else:
            num_incorrect += 1
    assert num_correct + num_incorrect == len(actual), 'Did not return the right number of correct and incorrect predictions'
    return num_correct, num_incorrect

def calculate_elapsed(start_time, end_time) -> str:
    """Return the elapsed time"""
    minutes, seconds = divmod(end_time - start_time, 60)
    return f'Elapsed {minutes} minutes {seconds} seconds'