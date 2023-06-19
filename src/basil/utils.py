import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Result:
    system_info: dict
    solver: str
    options: dict
    parameters: dict
    results: dict


@dataclass
class BenchmarkFamily:
    name: str
    results: List[Result] = None

    @classmethod
    def from_dict(cls, name, data):
        results = []
        for result in data:
            results.append(Result(**result))
        return cls(name, results)


def load_results(directory):
    # Get the list of all json files in the directory
    json_files = [
        pos_json for pos_json in os.listdir(directory)
        if pos_json.endswith('.json')
    ]
    print(os.listdir(directory))

    # For each file
    results = []
    for json_file in json_files:
        # Load in a dictionary
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)

        # Get the name of the benchmark family
        name = json_file.split('.')[0]

        # Create a BenchmarkFamily object
        benchmark_family = BenchmarkFamily.from_dict(name, data)

        # Add it to the list
        results.append(benchmark_family)

    return results


def compute_pareto_frontier(df,
                            x_axis,
                            y_axis,
                            x_better="Higher is better",
                            y_better="Higher is better"):
    """
    Compute the Pareto frontier for a given dataset and objectives
    """
    data_points = df[[x_axis, y_axis]].values

    if x_better == "Lower is better":
        data_points[:, 0] = -data_points[:, 0]

    if y_better == "Lower is better":
        data_points[:, 1] = -data_points[:, 1]

    # Order points by x_axis
    data_points = data_points[data_points[:, 0].argsort()]

    # Compute Pareto frontier
    pareto_frontier = [data_points[0]]
    for point in data_points[1:]:
        if point[1] >= pareto_frontier[-1][
                1]:    # Check if the point y value is greater than last point y
            pareto_frontier.append(point)

    return np.array(pareto_frontier)


def camel_to_snake_case(name):
    return ''.join(['_' + c.lower() if c.isupper() else c
                    for c in name]).lstrip('_')
