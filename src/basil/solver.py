import datetime
import json
import os
import platform
import subprocess

import psutil

from basil.utils import camel_to_snake_case


class Solver(object):

    def setup(self, inputs):
        pass

    @property
    def name(self):
        return camel_to_snake_case(self.__class__.__name__)

    @property
    def options(self):
        return {}

    def collect_system_info(self):
        info = {}

        # OS and Processor Info
        info["OS"] = platform.system()
        info["OS_version"] = platform.version()
        info["CPU"] = platform.processor()

        # RAM Info
        ram_info = psutil.virtual_memory()
        info["RAM"] = ram_info.total

        # GPU Info - this works best on Unix systems, may not be portable
        try:
            gpu_info = subprocess.check_output("lspci | grep ' VGA '",
                                               shell=True).decode("utf8")
            info["GPU"] = gpu_info
        except Exception as e:
            info["GPU"] = "No GPU info"

        return info


class Run:
    run_at = None
    sys_info = None
    benchmark = None
    solver = None
    _results = None

    def __init__(self, benchmark, solver):
        self.run_at = datetime.datetime.now()
        self.sys_info = solver.collect_system_info()
        self.benchmark = benchmark
        self.solver = solver

    def run_benchmark(self):
        # Generate inputs and setup solver
        input_and_aux = self.benchmark.inputs
        inputs, aux = input_and_aux["inputs"], input_and_aux["aux"]
        self.solver.setup(inputs)

        # Run benchmark
        outputs = self.solver.run(inputs)

        # Get results
        self.benchmark.validate_outputs(outputs)
        results = self.benchmark.evaluate(outputs, aux)

        # Add them to the object
        self.add_results(results)

        return results

    def add_results(self, results):
        # Construct the results dictionary
        self._results = {
            "system_info": self.sys_info,
            "solver": self.solver.name,
            "options": self.solver.options,
            "parameters": self.benchmark.parameters,
            "results": results
        }

    def save_results(self):
        import json
        import os

        benchmark_name = self.benchmark.name

        # Make sure the `results` folder exists, otherwise create it
        if not os.path.exists("results"):
            os.mkdir("results")

        # Check if the file exists, otherwise initialize it as
        # a list JSON object
        if not os.path.exists(f"results/{benchmark_name}.json"):
            with open(f"results/{benchmark_name}.json", "w") as f:
                f.write("[]")

        # Append the results to the file as a new entry of the list
        with open(f"results/{benchmark_name}.json", "r+") as f:
            # TODO: This method of loading and saving the file is not
            #      very efficient, should be probably changed by using
            #      a database
            results = json.load(f)
            results.append(self._results)
            # Write the results back to the file in a pretty format
            f.seek(0)
            json.dump(results, f, indent=2)
            f.truncate()

    def __call__(self):
        _ = self.run_benchmark()
        self.save_results()
        self.display_results()
