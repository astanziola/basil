import importlib.util
import itertools
import os

from .solver import Run


def load_modules_from_directory(directory):
    # Traverse directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if file is a Python file and starts in 'benchmark_'
            if file.endswith('.py') and file.startswith('benchmark_'):
                file_path = os.path.join(root, file)
                # Import module
                load_module_from_file(file_path)


def load_module_from_file(file_path):
    # Create module name from file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    module = importlib.util.module_from_spec(spec)

    # Run the module
    spec.loader.exec_module(module)


class RunRegister:
    runs = []

    def add_run(self, run):
        self.runs.append(run)

    def run_all(self, save_results=True):
        print(f"Running all {len(self.runs)} runs")

        while len(self.runs) > 0:
            run = self.runs.pop(0)
            print(f"Benchmark: {run.benchmark.name}")

            _ = run.run_benchmark()

            print(f"Results: {run._results}\n")

            if save_results:
                run.save_results()

            # Delete the run object
            del run

        print("All runs finished")


# Instantiate the register
run_register = RunRegister()


class BenchmarkRegistration:

    def __init__(self, func=None, parameters={}):
        self.func = func
        self.parameters = parameters

    def add_run(self):
        # Make run
        definition = self.func(**self.parameters)
        benchmark, solver = definition["benchmark"], definition["solver"]
        run = Run(benchmark, solver)

        print(f"Adding {benchmark.name} with {solver.name} to register")
        run_register.add_run(run)


def register_benchmark(vals):
    if isinstance(vals, tuple):
        func, parameters = vals
    else:
        func = vals
        parameters = None

    if parameters is None:
        registration = BenchmarkRegistration(func)
        registration.add_run()
    else:
        for params in parameters:
            registration = BenchmarkRegistration(func, params)
            registration.add_run()


def parametrize(param_name, values):

    def decorator(vals):
        if isinstance(vals, tuple):
            func, params = vals
        else:
            func = vals
            params = [{}]
        # the `params` is a list of dictionaries. Generate a new
        # list of dictionaries with the new parameter added as a key.
        # This should generate all combinations for the previous parameters
        # and the new one.
        new_params = []
        for value in values:
            new_entry = {param_name: value}
            params_update = params.copy()
            params_update = [dict(p, **new_entry) for p in params_update]
            new_params.extend(params_update)

        return func, new_params

    return decorator
