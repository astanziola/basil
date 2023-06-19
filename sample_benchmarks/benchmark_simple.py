from time import time

import jax
import numpy as np
from jax import numpy as jnp

from basil import Benchmark, Solver, parametrize, register_benchmark


# Define the benchmark
class SimpleBenchmark(Benchmark):

    def __init__(self, N):
        self.N = N

    @property
    def parameters(self):
        return {"num_values": self.N}

    @property
    def inputs(self):
        matrix = np.random.rand(self.N, self.N)
        ref_value = np.sum(matrix)
        return {"inputs": {"matrix": matrix}, 'aux': ref_value}

    def evaluate(self, outputs, ref_value):
        return {
            "error": abs(outputs["result"] - ref_value),
            "execution_time": outputs["execution_time"]
        }


# Define benchmark solver
class JaxSolver(Solver):

    def run(self, inputs):
        func = jax.jit(lambda x: jnp.sum(x))

        # Compile the function
        mock_inputs = jnp.ones_like(inputs["matrix"])
        _ = func(mock_inputs)

        # Run the function
        x = jnp.asarray(inputs["matrix"])
        start = time()
        result = func(x).block_until_ready()
        execution_time = time() - start
        return {"result": float(result), "execution_time": execution_time}


# Define the benchmark instances
@register_benchmark
@parametrize("N", [100, 1000])
def sum_test(N: int):
    return {"benchmark": SimpleBenchmark(N), "solver": JaxSolver()}
