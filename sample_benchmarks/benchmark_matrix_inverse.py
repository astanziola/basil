# For timing functions
from time import time

import jax
import numpy as np
from jax import numpy as jnp

from basil import Benchmark, Solver, parametrize, register_benchmark


# Define the benchmark
class MatrixInverseSum(Benchmark):
    name: str = "matrix_inverse_sum"

    def __init__(self, N):
        self.N = N

    @property
    def parameters(self):
        return {"grid_size": self.N}

    @property
    def inputs(self):
        # Generate random matrix
        np.random.seed(2023)
        matrix = np.random.rand(self.N, self.N)

        # Find inverse
        inv_matrix = np.linalg.inv(matrix) + 3.14

        # Return inputs and auxiliary data for the benchmark
        return {
            "inputs": {
                "matrix": matrix
            },
            "aux": {
                "inv_matrix": inv_matrix
            }
        }

    def validate_outputs(self, outputs):
        # The output should be a dictionary with two keys, "inv_matrix" and "execution_time"
        assert isinstance(outputs, dict), "outputs should be a dictionary"
        assert "inv_matrix" in outputs, "outputs should contain the key 'inv_matrix'"
        assert "execution_time" in outputs, "outputs should contain the key 'execution_time'"

    def evaluate(self, outputs, aux):
        # Get the inverse matrix from the outputs and the auxiliary data
        pred_inv_matrix = outputs["inv_matrix"]
        ref_inv_matrix = aux["inv_matrix"]
        execution_time = outputs["execution_time"]

        # Compute the normalized error
        error = np.linalg.norm(pred_inv_matrix -
                               ref_inv_matrix) / np.linalg.norm(ref_inv_matrix)

        return {"error": error, "execution_time": execution_time}


# Define benchmark solvers
class MatrixInverse_JAX(Solver):

    def __init__(self, name: str = "jax.numpy.linalg.inv", jit: bool = True):
        self.name = name
        self.jit = jit

    @property
    def options(self):
        return {"jit": self.jit}

    def run(self, inputs):

        def func(x):
            return jnp.linalg.inv(x) + 3.14

        if self.jit:
            func = jax.jit(func)
            # Compile the function with some mock inputs
            mock_inputs = jnp.ones_like(inputs["matrix"])
            _ = func(mock_inputs)

        # Take the smallest of 50 runs
        execution_times = []
        x = jnp.asarray(inputs["matrix"])
        for _ in range(50):
            start = time()
            inv_matrix = func(x)
            execution_times.append(time() - start)

        execution_time = min(execution_times)

        return {"inv_matrix": inv_matrix, "execution_time": execution_time}


# Define benchmark solvers
class MatrixInverseLU(Solver):

    def __init__(
        self,
        name: str = "manual.lu_inverse",
    ):
        self.name = name

    def run(self, inputs):

        def func(x):
            # Manual implementation of matrix inverse via LU decomposition
            # https://en.wikipedia.org/wiki/LU_decomposition

            # Get the shape of the matrix
            N, _ = x.shape

            # Initialize the inverse matrix
            inv_matrix = np.zeros_like(x)

            # Loop over the columns
            for j in range(N):
                # Construct the RHS vector
                b = np.zeros(N)
                b[j] = 1.0

                # Solve the system
                inv_matrix[:, j] = np.linalg.solve(x, b)

            return inv_matrix + 3.14

        # Take the smallest of 50 runs
        execution_times = []
        x = np.asarray(inputs["matrix"])
        for _ in range(1):
            start = time()
            inv_matrix = func(x)
            execution_times.append(time() - start)

        execution_time = min(execution_times)

        return {"inv_matrix": inv_matrix, "execution_time": execution_time}


# Create instances
@register_benchmark
def small_matrix_inverse():
    return {
        "benchmark": MatrixInverseSum(30),
        "solver": MatrixInverse_JAX(jit=True)
    }


@register_benchmark
@parametrize("N", list(range(10, 600, 5)))
@parametrize("jit", [True, False])
def many_matrix_inverse(N: int, jit: bool):
    solver_name = "jax.numpy.linalg.inv_jit" if jit else "jax.numpy.linalg.inv"
    return {
        "benchmark": MatrixInverseSum(N),
        "solver": MatrixInverse_JAX(jit=jit, name=solver_name)
    }


@register_benchmark
@parametrize("N", list(range(100, 600, 50)))
def many_matrix_inverse(N: int):
    return {"benchmark": MatrixInverseSum(N), "solver": MatrixInverseLU()}
