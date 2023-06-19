<div align="center">
  <img src="static/assets/logo_basil.png" width="100" align="center">
  <h1>Basil</h1>
</div>



__Benchmarking And Statistical Inspection Library__

⚠️ This library is still in development. Breaking changes may occur at any time.

## Install

Use `pip` to install the package:

```bash
pip install basil-benchmark
```

## Example

Define a benchmark to solve, a solver and the instances, which are made of a
pair of a benchmark and a solver.

```python
from time import time

import jax
import numpy as np
from jax import numpy as jnp

from basil import Benchmark, Solver, parametrize, register_benchmark

class SimpleBenchmark(Benchmark):    # Define the benchmark
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


class JaxSolver(Solver):    # Define benchmark solver
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
```

After saving this in a file that follows the pattern `benchmark_*.py` into some `BENCHMARKS_PATH` subfolder, invoke `basil` to run the benchmark instances.

```bash
basil BENCHMARKS_PATH
```

It should print something like this:

```
Adding simple_benchmark with jax_solver to register
Adding simple_benchmark with jax_solver to register
Running all 2 runs
Benchmark: simple_benchmark
Results: {'system_info': {'OS': 'Linux', 'OS_version': '#1 SMP Fri Jan 27 02:56:13 UTC 2023', 'CPU': 'x86_64', 'RAM': 16714588160, 'GPU': 'No GPU info'}, 'solver': 'jax_solver', 'options': {}, 'parameters': {'num_values': 100}, 'results': {'error': 0.0001370069894619519, 'execution_time': 0.00036644935607910156}}

Benchmark: simple_benchmark
Results: {'system_info': {'OS': 'Linux', 'OS_version': '#1 SMP Fri Jan 27 02:56:13 UTC 2023', 'CPU': 'x86_64', 'RAM': 16714588160, 'GPU': 'No GPU info'}, 'solver': 'jax_solver', 'options': {}, 'parameters': {'num_values': 1000}, 'results': {'error': 0.025644428096711636, 'execution_time': 0.0019795894622802734}}

All runs finished
```

The results are also saved in a json file, with the same name as the benchmark (in this case, it would be `simple_benchmark.json`).

## Visualize results

A prototype visualization via `streamlit` is provided, but it is not yet complete. To preview it in development mode, use

```bash
streamlit run src/basil/_dashboard.py -- --folder RESULTS_FOLDER
```
