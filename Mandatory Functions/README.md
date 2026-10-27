# Mandatory Functions

`implementations.py` contains the six required Project 1 methods. Each returns
`(w, loss)`, with one-dimensional weights and the unregularized loss evaluated
at the final weights. Only NumPy is required.

The implementation adapts these local course solutions:

- `labs/ex02/solution/`: half-MSE, its gradient, GD, and SGD. With batch size one,
  `helpers.batch_iter` draws a uniform index independently for each update using
  `np.random.randint`. Call `np.random.seed(...)` before SGD for reproducibility.
- `labs/ex04/solution/least_squares.py` and `ridge_regression.py`: normal equations
  solved with `np.linalg.solve`; ridge adds `2 * N * lambda_ * I`.
- `labs/ex05/solution/ex05.ipynb`: mean binary negative log likelihood, its mean
  gradient, and the L2 gradient `2 * lambda_ * w`. Logistic calculations use
  equivalent stable expressions, adapted to one-dimensional vectors.

All iterative methods use exactly `max_iters` updates with the given `gamma`.
All coefficients, including an intercept if supplied in `tx`, are regularized.
The direct solvers require nonsingular normal-equation matrices, as in the labs.

From this folder, the required imports work directly:

```python
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
```

Run the deterministic checks from the repository root:

```text
python -B -m unittest discover -s "Mandatory Functions" -p "test_*.py" -v
```

No Project 1 PDF or official Project 1 grading tests were present when this was
implemented; the local labs provide the course conventions and reference cases.
For submission, place this folder's `implementations.py` at the location expected
by the course grader.
