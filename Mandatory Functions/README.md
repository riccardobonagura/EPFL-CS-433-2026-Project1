# Mandatory Functions

[`implementations.py`](../implementations.py), at the repository root where the
course autograder imports it, contains the six required Project 1 methods. Each returns
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

From the repository root, the required imports work directly:

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

The local labs provide the course conventions and reference cases. The tests in
this folder add the repository root to `sys.path` to import the graded module.
This folder keeps the experimental `implementations_augmented.py` and both test files.

## Experimental augmented API

`implementations_augmented.py` provides separate `fit_*` functions returning
`FitResult`, with optional sample weights, deterministic mini-batches, validation
stopping, intercept-penalty control, and prediction/threshold helpers. It does
not change or replace `implementations.py` or its autograder API.

See [README_augmented.md](README_augmented.md) for mathematical conventions,
selection behavior, and usage examples.
