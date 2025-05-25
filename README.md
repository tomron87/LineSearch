# Line Search Optimization Methods

This project implements and compares different line search optimization methods for unconstrained minimization problems. It provides implementations of Gradient Descent and Newton's method, along with visualization tools to analyze their performance.

## Features

- Implementation of two optimization methods:
  - Gradient Descent (GD)
  - Newton's Method
- Backtracking line search with Wolfe conditions
- Visualization tools:
  - Contour plots with optimization paths
  - Convergence plots comparing different methods
- Test suite with various test functions:
  - Quadratic functions (Func1, Func2, Func3)
  - Rosenbrock function
  - Linear function
  - Exponential function (Func4)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tomron87/LineSearch.git
cd LineSearch
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

To run the test suite and see the optimization results:

```bash
python -m unittest tests/test_unconstrained_min.py
```

This will:
- Run optimization on all test functions using both GD and Newton's method
- Generate contour plots showing the optimization paths
- Create convergence plots comparing the methods
- Print convergence information for each test case

### Using the Optimization Methods

Here's a basic example of how to use the optimization methods:

```python
from src.unconstrained_min import LineSearch
import numpy as np

# Define your objective function
def objective_function(x):
    # Function should return [f(x), gradient(x), hessian(x)]
    # For GD, hessian can be None
    return [f_x, gradient, hessian]

# Create optimizer instance
optimizer = LineSearch(objective_function)

# Run optimization
result = optimizer.minimize(
    x0=np.array([1.0, 1.0]),  # Initial point
    method='GD',              # or 'Newton'
    obj_tol=1e-12,           # Objective function tolerance
    param_tol=1e-8,          # Parameter tolerance
    max_iter=1000            # Maximum iterations
)

# Access results
print(f"Optimal point: {result['x']}")
print(f"Optimal value: {result['f']}")
print(f"Number of iterations: {result['iter']}")
print(f"Convergence: {result['success']}")
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── unconstrained_min.py  # Main optimization implementation
│   └── utils.py             # Visualization utilities
├── tests/
│   ├── __init__.py
│   ├── examples.py          # Test functions
│   └── test_unconstrained_min.py  # Test suite
├── requirements.txt
└── README.md
```