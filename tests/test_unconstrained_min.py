import unittest
import numpy as np
from src.unconstrained_min import LineSearch
from tests.examples import func1, func2, func3, rosenbrock, linear, func4
from src.utils import plot_contour, plot_convergence

class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        self.functions = [
            ("Func1", func1, [-2, 2], [-2, 2]),
            ("Func2", func2, [-5, 5], [-5, 5]),
            ("Func3", func3, [-5, 5], [-5, 5]),
            ("Rosenbrock", rosenbrock, [-5, 5], [-5, 5]),
            ("Linear", linear, [0, 2], [0, 2]),
            ("Func4", func4, [-5, 5], [-5, 5])
        ]
        self.methods = ["GD", "Newton"]

    def test_optimization(self):
        for func_name, func, x_range, y_range in self.functions:
            minimizers = {}
            results = {}
            paths = {}

            for method in self.methods:
                # Skip Newton for Linear since Hessian is singular
                if func_name == "Linear" and method == "Newton":
                    continue

                if func_name == "Rosenbrock" and method == "GD":
                    max_iter = 10000
                else:
                    max_iter = 100

                if func_name == 'Rosenbrock':
                    x0 = np.array([-1.0, 2.0])
                else:
                    x0 = np.array([1.0, 1.0])

                obj_tol = 1e-12
                param_tol = 1e-8

                minimizer = LineSearch(func)
                result = minimizer.minimize(
                    x0,
                    method=method,
                    obj_tol=obj_tol,
                    param_tol=param_tol,
                    max_iter=max_iter
                )
                minimizers[method] = minimizer
                results[method] = result
                path = np.array([point[0] for point in getattr(minimizer, 'path', [])])
                paths[method] = path
                if result['success']:
                    print(f"{method} method converged for {func_name} at iteration {result['iter']} at x={result['x']}")

                # Unit test assertions (per method, so failures are specific)
                with self.subTest(func=func_name, method=method):
                    self.assertTrue(result['success'],
                        f"{method} method failed to converge for {func_name}")
                    self.assertTrue(np.isfinite(result['f']),
                        f"{method} method produced non-finite result for {func_name}")

            # Plotting section (per function)
            try:
                # Only plot if at least one method succeeded
                plot_contour(
                    func=func,
                    x_range=x_range,
                    y_range=y_range,
                    title=f'Contour Plot with Optimization Paths - {func_name}',
                    path_dict=paths  # multiple methods, named
                )
                plot_convergence(minimizers)
            except Exception as e:
                print(f"Plotting failed for {func_name}: {e}")

if __name__ == '__main__':
    unittest.main()