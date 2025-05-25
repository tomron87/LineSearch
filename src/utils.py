import numpy as np
import matplotlib.pyplot as plt

def plot_contour(
    func,
    x_range,
    y_range,
    title="Contour Plot",
    path_dict=None,
    x_label="x1",
    y_label="x2",
    levels=30,
):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)

    # Compute Z for each grid point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))[0]
            except Exception:
                Z[i, j] = np.nan

    finite_Z = Z[np.isfinite(Z)]
    # If all Z > 0, use logspace for levels for exponentials
    if np.all(finite_Z > 0):
        vmin = np.percentile(finite_Z, 5)
        vmax = np.percentile(finite_Z, 95)
        levels_arr = np.logspace(np.log10(vmin), np.log10(vmax), levels)
    else:
        vmin = np.percentile(finite_Z, 5)
        vmax = np.percentile(finite_Z, 95)
        levels_arr = np.linspace(vmin, vmax, levels)

    plt.figure(figsize=(7, 5))
    contour = plt.contour(X, Y, Z, levels=levels_arr, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.title(title)
    plt.colorbar(contour)

    # Plot algorithm paths if provided
    if path_dict is not None:
        for alg_name, path in path_dict.items():
            if path is not None and len(path) > 0:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], marker='o', label=alg_name, linewidth=2, markersize=5)
        plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_convergence(methods_results):
    """
    Plots function value versus iteration for different optimization methods,
    all on the same (single) plot for comparison.

    Parameters
    ----------
    methods_results : dict
        Keys: method names (str).
        Values: LineSearch instances, each with .path containing (x, obj_val) per iteration.
    """
    plt.figure(figsize=(10, 6))
    has_data = False
    for method_name, minimizer in methods_results.items():
        if hasattr(minimizer, "path") and minimizer.path:
            obj_values = [val for (_, val) in minimizer.path]
            plt.semilogy(range(len(obj_values)), obj_values,
                         label=method_name, marker='o', markersize=3)
            has_data = True
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (log scale)')
    plt.title('Convergence of Optimization Methods')
    plt.grid(True, which='both', linestyle='--')
    if has_data:
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("[plot_convergence] No data to plot.")