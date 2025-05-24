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
    """
    Plots contour lines of a 2D objective function over specified axes limits.
    If algorithm paths are provided, overlays them with names in the legend.

    Parameters
    ----------
    func : callable
        Objective function taking a 1D numpy array as input and returning a list whose first element is the function value.
    x_range, y_range : [float, float]
        Ranges for x and y axes, e.g. [-2, 2].
    title : str
        Plot title. Should include function name for clarity.
    path_dict : dict or None
        If provided, should be a dict mapping algorithm names (str) to Nx2 arrays of path points.
        E.g. {'Gradient Descent': path_gd, 'Newton': path_newton}
    x_label, y_label : str
        Axis labels.
    levels : int
        Number of contour levels to plot.
    """

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

    # Choose contour levels to focus on interesting area (remove outliers, auto-scale)
    finite_Z = Z[np.isfinite(Z)]
    vmin = np.percentile(finite_Z, 1)
    vmax = np.percentile(finite_Z, 99)
    levels_arr = np.linspace(vmin, vmax, levels)

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=levels_arr, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Plot algorithm paths if provided
    if path_dict is not None:
        for alg_name, path in path_dict.items():
            if path is not None and len(path) > 0:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], marker='o', label=alg_name, linewidth=2, markersize=4)

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