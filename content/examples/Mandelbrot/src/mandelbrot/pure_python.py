from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from .types import Area, Dimensions


def mandelbrot_pure_python(dimensions: Dimensions, area: Area, max_iter: int = 20):
    """Returns an array containing the Mandelbrot fractal.

    This function does not use numpy to compute the fractal values.
    We still generate the nested lists using numpy since this is much simpler.
    """
    x = np.linspace(area.x_min, area.x_max, dimensions.width).reshape(1, -1)
    y = np.linspace(area.y_min, area.y_max, dimensions.height).reshape(-1, 1)
    c = (x + y * 1j).tolist()
    z = deepcopy(c)

    diverge_start = [[max_iter] * len(row) for row in z]
    for i_row, row in enumerate(z):
        for i_col in range(dimensions.width):
            for i in range(max_iter):
                z_val = row[i_col]
                new_val = z_val**2 + c[i_row][i_col]
                row[i_col] = new_val
                # Are we diverging?
                if z_val * np.conj(z_val) > 4:
                    # Do we start diverging in this step?
                    if diverge_start[i_row][i_col] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[i_row][i_col] = i
                    # Avoid diverging too much
                    row[i_col] = 2
    return diverge_start


def mandelbrot_numpy(dimensions: Dimensions, area: Area, max_iter: int = 20):
    """Returns an array containing the Mandelbrot fractal."""
    x = np.linspace(area.x_min, area.x_max, dimensions.width).reshape(1, -1)
    y = np.linspace(area.y_min, area.y_max, dimensions.height).reshape(-1, 1)
    c = x + y * 1j
    z = c
    diverge_start = np.full(z.shape, np.int64(max_iter), dtype=np.int64)

    for i in range(max_iter):
        z = z**2 + c
        # Who is diverging?
        diverge = z * np.conj(z) > 4
        # Who starts diverging this step?
        div_now = diverge & (diverge_start == max_iter)
        # Note the divergence start for points starting this step
        diverge_start[div_now] = i
        # Avoid diverging too much
        z[diverge] = 2

    return diverge_start


def plot_mandelbrot(
    mandel,
    cmap: str = "bone_r",
    max_value: int = 20,
):
    """Plots an image of the Mandelbrot fractal of size (h,w)"""

    mandel = np.minimum(max_value, mandel)
    plt.imshow(mandel, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()
