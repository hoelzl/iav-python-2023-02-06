from numba import njit, prange
import numpy as np
from .types import Area, Dimensions


@njit(cache=True)
def mandelbrot_numba_loop(dimensions: Dimensions, area: Area, max_iter: int = 20):
    """Returns an array containing the Mandelbrot fractal.

    This function does not use numpy to compute the fractal values.
    We still generate the nested lists using numpy since this is much simpler.
    """
    x = np.linspace(area.x_min, area.x_max, dimensions.width).reshape(1, -1)
    y = np.linspace(area.y_min, area.y_max, dimensions.height).reshape(-1, 1)

    c = x + y * 1j
    z = c.copy()
    nrows, ncols = z.shape

    diverge_start = np.full(z.shape, np.int64(max_iter), dtype=np.int64)

    for i_row in range(nrows):
        for i_col in range(ncols):
            for i in range(max_iter):
                z_val = z[i_row, i_col]
                new_val = z_val**2 + c[i_row, i_col]
                z[i_row, i_col] = new_val
                # Are we diverging?
                is_diverging = np.abs(z_val * np.conj(z_val)) > 4
                if is_diverging:
                    # Do we start diverging in this step?
                    if diverge_start[i_row, i_col] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[i_row, i_col] = i
                    # Avoid diverging too much
                    z[i_row, i_col] = 2
    return diverge_start


@njit(cache=True, parallel=True)
def mandelbrot_numba_loop_parallel(
    dimensions: Dimensions, area: Area, max_iter: int = 20
):
    """Returns an array containing the Mandelbrot fractal.

    This function does not use numpy to compute the fractal values.
    We still generate the nested lists using numpy since this is much simpler.
    """
    x = np.linspace(area.x_min, area.x_max, dimensions.width).reshape(1, -1)
    y = np.linspace(area.y_min, area.y_max, dimensions.height).reshape(-1, 1)

    c = x + y * 1j
    z = c.copy()
    nrows, ncols = z.shape

    diverge_start = np.full(z.shape, np.int64(max_iter), dtype=np.int64)

    for i_row in prange(nrows):
        for i_col in prange(ncols):
            for i in range(max_iter):
                val = z[i_row, i_col]
                new_val = val**2 + c[i_row, i_col]
                z[i_row, i_col] = new_val
                # Are we diverging?
                is_diverging = new_val * np.conj(new_val) > 4
                if is_diverging:
                    # Do we start diverging in this step?
                    if diverge_start[i_row, i_col] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[i_row, i_col] = i
                    # Avoid diverging too much
                    z[i_row, i_col] = 2
    return diverge_start


@njit(cache=True)
def mandelbrot_numba_broadcast(dimensions: Dimensions, area: Area, max_iter: int = 20):
    """Returns an array containing the Mandelbrot fractal."""
    x = np.linspace(area.x_min, area.x_max, dimensions.width).reshape(1, -1)
    y = np.linspace(area.y_min, area.y_max, dimensions.height).reshape(-1, 1)
    c = x + y * 1j
    z = c
    diverge_start = np.full(z.shape, np.int64(max_iter), dtype=np.int64)

    for i in range(max_iter):
        z = z**2 + c
        # Who is diverging?
        is_diverging = np.abs(z * np.conj(z)) > 4
        # Who starts diverging this step?
        div_now = is_diverging & (diverge_start == max_iter)
        # Note the divergence start for points starting this step
        diverge_start.reshape(-1)[div_now.reshape(-1)] = np.int64(i)
        # Avoid diverging too much
        z.reshape(-1)[is_diverging.reshape(-1)] = 2

    return diverge_start
