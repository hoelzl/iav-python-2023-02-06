# cython: language_level=3

from collections import namedtuple
from copy import deepcopy

import numpy as np
cimport numpy as np
import cython
cimport cython
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
# from libc.complex cimport conj

cdef extern from "complex.h":
    double complex conj(double complex z) nogil;


Dimensions = namedtuple("Dimensions", ["height", "width"])
Area = namedtuple("Area", ["x_min", "x_max", "y_min", "y_max"])


def mandelbrot_cython_loop(dimensions: Dimensions, area: Area, max_iter: int = 20):
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



def mandelbrot_cython_numpy(dimensions: Dimensions, area: Area, max_iter: int = 20):
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


# TODO: There doesn't seem to be much that we can do with types without going to the
#       loop version?
#       Cython seems to reject numpy broadcasting operations when we declare variables
#       as typed memoryviews.
#
def mandelbrot_cython_numpy_typed(dimensions: Dimensions, area: Area, max_iter: int = 20):
    """Returns an array containing the Mandelbrot fractal."""
    cdef Py_ssize_t ncols = dimensions.width
    cdef Py_ssize_t nrows = dimensions.height

    cdef np.ndarray[double, ndim=2] x = np.linspace(area.x_min, area.x_max, ncols).reshape(1, -1)
    cdef np.ndarray[double, ndim=2] y = np.linspace(area.y_min, area.y_max, nrows).reshape(-1, 1)
    cdef np.ndarray[complex, ndim=2] c = x + y * 1j
    cdef np.ndarray[complex, ndim=2] z = c
    cdef np.ndarray[long, ndim=2] diverge_start = np.full((nrows, ncols), max_iter, dtype=np.int64)

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

def mandelbrot_cython_loop_numpy(dimensions: Dimensions, area: Area, max_iter: int = 20):
    """Returns an array containing the Mandelbrot fractal."""
    x = np.linspace(area.x_min, area.x_max, dimensions.width).reshape(1, -1)
    y = np.linspace(area.y_min, area.y_max, dimensions.height).reshape(-1, 1)

    c = x + y * 1j
    z = c.copy()
    z_x, z_y = z.shape

    diverge_start = np.full(z.shape, np.int64(max_iter), dtype=np.int64)

    for i_y in range(z_y):
        for i_x in range(z_x):
            for i in range(max_iter):
                z_val = z[i_x, i_y]
                new_val = z_val ** 2 + c[i_x, i_y]
                z[i_x, i_y] = new_val
                # Are we diverging?
                if np.abs(z_val * np.conj(z_val)) > 4:
                    # Do we start diverging in this step?
                    if diverge_start[i_x, i_y] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[i_x, i_y] = i
                    # Avoid diverging too much
                    z[i_x, i_y] = 2
    return diverge_start

cpdef mandelbrot_cython_loop_numpy_typed(
        dimensions: Dimensions = Dimensions(10, 12),
        area: Area = Area(-2.2, 1.2, -1.4, 1.4),
        int max_iter = 20):
    """Returns an array containing the Mandelbrot fractal."""
    cdef Py_ssize_t ncols = dimensions.width
    cdef Py_ssize_t nrows = dimensions.height
    cdef double[::1] xs = np.linspace(area.x_min, area.x_max, ncols)
    cdef double[::1] ys = np.linspace(area.y_min, area.y_max, nrows)

    cdef double complex[:, ::1] c = np.empty((nrows, ncols), dtype=complex)
    cdef double complex[:, ::1] z = np.empty((nrows, ncols), dtype=complex)
    icol1 = cython.declare(Py_ssize_t)
    irow1 = cython.declare(Py_ssize_t)
    val = cython.declare(complex)
    for icol1 in range(ncols):
        for irow1 in range(nrows):
            val = xs[icol1] + ys[irow1] * 1j
            c[irow1, icol1] = val
            z[irow1, icol1] = val

    cdef long[:, :] diverge_start = np.full((nrows, ncols), np.int64(max_iter), dtype=np.int64)


    icol2 = cython.declare(Py_ssize_t)
    irow2 = cython.declare(Py_ssize_t)
    i = cython.declare(Py_ssize_t)
    cdef double abs_val

    for icol2 in range(ncols):
        for irow2 in range(nrows):
            for i in range(max_iter):
                z_val = cython.declare(cython.complex)
                new_val = cython.declare(cython.complex)

                z_val = z[irow2, icol2]
                new_val = z_val ** 2 + c[irow2, icol2]
                z[irow2, icol2] = new_val
                # Are we diverging?
                abs_val = z_val.real ** 2 + z_val.imag ** 2
                if abs_val > 4.0:
                    # Do we start diverging in this step?
                    if diverge_start[irow2, icol2] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[irow2, icol2] = i
                    # Avoid diverging too much
                    z[irow2, icol2] = 2
    return diverge_start


cdef double* linspace_double(double start, double stop, int num):
    cdef double step = (stop - start) / (num - 1) if num > 1 else 0
    cdef double *result = <double*> malloc(num * sizeof(double))
    for i in range(num):
        result[i] = start + i * step
    return result

def test_linspace_double(start, stop, num):
    cdef double* buffer = linspace_double(start, stop, num)
    result = [x for x in buffer[:num]]
    free(<void*> buffer)
    return result

cdef linspace_iter(start, stop, num):
    step = (stop - start) / (num - 1) if num > 1 else 0
    return (start + i * step for i in range(num))

def test_linspace_iter(start, stop, num):
    return list(linspace_iter(start, stop, num))

@cython.inline
cdef Py_ssize_t linear_index(Py_ssize_t row, Py_ssize_t col, Py_ssize_t num_cols) nogil:
    return col + num_cols * row

def test_linear_index(row, col, num_cols):
    return linear_index(row, col, num_cols)

@cython.inline
cdef int* allocate_linear_buffer_int(Py_ssize_t size) nogil:
    return <int*> malloc(size * sizeof(int))

@cython.inline
cdef double complex* allocate_linear_buffer_complex(Py_ssize_t size) nogil:
    return <double complex*> malloc(size * sizeof(double complex))

cdef copy_to_new_numpy_array(
        int *buffer, Py_ssize_t nrows, Py_ssize_t ncols, dtype):
    result = np.empty((nrows, ncols), dtype=dtype)
    cdef int[:, :] tmp = result
    for col in range(ncols):
        for row in range(nrows):
            tmp[row, col] = buffer[linear_index(row, col, ncols)]
    return result

cdef init_cz(
        double complex* c, double complex* z,
        area: Area,
        Py_ssize_t nrows, Py_ssize_t ncols):
    for col, x in enumerate(linspace_iter(area.x_min, area.x_max, ncols)):
        for row, y in enumerate(linspace_iter(area.y_min, area.y_max, nrows)):
            val = x + y * 1j
            c[linear_index(row, col, ncols)] = val
            z[linear_index(row, col, ncols)] = val


cpdef mandelbrot_cython_native(
        dimensions: Dimensions = Dimensions(10, 12),
        area: Area = Area(-2.2, 1.2, -1.4, 1.4),
        int max_iter = 20):
    """Returns an array containing the Mandelbrot fractal."""
    cdef Py_ssize_t ncols = dimensions.width
    cdef Py_ssize_t nrows = dimensions.height
    cdef Py_ssize_t size = ncols * nrows

    cdef double complex* c = allocate_linear_buffer_complex(size)
    cdef double complex* z = allocate_linear_buffer_complex(size)

    init_cz(c, z, area, nrows, ncols)

    cdef int* diverge_start = allocate_linear_buffer_int(size)
    for i in range(size):
        diverge_start[i] = max_iter

    cdef double complex z_val = 0.0
    cdef double complex new_val = 0.0
    cdef double abs_val = 0.0
    for icol in range(ncols):
        for irow in range(nrows):
            for i in range(max_iter):
                z_val = z[linear_index(irow, icol, ncols)]
                new_val = z_val ** 2 + c[linear_index(irow, icol, ncols)]
                z[linear_index(irow, icol, ncols)] = new_val
                # Are we diverging?
                abs_val = z_val.real ** 2 + z_val.imag ** 2
                if abs_val > 4.0:
                    # Do we start diverging in this step?
                    if diverge_start[linear_index(irow, icol, ncols)] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[linear_index(irow, icol, ncols)] = i
                    # Avoid diverging too much
                    z[linear_index(irow, icol, ncols)] = 2.0
    free(<void*> c)
    free(<void*> z)
    result = copy_to_new_numpy_array(diverge_start, nrows, ncols, np.int32)
    free(<void*> diverge_start)
    return result


cpdef mandelbrot_cython_native_parallel(
        dimensions: Dimensions = Dimensions(10, 12),
        area: Area = Area(-2.2, 1.2, -1.4, 1.4),
        int max_iter = 20):
    """Returns an array containing the Mandelbrot fractal."""
    cdef Py_ssize_t ncols = dimensions.width
    cdef Py_ssize_t nrows = dimensions.height
    cdef Py_ssize_t size = ncols * nrows

    cdef double complex* c = allocate_linear_buffer_complex(size)
    cdef double complex* z = allocate_linear_buffer_complex(size)

    init_cz(c, z, area, nrows, ncols)

    cdef int* diverge_start = allocate_linear_buffer_int(size)
    for i1 in range(size):
        diverge_start[i1] = max_iter

    cdef double complex z_val = 0.0
    cdef double complex new_val = 0.0
    cdef double abs_val = 0.0
    icol = cython.declare(Py_ssize_t)
    irow = cython.declare(Py_ssize_t)
    i = cython.declare(Py_ssize_t)
    for icol in prange(ncols, nogil=True):
        for irow in prange(nrows):
            for i in range(max_iter):
                z_val = z[linear_index(irow, icol, ncols)]
                new_val = z_val ** 2 + c[linear_index(irow, icol, ncols)]
                z[linear_index(irow, icol, ncols)] = new_val
                # Are we diverging?
                abs_val = z_val.real ** 2 + z_val.imag ** 2
                if abs_val > 4.0:
                    # Do we start diverging in this step?
                    if diverge_start[linear_index(irow, icol, ncols)] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[linear_index(irow, icol, ncols)] = i
                    # Avoid diverging too much
                    z[linear_index(irow, icol, ncols)] = 2.0
    free(<void*> c)
    free(<void*> z)
    result = copy_to_new_numpy_array(diverge_start, nrows, ncols, np.int32)
    free(<void*> diverge_start)
    return result

@cython.boundscheck(False)
cpdef mandelbrot_cython_loop_numpy_parallel(
        dimensions: Dimensions = Dimensions(10, 12),
        area: Area = Area(-2.2, 1.2, -1.4, 1.4),
        int max_iter = 20):
    """Returns an array containing the Mandelbrot fractal."""
    cdef Py_ssize_t ncols = dimensions.width
    cdef Py_ssize_t nrows = dimensions.height
    cdef double[::1] xs = np.linspace(area.x_min, area.x_max, ncols)
    cdef double[::1] ys = np.linspace(area.y_min, area.y_max, nrows)

    cdef double complex[:, ::1] c = np.empty((nrows, ncols), dtype=complex)
    cdef double complex[:, ::1] z = np.empty((nrows, ncols), dtype=complex)
    icol1 = cython.declare(Py_ssize_t)
    irow1 = cython.declare(Py_ssize_t)
    val = cython.declare(complex)
    for icol1 in range(ncols):
        for irow1 in range(nrows):
            val = xs[icol1] + ys[irow1] * 1j
            c[irow1, icol1] = val
            z[irow1, icol1] = val

    cdef long[:, :] diverge_start = np.full((nrows, ncols), np.int64(max_iter), dtype=np.int64)


    icol2 = cython.declare(Py_ssize_t)
    irow2 = cython.declare(Py_ssize_t)
    i = cython.declare(Py_ssize_t)
    cdef double abs_val

    for icol2 in prange(ncols, nogil=True):
        for irow2 in prange(nrows):
            for i in range(max_iter):
                z_val = cython.declare(cython.complex)
                new_val = cython.declare(cython.complex)

                z_val = z[irow2, icol2]
                new_val = z_val ** 2 + c[irow2, icol2]
                z[irow2, icol2] = new_val
                # Are we diverging?
                abs_val = z_val.real ** 2 + z_val.imag ** 2
                if abs_val > 4.0:
                    # Do we start diverging in this step?
                    if diverge_start[irow2, icol2] == max_iter:
                        # Note the divergence start for points starting this step
                        diverge_start[irow2, icol2] = i
                    # Avoid diverging too much
                    z[irow2, icol2] = 2
    return diverge_start
