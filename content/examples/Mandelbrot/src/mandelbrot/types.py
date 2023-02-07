try:
    from mandelbrot_cython import Area, Dimensions  # noqa
except ModuleNotFoundError:
    from collections import namedtuple

    Dimensions = namedtuple("Dimensions", ["height", "width"])
    Area = namedtuple("Area", ["x_min", "x_max", "y_min", "y_max"])
