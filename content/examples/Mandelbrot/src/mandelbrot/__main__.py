import argparse
from datetime import datetime
from logging import warning
from .types import Area, Dimensions
from .pure_python import (
    mandelbrot_pure_python,
    mandelbrot_numpy,
    plot_mandelbrot,
)
from .numba_impl import (
    mandelbrot_numba_broadcast,
    mandelbrot_numba_loop,
    mandelbrot_numba_loop_parallel,
)
from mandelbrot_cython import (
    mandelbrot_cython_loop,
    mandelbrot_cython_loop_numpy,
    mandelbrot_cython_loop_numpy_typed,
    mandelbrot_cython_numpy,
    mandelbrot_cython_numpy_typed,
    mandelbrot_cython_native,
    mandelbrot_cython_native_parallel,
    mandelbrot_cython_loop_numpy_parallel,
)

fun_map = {
    "python": mandelbrot_pure_python,
    "numpy": mandelbrot_numpy,
    "cython": mandelbrot_cython_loop,
    "cython_numpy": mandelbrot_cython_numpy,
    "cython_numpy_typed": mandelbrot_cython_numpy_typed,
    "cython_loop_numpy": mandelbrot_cython_loop_numpy,
    "cython_loop_numpy_typed": mandelbrot_cython_loop_numpy_typed,
    "cython_native": mandelbrot_cython_native,
    "cython_native_par": mandelbrot_cython_native_parallel,
    "cython_loop_numpy_par": mandelbrot_cython_loop_numpy_parallel,
    "numba": mandelbrot_numba_broadcast,
    "numba_loop": mandelbrot_numba_loop,
    "numba_loop_par": mandelbrot_numba_loop_parallel,
}

warmup_funs = {"numba_loop", "numba_loop_par", "numba"}


def main():
    parser = argparse.ArgumentParser(
        prog="mandelbrot",
        description="An implementation of Mandelbrot sets.",
        epilog="Have fun!",
        add_help=False,
    )
    parser.add_argument("--help", action="help", help="Show the help message")
    parser.add_argument(
        "--height", "-h", default=640, type=int, help="The height of the image"
    )
    parser.add_argument(
        "--width", "-w", default=800, type=int, help="The width of the image"
    )
    parser.add_argument("--x-min", default=-2.2, type=float, help="The minimal x value")
    parser.add_argument("--x-max", default=1.2, type=float, help="The maximal x value")
    parser.add_argument("--y-min", default=-1.4, type=float, help="The minimal y value")
    parser.add_argument("--y-max", default=1.4, type=float, help="The maximal y value")
    parser.add_argument(
        "--max-iterations", "-i", default=20, type=int, help="The number of iterations"
    )
    parser.add_argument(
        "--color-map",
        "-c",
        default="Spectral",
        help="The matplotlib colormap to use (try cividis, twilight_shifted, or turbo)",
    )
    parser.add_argument(
        "--max-value",
        "-m",
        default=20,
        type=int,
        help="The max value for the divergence",
    )
    parser.add_argument(
        "--fun",
        "-f",
        default="numpy",
        help="Which Mandelbrot implementation to use?",
    )
    parser.add_argument(
        "--force-display",
        default=False,
        type=bool,
        help="Force the display of result images when all test are run",
    )
    args = parser.parse_args()

    if args.fun == "all":
        for fun_name, fun in fun_map.items():
            print(f"Running {fun_name}\n  ", end="")
            run_fun(fun_name, fun, args, display=args.force_display)
    elif fun := fun_map.get(args.fun):
        run_fun(args.fun, fun, args)
    else:
        msg = (
            f"No function named {args.fun}. Valid choices are:\n  "
            f"{', '.join(fun_map.keys())}"
        )
        warning(msg)


def run_fun(fun_name, fun, args, display=True):
    if fun_name in warmup_funs:
        fun(
            Dimensions(height=args.height, width=args.width),
            Area(args.x_min, args.x_max, args.y_min, args.y_max),
            max_iter=args.max_iterations,
        )

    start_time = datetime.now()
    mandel = fun(
        Dimensions(height=args.height, width=args.width),
        Area(args.x_min, args.x_max, args.y_min, args.y_max),
        max_iter=args.max_iterations,
    )
    time_delta = datetime.now() - start_time
    print(f"Computation took {time_delta.total_seconds():.2f}s")
    if display:
        plot_mandelbrot(
            mandel,
            cmap=args.color_map,
            max_value=args.max_value,
        )


if __name__ == "__main__":
    main()
