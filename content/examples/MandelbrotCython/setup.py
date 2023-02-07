from distutils.extension import Extension
from setuptools import find_packages, setup
from Cython.Build import build_ext
import numpy as np

extensions = [
    Extension(
        "mandelbrot_cython",
        ["src/mandelbrot_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="mandelbrot-cython",
    version="0.0.3",
    author="Dr. Matthias Hölzl",
    author_email="tc@xantira.com",
    description="Cython module for the Mandelbrot program",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hoelzl/Mandelbrot/",
    project_urls={
        "Bug Tracker": "https://github.com/hoelzl/Mandelbrot/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"],
    ext_modules=extensions,  # type: ignore
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=find_packages(),
    package_dir={"mandelbrot": "src/mandelbrot", "mandelbrot_cython": "src"},
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mandelbrot = mandelbrot.__main__:main",
        ],
    },
)
