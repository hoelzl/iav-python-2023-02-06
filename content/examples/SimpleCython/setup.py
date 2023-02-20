from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import build_ext

extensions = [Extension("cython_hello", ["hello.pyx"])]

setup(
    name="cython-test",
    setup_requires=["cython>=0.29"],
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
