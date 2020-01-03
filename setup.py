from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
        Extension("simplehod.simplehod", ["simplehod/simplehod.pyx"])
]

setup(
    name="simplehod",
    version="0.0.1",
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    url="http://github.com/bccp/simplehod",
    description="Simple HOD modelling of galaxy from simulation catalogs",
    install_requires=['cython', 'numpy'],
    license='BSD-2-Clause',
    zip_safe = False,
    package_dir = {'simplehod': 'simplehod'},
    packages=['simplehod'],
    ext_modules = cythonize(extensions),
)

