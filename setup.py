from setuptools import setup
from Cython.Build import cythonize

setup(
    name="simplehod",
    version="0.0.1",
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    url="http://github.com/bccp/simplehod",
    description="Simple HOD modelling of galaxy from simulation catalogs",
    install_requires=['cython', 'numpy'],
    license='BSD-2-Clause',
    test_requires=['nose'],
    test_suite='nose.collector',
    ext_modules = cythonize("simplehod.pyx"),
)

