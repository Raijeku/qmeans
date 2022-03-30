"""Module including package metadata"""

from setuptools import setup

with open("README.md", 'r', encoding="utf-8") as f:
    long_description = f.read()

setup(
   name='qmeans',
   version='1.0',
   description='Q-Means algorithm implementation using Qiskit compatible with Scikit-Learn.',
   license="Apache-2.0",
   long_description=long_description,
   author='David Quiroga',
   author_email='raijeku@gmail.com',
   url="http://www.notavailable.com/",
   packages=['qmeans'],
   install_requires=['wheel', 'numpy', 'pandas', 'qiskit', 'sklearn', 'pytest', 'hypothesis', 'sphinx', 'sphinxcontrib-napoleon'],
)
