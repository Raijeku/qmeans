from setuptools import setup

with open("README", 'r') as f:
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
   install_requires=['numpy', 'pandas', 'qiskit', 'sklearn'],
)