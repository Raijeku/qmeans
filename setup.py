"""Module including package metadata"""

from setuptools import setup

with open("README.md", 'r', encoding="utf-8") as f:
    long_description = f.read()

setup(
   name='qmeans',
   version='0.1.3',
   description='Q-Means algorithm implementation using Qiskit compatible with Scikit-Learn.',
   license="Apache-2.0",
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='David A. Quiroga',
   author_email='raijeku@gmail.com',
   url="http://qmeans.readthedocs.io/",
   packages=['qmeans'],
   install_requires=['wheel', 'twine', 'setuptools', 'numpy==1.26.4', 'pandas==2.2.2', 'qiskit==1.0.2', 
   'qiskit_aer==0.14.0.1', 'qiskit_ibm_runtime==0.23.0', 'scikit-learn==1.4.2', 'pytest==8.2.0', 'sphinx==7.3.7', 
   'sphinx-rtd-theme==2.0.0', 'sphinxcontrib-napoleon==0.7'],
)
