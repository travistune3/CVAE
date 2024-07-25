# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:36:37 2024

@author: ttune3
"""

from setuptools import setup, find_packages

from distutils.core import setup as setup_

setup(name='CVAE',
      version='0.1',
      description='Conditional Variational Autoencoder in Pytorch',
      url='https://github.com/travistune3/CVAE',
      author='Travis C Tune',
      author_email='ttune3@uw.edu',
      packages=find_packages(),
      setup_requires = ['numpy'],
      install_requires=['ujson', 'matplotlib', 'numba', 'scipy'])