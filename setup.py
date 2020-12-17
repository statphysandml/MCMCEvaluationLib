#!/usr/bin/env python

# python setup.py sdist
# pip install -e .

# https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html


from distutils.core import setup

setup(name='mcmctools',
      version='1.0',
      description='Python modules for performing operations on mcmc simualtion data',
      author='Lukas Kades',
      author_email='lukaskades@googlemail.com',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages=['mcmctools',
                'mcmctools.modes',
                'mcmctools.pytorch',
                'mcmctools.loading',
                'mcmctools.utils']
     )
