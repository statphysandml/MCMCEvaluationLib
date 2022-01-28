#!/usr/bin/env python

from distutils.core import setup

setup(name='mcmctools',
      version='0.1',
      description='Python modules for performing operations on mcmc simualtion data.',
      author='Lukas Kades',
      author_email='lukaskades@googlemail.com',
      url='https://github.com/statphysandml/MCMCEvaluationLib',
      packages=['mcmctools',
                'mcmctools.modes',
                'mcmctools.pytorch',
                'mcmctools.loading',
                'mcmctools.mcmc',
                'mcmctools.utils']
     )
