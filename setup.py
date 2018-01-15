#!/usr/bin/env python

from distutils.core import setup

setup(name='Topology learning',
      version='1.0',
      description='Machine learning experiments for geospatial vector geometries',
      author='Rein van \'t Veer',
      author_email='rein.van.t.veer@geodan.nl',
      url='https://github.com/reinvantveer/Topology-Learning',
      packages=['model', 'model.topoml_util', 'model.baseline'],
      license='MIT',
      install_requires=[
          'sklearn',
          'slackclient',
          'scipy',
          'keras',
          'numpy',
          'shapely',
          'tensorflow-gpu'
      ],
      )
