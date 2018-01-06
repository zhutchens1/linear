from setuptools import setup
from os import path

HERE = path.abspath(path.dirname(__file__))

setup(name='linear',
      version='1.0',
      description="A python package for vectors and matrices",
      url="https://github.com/zhutchens1/linear",
      author="Zackary L. Hutchens",
      license='MIT',
      packages=['linear']
      zip_safe=False)
