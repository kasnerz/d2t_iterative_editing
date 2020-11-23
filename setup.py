#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='d2t_iterative_editing',
    version='0.0.0',
    description='Data-to-Text Generation with Iterative Text Editing',
    author='Zdeněk Kasner',
    author_email='kasner@ufal.mff.cuni.cz',
    url='https://github.com/kasnerz/d2t_iterative_editing',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

