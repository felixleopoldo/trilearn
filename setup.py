#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup, find_packages

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
print("packages: {}".format(packages))

package_data = {
}

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.readlines()

classifiers = [
        'Development Status :: 1 - Beta',
        'Intended Audience :: Researchers / Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
]

setup(
    name='trilearn',
    version=0.3,
    description='Learning decomposable graphical models using particle Gibbs.',
    long_description=readme,
    packages=packages,
    package_data=package_data,
    install_requires=requirements,
    author="Felix Rios",
    author_email='felix.leopoldo.rios@gmail.com',
    url='https://github.com/felixleopoldo/trilearn',
    license='Apache 2.0',
    classifiers=classifiers,
)
