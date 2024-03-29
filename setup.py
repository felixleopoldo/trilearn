#!/usr/scripts/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup, find_packages 

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

version = open('VERSION').read().strip()

packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

package_data = {
}

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.readlines()

classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
]

setup(
    name='trilearn',
    python_requires='>3.0',
    version=version,
    description='Bayesian predictive classification and structure learning in decomposable graphical models using particle Gibbs.',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=packages,
    package_data=package_data,
    install_requires=requirements,
    scripts=["scripts/count_chordal_graphs", 
             "scripts/gen_g-intraclass_precmat", 
             "scripts/analyze_graph_trajectories",
             "scripts/pgibbs_ggm_sample",
             "scripts/pgibbs_loglinear_sample",
             "scripts/pgibbs_uniform_jt_sample", 
             "scripts/sample_cta",
             "scripts/mh_ggm_sample",
             "scripts/sample_g-inv_wish", 
             "scripts/sample_ggm_AR_data", 
             "scripts/sample_ggm_intraclass_data", 
             "scripts/sample_loglinear_data", 
             "scripts/sample_loglinear_parameters", 
             "scripts/sample_normal_data", 
             "scripts/smc_ggm_analyze",
             "scripts/smc_ggm_sample"],
    author="Felix Rios",
    author_email='felix.leopoldo.rios@gmail.com',
    url='https://github.com/felixleopoldo/trilearn',
    download_url = 'https://github.com/felixleopoldo/trilearn/archive/'+version+
                   '.tar.gz',
    license='Apache 2.0',
    classifiers=classifiers,
)
