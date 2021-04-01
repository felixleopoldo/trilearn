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
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
]

setup(
    name='trilearn',
    python_requires='<3.0',
    version=1.21,
    description='Bayesian predictive classification and structure learning in decomposable graphical models using particle Gibbs.',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=packages,
    package_data=package_data,
    install_requires=requirements,
    scripts=["bin/count_chordal_graphs", 
             "bin/gen_g-intraclass_precmat", 
             "bin/analyze_graph_trajectories",
             "bin/pgibbs_ggm_sample",
             "bin/pgibbs_loglinear_sample",
             "bin/pgibbs_uniform_jt_sample", 
             "bin/sample_cta",
             "bin/mh_ggm_sample",
             "bin/sample_g-inv_wish", 
             "bin/sample_ggm_AR_data", 
             "bin/sample_ggm_intraclass_data", 
             "bin/sample_loglinear_data", 
             "bin/sample_loglinear_parameters", 
             "bin/sample_normal_data", 
             "bin/smc_ggm_analyze",
             "bin/smc_ggm_sample"],
    author="Felix Rios",
    author_email='felix.leopoldo.rios@gmail.com',
    url='https://github.com/felixleopoldo/trilearn',
    download_url = 'https://github.com/felixleopoldo/trilearn/archive/1.21'
                   '.tar.gz',
    license='Apache 2.0',
    classifiers=classifiers,
)
