import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pygfunction",
    version="2.0.0",
    packages=['pygfunction',
              'pygfunction/examples'],
    include_package_data=True,
    install_requires=['coolprop>=6.4.1',
                      'matplotlib>=3.3.4',
                      'numpy>=1.19.2',
                      'scipy>=1.6.2'],
    test_suite='tests',

    # metadata for upload to PyPI
    author="Massimo Cimmino",
    author_email="massimo.cimmino@polymtl.ca",
    description="A g-function calculator for Python",
    long_description=read('README.md'),
    license="BSD-3-Clause",
    keywords="Geothermal, Ground-source, Ground heat exchangers",
    url="https://github.com/MassimoCimmino/pygfunction",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
    ],
)
