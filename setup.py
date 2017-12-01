import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pygfunction",
    version="1.0.0",
    packages=['pygfunction',
			  'pygfunction/examples'],
	include_package_data=True,
    install_requires=['matplotlib>=1.5.3',
	                  'numpy>=1.11.3',
					  'scipy>=1.0.0'],

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
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
		"Topic :: Scientific/Engineering",
		"Topic :: Utilities",
    ],
)
