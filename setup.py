from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="pygfunction",
    version="1.0.0.a0.dev",
    packages=['pygfunction'],
    scripts=['examples/comparison_load_aggregation.py',
	         'examples/custom_bore_field.py',
	         'examples/custom_bore_field_from_file.py',
	         'examples/equal_inlet_temperature.py',
	         'examples/fluid_temperature.py',
	         'examples/load_aggregation.py',
	         'examples/multiple_independent_Utubes.py',
	         'examples/multipole_temperature.py',
	         'examples/regular_bore_field.py',
	         'examples/uniform_heat_extraction_rate.py',
	         'examples/uniform_temperature.py',
	         ],

    install_requires=['matplotlib>=1.5.3',
	                  'numpy>=1.11.3',
					  'scipy>=1.0.0'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
    },

    # metadata for upload to PyPI
    author="Massimo Cimmino",
    author_email="massimo.cimmino@polymtl.ca",
    description="A g-function calculator for Python",
    license="BSD-3-Clause",
    keywords="Geothermal, Ground-source, Ground heat exchangers",
    url="https://github.com/MassimoCimmino/pygfunction",
)
