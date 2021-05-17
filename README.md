# pygfunction: A g-function calculator for Python

[![Build Status](https://travis-ci.org/MassimoCimmino/pygfunction.svg?branch=master)](https://travis-ci.org/MassimoCimmino/pygfunction) [![DOI](https://zenodo.org/badge/100305705.svg)](https://zenodo.org/badge/latestdoi/100305705)


## What is *pygfunction*?

*pygfunction* is a Python module for the calculation of thermal response
factors, or *g*-functions, for fields of geothermal boreholes. *g*-functions
form the basis of many simulation and sizing programs for geothermal heat pump
systems. *g*-Functions are superimposed in time to predict fluid and ground
temperatures in these systems.

At its core, *pygfunction* relies on the analytical finite line source solution
to evaluate the thermal interference between boreholes in the same bore field.
This allows for the very fast calculation of *g*-functions, even for very large
bore fields with hundreds of boreholes.

Using *pygfunction*, *g*-functions can be calculated for any bore field
configuration (i.e. arbitrarily positionned in space), including fields of
boreholes with individually different lengths and radiuses. For regular fields
of boreholes of equal size, setting-up the calculation of the *g*-function is
as simple as a few lines of code. For example, the code for the calculation of
the *g*-function of a 10 x 10 square array of boreholes (100 boreholes
total):

```python
import pygfunction as gt
import numpy as np
time = np.array([(i+1)*3600. for i in range(24)]) # Calculate hourly for one day
boreField = gt.boreholes.rectangle_field(N_1=10, N_2=10, B_1=7.5, B_2=7.5, H=150., D=4., r_b=0.075)
gFunc = gt.gfunction.gFunction(boreField, alpha=1.0e-6, time=time)
gFunc.visualize_g_function()
```

Once the *g*-function is evaluated, *pygfunction* provides tools to predict
borehole temperature variations (using load aggregation methods) and to evaluate
fluid temperatures in the boreholes for several U-tube pipe configurations.


## Requirements

*pygfunction* was developed and tested using Python 2.7 and supports Python 3.6. In addition, the
following packages are needed to run *pygfunction* and its examples:
- matplotlib (>= 1.5.3), required for the examples
- numpy (>= 1.11.3)
- scipy (>= 1.0.0)

The documentation is generated using [Sphinx](http://www.sphinx-doc.org). The
following packages are needed to build the documentation:
- sphinx (>= 1.5.1)
- numpydoc (>= 0.6.0)


## Quick start

**Users** - [Download pip](https://pip.pypa.io/en/latest/) and install the latest release:

```
pip install pygfunction
```

Alternatively, [download the latest release](https://github.com/MassimoCimmino/pygfunction/releases) and run the installation script:

```
python setup.py install
```

**Developers** - To get the latest version of the code, you can [download the
repository from github](https://github.com/MassimoCimmino/pygfunction) or clone
the project in a local directory using git:

```
git clone https://github.com/MassimoCimmino/pygfunction.git
```

Once *pygfunction* is copied to a local directory, you can verify that it is
working properly by running the examples in `pygfunction/examples/`.


## Documentation

*pygfunction*'s documentation is hosted on
[ReadTheDocs](https://pygfunction.readthedocs.io).


## Contributing to *pygfunction*

You can report bugs and propose enhancements on the
[issue tracker](https://github.com/MassimoCimmino/pygfunction/issues).

To contribute code to *pygfunction*, follow the
[contribution workflow](CONTRIBUTING.md).


## License

*pygfunction* is licensed under the terms of the 3-clause BSD-license.
See [pygfunction license](LICENSE.md).
