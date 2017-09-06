# pygfunction: A g-function calculator for Python


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
the *g*-function of a field 10 x 10 square array of boreholes (100 boreholes
total):

```python
time = [(i+1)*3600. for i in range(24)] # Calculate hourly for one day
boreField = gt.boreholes.rectangle_field(N_1=10, N_2=10, B_1=7.5, B_2=7.5, H=150., D=4., r_b=0.075)
gFunc = gt.gfunction.uniform_temperature(boreField, time, alpha=1.0e-6)
```


## Requirements

*pygfunction* was developed and tested using Python 2.7. In addition, the
following packages are needed to run *pygfunction* and its examples:
- matplotlib (>= 1.5.3), required for the examples
- numpy (>= 1.11.1)
- scipy (>= 0.18.1)

The documentation is generated using [Sphinx](http://www.sphinx-doc.org). The
following packages are needed to build the documentation:
- sphinx (>= 1.5.1)
- sphinx-bootstrap-theme (>= 0.4.14)
- numpydoc (>= 0.6.0)


## Quick start

To get the latest version of the code, you can [download the repository from
github](https://github.com/MassimoCimmino/pygfunction) or clone the project
in a local directory using git:

```
git clone https://github.com/MassimoCimmino/pygfunction.git
```

Once *pygfunction* is copied to a local directory, you can verify that it is
working properly by running the examples in `pygfunction/examples/`.


## Contributing to *pygfunction*

You can report bugs and propose enhancements on the
[issue tracker](https://github.com/MassimoCimmino/pygfunction/issues).

To contribute code to *pygfunction*, follow the
[contribution workflow](CONTRIBUTING.md).


## License

*pygfunction* is licensed under the terms of the 3-clause BSD-license.
See [pygfunction license](LICENSE.md).
