# History of changes

## Current version

### New features

* [Issue 33](https://github.com/MassimoCimmino/pygfunction/issues/33), [Issue 54](https://github.com/MassimoCimmino/pygfunction/issues/54), [Issue 85](https://github.com/MassimoCimmino/pygfunction/issues/85) - New class `gFunction` for the calculation of g-functions. The new class is a common interface to all boundary conditions and calculation methods. The new implementation of the solver reduces the memory requirements of pygfunction. The new class implements visualization features for the g-function and for heat extraction rates and borehole wall temperatures (both as a function of time and for the profiles along the borehole lengths).
* [Issue 75](https://github.com/MassimoCimmino/pygfunction/issues/75) - New module `media` with properties of brine mixtures.
* [Issue 81](https://github.com/MassimoCimmino/pygfunction/issues/81) - Added functions to find a remove duplicate boreholes.

### Enhancements

* [Issue 78](https://github.com/MassimoCimmino/pygfunction/issues/78) - Optimization of solvers for the calculation of g-functions. The finite line source (FLS) solution is now calculated using `scipy.integrate.quad_vec` which significantly improves calculation time over `scipy.integrate.quad`. The identification of similarities in the 'similarities' solver has also been refactored to identify similarities between boreholes as an intermediate step before identifying similarities between segments. The calculation time for the identification of similarities is significantly decreased.

### Bug fixes

* [Issue 86](https://github.com/MassimoCimmino/pygfunction/issues/86) - Documentation is now built using Python 3 to support Python 3 features in the code.
* [Issue 103](https://github.com/MassimoCimmino/pygfunction/issues/103) - Fixed `gFunction` class to allow both builtin `float` and `numpy.floating` inputs.

### Other changes

* [Issue 72](https://github.com/MassimoCimmino/pygfunction/issues/72) - Added a list of contributors to the front page. The list is managed using [all-contributors](https://github.com/all-contributors/all-contributors).
* [Issue 87](https://github.com/MassimoCimmino/pygfunction/issues/87) - Drop support for Python 2. All package requirements are updated to the latest conda version.

## Version 1.1.2 (2021-01-21)

### New features

* [Issue 47](https://github.com/MassimoCimmino/pygfunction/issues/47) - Added verification of the validity of pipe geometry to pipe classes. Extended visualization of the borehole cross-section.
* [Issue 66](https://github.com/MassimoCimmino/pygfunction/issues/66) - Added a class method to the Claesson & Javed load aggregation method to retrieve the thermal response factor increment.

### Enhancements

* [Issue 59](https://github.com/MassimoCimmino/pygfunction/issues/59) - Use a relative tolerance instead of an absolute tolerance in the identification of borehole pair similarities. This provides faster execution times and similar accuracy.

### Bug fixes

* [Issue 58](https://github.com/MassimoCimmino/pygfunction/issues/58) - Store matrix coefficients in `Network` class methods for re-use when inlet conditions are constant.
* [Issue 64](https://github.com/MassimoCimmino/pygfunction/issues/64) - Fixed an issue where the g-function was returned as an array of integers if time values were integers.

## Version 1.1.1 (2020-06-20)

### New features

* [Issue 40](https://github.com/MassimoCimmino/pygfunction/issues/40) - Added Network class for simulations involving networks of boreholes.

### Bug fixes

* [Commit a4f6591](https://github.com/MassimoCimmino/pygfunction/commit/a4f6591384295c9918cb13b60f07c0afa500e700) - Fixed import of Axes3D necessary in `borehole.visualize_field()`.

## Version 1.1.0 (2018-03-09)

### New features

* [Commit 2bd12bd](https://github.com/MassimoCimmino/pygfunction/commit/2bd12bd254928889431366c2ddd38539e246ef05) - Implemented `UTube.visualize_pipes()` class method.
* [Issue 30](https://github.com/MassimoCimmino/pygfunction/issues/30) - Laminar regime is now considered for calculation of convection heat transfer coefficient.
* [Issue 32](https://github.com/MassimoCimmino/pygfunction/issues/32) - g-Functions for bore fields with mixed series and parallel connections between boreholes.

### Bug fixes

* [Commit 2523f67](https://github.com/MassimoCimmino/pygfunction/commit/2523f67e7a932538c1135bba52e0d4035f866e3e) - `boreholes.visualize_field()` now returns the figure object.
* [Issue 25](https://github.com/MassimoCimmino/pygfunction/issues/25) - Fixed documentation of ./examples/uniform_temperature.py.
* [Issue 27](https://github.com/MassimoCimmino/pygfunction/issues/27) - `thermal_response_factors()` is now part of the `heat_transfer` module (moved from `gfunction`).
* [Commit 1f59872](https://github.com/MassimoCimmino/pygfunction/commit/1f59872747190353d8eb937c021f1e6107b60ab8) - Fixed incorrect summation limit in pipes._F_mk.

## Version 1.0.0 (2017-12-01)

### New features

* [Issue 4](https://github.com/MassimoCimmino/pygfunction/issues/4) - Unit testing and integration with [Travis CI](https://travis-ci.org/MassimoCimmino/pygfunction/).
* [Issue 16](https://github.com/MassimoCimmino/pygfunction/issues/16) - Added capability to import bore field from external text files.
* [Issue 18](https://github.com/MassimoCimmino/pygfunction/issues/18) - Added capability to visualize bore fields.
* [Issue 20](https://github.com/MassimoCimmino/pygfunction/issues/20) - Added utilities module.
* [Issue 5](https://github.com/MassimoCimmino/pygfunction/issues/5) - Added setup.py installation script using setuptools. *pygfunction* will now be available on [pypi](https://pypi.python.org/pypi/pygfunction).

### Bug fixes

* [Commit 0b2d364](https://github.com/MassimoCimmino/pygfunction/commit/0b2d3645e480be9892533dcbd9df80412ca7210f) - `boreholes.U_shaped_field()` and `boreholes.U_shaped_field()` did not construct the field properly when called with low form factors
* [Commit 62acd42](https://github.com/MassimoCimmino/pygfunction/commit/62acd428187854c71c9e37fe2dc479bbbb9c0eb4) - Criteria for smooth pipe correlation in calculation of Darcy friction factor was too large. Colebrooke-White equation is now always used for turbulent flow.
* [Issue 14](https://github.com/MassimoCimmino/pygfunction/issues/14) - Evaluate g-functions at a single time value.
* [Commit 7b408e9](https://github.com/MassimoCimmino/pygfunction/commit/7b408e9392aff96832c315ac3e436e306a9b471c) - Fix interpolation of thermal response factors in cases where the minimum timestep does not correspond to the first timestep. This also fixes errors caused by rounding errors in the interpolation.
* [Issue 22](https://github.com/MassimoCimmino/pygfunction/issues/22) - Allow 1d arrays for g-function values in the initilization of load aggregation algorithms.

## Version 0.3.0 (2017-10-17)

### New features

* [Issue 6](https://github.com/MassimoCimmino/pygfunction/issues/6) - Store coefficients in pipe models for faster computation
* [Issue 7](https://github.com/MassimoCimmino/pygfunction/issues/7) - Multipole method to evaluate borehole internal resistances
