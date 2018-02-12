# History of changes

## Current version

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
