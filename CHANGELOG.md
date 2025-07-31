# History of changes

## Version 2.3.1 (2025-08-04)

### New features

* [Pull Request 325](https://github.com/MassimoCimmino/pygfunction/pull/325) - Borefields and boreholes can now be concatenated using the `+` operator, e.g. using `new_field = field_1 + field_2`.
* [Pull Request 326](https://github.com/MassimoCimmino/pygfunction/pull/326) - Introduced `gFunction.from_static_params` and `Network.from_static_params` methods. These methods facilitate the creation of `Network` objects and the evaluation of g-functions by automatically evaluating the required thermal resistances for the creation of `Pipe` objects.

### Other changes

* [Issue 319](https://github.com/MassimoCimmino/pygfunction/issues/319) - Created `solvers` module. `Solver` classes are moved out of the `gfunction` module and into the new module.

## Version 2.3 (2025-04-29)

### New features

* [Issue 215](https://github.com/MassimoCimmino/pygfunction/issues/215) - Implemented variable fluid mass flow rate g-functions. Bore fields with series-connected boreholes and reversible flow direction can now be simulated.
* [Issue 282](https://github.com/MassimoCimmino/pygfunction/issues/282) - Enabled the use of negative mass flow rates in `Pipe` and `Network` classes to model reversed flow direction.
* [Pull Request 308](https://github.com/MassimoCimmino/pygfunction/pull/308) - Introduced a new `borefield` module. The new `Borefield` class replaces lists of `Borehole` objects as the preferred way to configure bore fields. The `Borefield.evaluate_g_function` method evaluates g-functions using the 'UHTR' and 'UBWT' boundary conditions. Deprecated bore field creation functions in the `boreholes` module (e.g. `boreholes.rectangle_field()`). These functions are replaced by methods of the new `Borefield` class. They will be removed in `v3.0.0`.

### Bug fixes

* [Issue 305](https://github.com/MassimoCimmino/pygfunction/issues/305) - Fixed `ClaessonJaved` to return a float when the *g*-function is a vector (i.e. when there is only one heat source). This is required for compatibility with `numpy` version `2.x`.

### Other changes

* [Issue 312](https://github.com/MassimoCimmino/pygfunction/issues/312) - The installation of `matplotlib` is now optional. Using `pip install pygfunction` will not install `matplotlib` and `pip install pygfunction[plot]` should be used instead.

## Version 2.2.3 (2024-07-01)

### New features

* [Issue 276](https://github.com/MassimoCimmino/pygfunction/issues/276) - Added functions to the `boreholes` module for the generation of rectangular fields in a staggered configuration.

### Enhancements

* [Issue 291](https://github.com/MassimoCimmino/pygfunction/issues/291) - Simplified the expressions in heat_transfer._finite_line_source_steady_state`. The function is now approximately 25% faster.

### Bug fixes

* [Issue 255](https://github.com/MassimoCimmino/pygfunction/issues/255) - Default to an `orientation` of `0.` when `tilt` is `0.` in `boreholes.Borehole` class.
* [Issue 266](https://github.com/MassimoCimmino/pygfunction/issues/266) - Fixed an issue were `SingleUTube.get_temperature` returned incorrect results when the fluid-to-pipe wall resistance was small in coaxial configurations. New coefficients are introduced in `SingleUTube.coefficients_temperature` and `SingleUTube.coefficients_outlet_fluid_temperature`. This also solves issues encountered when the fluid mass flow rate is small.
* [Issue 274](https://github.com/MassimoCimmino/pygfunction/issues/274) - Fixed scalar assignment from ndim-1 array. It is deprecated as of `numpy` version `1.25`. Only ndim-0 arrays can be treated as scalars.
* [Issue 285](https://github.com/MassimoCimmino/pygfunction/issues/285) - Use `numpy.complex128` instead of `numpy.cfloat`. This is to comply with backward-incompatible changes introduced in `numpy` version `2.0`.
* [Issue 286](https://github.com/MassimoCimmino/pygfunction/issues/286) - Fixed incorrect coefficients in `pipes.SingleUTube._continuity_condition_base` which caused errors in all dependent class methods when `segment_ratios` were not symmetric around the borehole mid-length.
* [Issue 298](https://github.com/MassimoCimmino/pygfunction/issues/298) - Fixed incorrect coefficients in `pipes._basePipe`, `pipes.MultipleUTube` and `pipes.IndependentMultipleUTube` which caused errors in fluid temperature profiles and outlet fluid temperatures.

## Version 2.2.2 (2023-01-09)

### Enhancements

* [Issue 204](https://github.com/MassimoCimmino/pygfunction/issues/204) - Added support for Python 3.9 and 3.10. [CoolProp](https://www.coolprop.org/) is removed from the dependencies and replace with [SecondaryCoolantProps](https://github.com/mitchute/SecondaryCoolantProps).

### Bug fixes

* [Issue 231](https://github.com/MassimoCimmino/pygfunction/issues/231) - Fixed an issue where the evaluation of g-functions at very low times raises an error due a singular matrix. g-Functions below a threshold time value `t=max(r_b)**2/(25*alpha)` are now linearized.

### Other changes

* [Issue 229](https://github.com/MassimoCimmino/pygfunction/issues/229), [Issue 247](https://github.com/MassimoCimmino/pygfunction/issues/247) - Added citation to IGSHPA conference paper on *pygfunction* v2.2 in the documention. Added a `CITATION.cff` file to suggest a correct citation on github.
* [Issue 230](https://github.com/MassimoCimmino/pygfunction/issues/230) - Configured github actions to publish *pygfunction* on Pypi on creation of a release on github.

## Version 2.2.1 (2022-08-12)

### Bug fixes

* [Issue 220](https://github.com/MassimoCimmino/pygfunction/issues/220) - Fixed the expected line length in `boreholes.field_from_file()` to correctly import fields of inclined boreholes.
* [Issue 224](https://github.com/MassimoCimmino/pygfunction/issues/224) - Fixed an issue where tests were not run on maintenance branches.

## Version 2.2.0 (2022-07-10)

### New features

* [Issue 50](https://github.com/MassimoCimmino/pygfunction/issues/50) - Implemented inclined boreholes for the evaluation of *g*-functions. The implementation includes an approximation of the FLS solution for inclined boreholes based on the method of Cimmino (2021) (see [Issue 138](https://github.com/MassimoCimmino/pygfunction/issues/138)). The `'equivalent'` solver is not yet supported.
* [Issue 138](https://github.com/MassimoCimmino/pygfunction/issues/138) - Implemented the approximation of the finite line source solution of Cimmino (2021). The approximation avoids the numerical evaluation of integrals. This speeds up the calculation of g-functions when enabled.
* [Issue 148](https://github.com/MassimoCimmino/pygfunction/issues/148) - Implemented `effective_borehole_thermal_resistance()` and `local_borehole_thermal_resistance()` methods for all pipe classes. Deprecated `pipes.borehole_thermal_resistance()`, which computed the effective borehole thermal resistance. It will be removed in `v3.0.0`. Implemented a new `update_thermal_resistances()` method to all pipe classes. This method allows to update the delta-circuit of thermal resistance of the boreholes based on provided values for the fluid thermal resistances. This allows simulations with time-variable fluid thermal resistances.

### Enhancements

* [Issue 152](https://github.com/MassimoCimmino/pygfunction/issues/152) - Vectorized `coefficients_temperature` and `_general_solution` in `pipe` objects to accept depths `z` as an array. This speeds up calculations for `get_temperature` and `get_borehole_heat_extraction_rate` class methods.
* [Issue 183](https://github.com/MassimoCimmino/pygfunction/issues/183) - Vectorized `pipes.multipole()` and `pipes._Fmk()` to decrease the calculation time of `pipes.thermal_resistances()`. A `memoization` technique is implemented to reduce computation time for repeated function calls to further speed-up the initialization of `Pipe` and `Network` objects.
* [Issue 198](https://github.com/MassimoCimmino/pygfunction/issues/198) - Refactored the `'detailed'` solver to evaluate same-borehole thermal response factors in a single call to `finite_line_source_vectorized()`. This speeds up calculations of *g*-functions using the `'detailed'` solver.
* [Issue 199](https://github.com/MassimoCimmino/pygfunction/issues/199) - Changed the integral bounds to avoid repeated evaluation of integrals over semi-infinite intervals. This speeds up calculations of *g*-functions using all solvers and the evaluation of the finite line source solution with `time` as an array.
* [Issue 206](https://github.com/MassimoCimmino/pygfunction/issues/206) - Refactored `boreholes.find_duplicates()` to use `scipy.spatial.distance.pdist()` for the calculation of distances between boreholes. This leads to faster initialization of the `gFunction` class for large borefields.

### Other changes

* [Issue 80](https://github.com/MassimoCimmino/pygfunction/issues/80) - Added references to the `pipes` module for the evaluation of borehole thermal resistances.
* [Issue 171](https://github.com/MassimoCimmino/pygfunction/issues/171) - Refactored modules and examples to use the built-in `enumerate(x)` instead of `range(len(x))`.
* [Issue 172](https://github.com/MassimoCimmino/pygfunction/issues/172) - Refactored reports of calculation time to use `time.perf_counter()` instead of `time.time()`.
* [Issue 173](https://github.com/MassimoCimmino/pygfunction/issues/173) - Refactored strings into f-strings instead of using `str.format()`.
* [Issue 177](https://github.com/MassimoCimmino/pygfunction/issues/177) - Converted `setup.py` script to `setup.cfg` and `pyproject.toml` files. This is motivated by [PEP518](https://www.python.org/dev/peps/pep-0518/) and [PEP621](https://www.python.org/dev/peps/pep-0621/).
* [Issue 179](https://github.com/MassimoCimmino/pygfunction/issues/179) - Refactored tests to use the `pytest` package instead of `unittests`.
* [Issue 180](https://github.com/MassimoCimmino/pygfunction/issues/180) - Configured `tox` and github actions for continuous integration.

### Bug fixes

* [Issue 192](https://github.com/MassimoCimmino/pygfunction/issues/192) - Fixed comparison of `time` with `numpy.inf` in `heat_transfer.finite_line_source` that caused the function to fail when `time` is an array.
* [Issue 193](https://github.com/MassimoCimmino/pygfunction/issues/193) - Fixed `heat_transfer._finite_line_source_integrand`, `heat_transfer._finite_line_source_equivalent_boreholes_integrand`, and `heat_transfer._finite_line_source_steady_state` to return an array of zeros of the expected shape when `reaSource==False and imgSource==False`.
* [Issue 196](https://github.com/MassimoCimmino/pygfunction/issues/196) - Fixed "invalid escape sequence" warnings when running tests on github actions.
* [Issue 202](https://github.com/MassimoCimmino/pygfunction/issues/202) - Added missing package `recommonmark` to requirements for documentation and development.
* [Issue 208](https://github.com/MassimoCimmino/pygfunction/issues/208) - Fixed an issue where `boreholes.field_from_file()` failed when the text file only contained 1 borehole.

## Version 2.1.0 (2021-11-12)

### New features

* [Issue 36](https://github.com/MassimoCimmino/pygfunction/issues/36) - Added a `Coaxial` class to the `pipes` module to model boreholes with coaxial pipes.
* [Issue 135](https://github.com/MassimoCimmino/pygfunction/issues/135) - Added functionality for non-uniform discretization of the segments along the boreholes. This increases the accuracy of *g*-function calculations for the same number of segments when compared to a uniform discretization. Segment lengths are defined using the `segment_ratios` option in the `gFunction` class. A `discretize` function is added to the `utilities` module to generate borehole discretizations using an expanding mesh.
* [Issue 146](https://github.com/MassimoCimmino/pygfunction/issues/146) - Added new solver `'equivalent'` to the `gFunction` class. This solver uses hierarchical agglomerative clustering to identify groups of boreholes that are expected to have similar borehole wall temperatures and heat extraction rates. Each group of boreholes is represented by a single equivalent borehole. The FLS solution is adapted to evaluate thermal interactions between groups of boreholes. This greatly reduces the number of evaluations of the FLS solution and the size of the system of equations to evaluate the g-function.

### Enhancements

* [Issue 118](https://github.com/MassimoCimmino/pygfunction/issues/118) - Refactored checks for stored `_BasePipe` and `Network` coefficicients to use `numpy.all()`. This decreases calculation time.
* [Issue 119](https://github.com/MassimoCimmino/pygfunction/issues/119) - Refactored `Network` class to change how coefficient matrices are calculated. This decreases calculation time.
* [Issue 132](https://github.com/MassimoCimmino/pygfunction/issues/132) - Refactored `SingleUtube` and `MultipleUTube` classes to eliminate `for` loops in the calculation of matrix coefficients. This decreases calculation time when `nSegments>>1`.
* [Issue 133](https://github.com/MassimoCimmino/pygfunction/issues/133) - The `nSegments` argument is now able to take in the number of segments for each borehole as a list. Each borehole must be split into at least 1 segment, and the length of the segment list must be equal to the number of boreholes.
* [Issue 141](https://github.com/MassimoCimmino/pygfunction/issues/141) - Changed the calculation of the convective heat transfer coefficient in the transition region (`2300. < Re < 4000.`) by `convective_heat_transfer_coefficient_circular_pipe()`. The Nusselt number is now interpolated between the laminar value (at `Re = 2300.`) and the turbulent value (at `Re = 4000.`). This avoids any discontinuities in the values of the convective heat transfer coefficient near `Re = 2300.`.

### Other changes

* [Issue 93](https://github.com/MassimoCimmino/pygfunction/issues/93) - Reformatted `pipes` and `networks` modules to use the `@` matrix product operator introduced in [PEP465](https://www.python.org/dev/peps/pep-0465/). This improves readability of the code.
* [Issue 100](https://github.com/MassimoCimmino/pygfunction/issues/100) - Replaced calls to `numpy.asscalar()` with calls to `array.item()`. `numpy.asscalar()` is deprecated as of `numpy` version `1.16`.
* [Issue 124](https://github.com/MassimoCimmino/pygfunction/issues/124) - Reformatted `pipes`and `networks` modules to clarify the expected values for `m_flow` parameters. These are replaced by any of `m_flow_pipe`, `m_flow_borehole` or `m_flow_network` depending on the function or class method. Added a nomenclature of commonly used variables to the documentation.
* [Issue 125](https://github.com/MassimoCimmino/pygfunction/issues/125) - Refactored class methods and docstrings in `Pipe` and `Network` objects to better represent the expected shapes of array inputs and outputs.
* [Issue 139](https://github.com/MassimoCimmino/pygfunction/issues/139) - Updated requirements for numpy from version `1.19.2` to `1.20.1`. Clarified the Python version `3.7` requirement in the `README.md` file.
* [Issue 154](https://github.com/MassimoCimmino/pygfunction/issues/154) - Replaced `numpy.int` and `numpy.bool` dtypes in array initializations with built-in types `int` and `bool`. `numpy.int` and `numpy.bool` are deprecated as of `numpy` version `1.20`.
* [Issue 158](https://github.com/MassimoCimmino/pygfunction/issues/158) - Changed default parameter values for *g*-function calculations. The `gFunction` class now uses the `'equivalent'` solver by default and a non-uniform discretization of `nSegments=8` given by `utilities.segment_ratios()`.
* [Issue 160](https://github.com/MassimoCimmino/pygfunction/issues/160) - Deprecated functions `gfunction.uniform_heat_extraction`, `gfunction.uniform_temperature`, `gfunction.equal_inlet_temperature` and `gfunction.mixed_inlet_temperature`. They will be removed in `v3.0.0`.

### Bug fixes

* [Issue 99](https://github.com/MassimoCimmino/pygfunction/issues/99) - Fixed an issue where `MultipleUTube._continuity_condition()` and `MultipleUTube._general_solution()` returned complex valued coefficient matrices.
* [Issue 130](https://github.com/MassimoCimmino/pygfunction/issues/130) - Fix incorrect initialization of variables `_mix_out` and `_mixing_m_flow` in `Network`.
* [Issue 155](https://github.com/MassimoCimmino/pygfunction/issues/155) - Fix incorrect initialization of variables in `Network` and `_BasePipe`. Stored variables are now initialized as `numpy.nan` instead of `numpy.empty`.
* [Issue 159](https://github.com/MassimoCimmino/pygfunction/issues/159) - Fix `segment_ratios` function in the `utilities` module to always expect 0 < `end_length_ratio` < 0.5, and allows for `nSegments=1` or `nSegments=2`. If 1<=`nSegments`<3 then the user is warned that the `end_length_ratio` parameter is being over-ridden.

## Version 2.0.0 (2021-05-22)

### New features

* [Issue 33](https://github.com/MassimoCimmino/pygfunction/issues/33), [Issue 54](https://github.com/MassimoCimmino/pygfunction/issues/54), [Issue 85](https://github.com/MassimoCimmino/pygfunction/issues/85) - New class `gFunction` for the calculation of g-functions. The new class is a common interface to all boundary conditions and calculation methods. The new implementation of the solver reduces the memory requirements of pygfunction. The new class implements visualization features for the g-function and for heat extraction rates and borehole wall temperatures (both as a function of time and for the profiles along the borehole lengths).
* [Issue 75](https://github.com/MassimoCimmino/pygfunction/issues/75) - New module `media` with properties of brine mixtures.
* [Issue 81](https://github.com/MassimoCimmino/pygfunction/issues/81) - Added functions to find a remove duplicate boreholes.

### Enhancements

* [Issue 78](https://github.com/MassimoCimmino/pygfunction/issues/78), [Issue 109](https://github.com/MassimoCimmino/pygfunction/issues/109) - Optimization of solvers for the calculation of g-functions. The finite line source (FLS) solution is now calculated using `scipy.integrate.quad_vec` which significantly improves calculation time over `scipy.integrate.quad`. The identification of similarities in the 'similarities' solver has also been refactored to identify similarities between boreholes as an intermediate step before identifying similarities between segments. The calculation time for the identification of similarities is significantly decreased.
* [Issue 94](https://github.com/MassimoCimmino/pygfunction/issues/94) - Refactor visualization functions and methods to uniformize figure styles across modules.
* [Issue 108](https://github.com/MassimoCimmino/pygfunction/issues/108) - Optimize the load aggregation algorithm of Claesson and Javed using `numpy.einsum()`.
* [Issue 112](https://github.com/MassimoCimmino/pygfunction/issues/112) - Optimization of `_BaseSolver.temporal_superposition()`. The computationally expensive for loop is replaced by a call to `numpy.einsum()`. This decreases the calculation time of large bore fields.
* [Issue 114](https://github.com/MassimoCimmino/pygfunction/issues/114) - Optimization of `_finite_line_source_integrand()`. The call to `_erfint()` is now vectorized. This decreases the number of calls by a factor 8 during integration. The calculation time of g-functions is decreased, especially for smaller bore fields.

### Bug fixes

* [Issue 86](https://github.com/MassimoCimmino/pygfunction/issues/86) - Documentation is now built using Python 3 to support Python 3 features in the code.
* [Issue 103](https://github.com/MassimoCimmino/pygfunction/issues/103) - Fixed `gFunction` class to allow both builtin `float` and `numpy.floating` inputs.
* [Issue 104](https://github.com/MassimoCimmino/pygfunction/issues/104) - Raise an error if g-function is calculated with inclined boreholes. This will be supported in a later version of *pygfunction*.

### Other changes

* [Issue 72](https://github.com/MassimoCimmino/pygfunction/issues/72) - Added a list of contributors to the front page. The list is managed using [all-contributors](https://github.com/all-contributors/all-contributors).
* [Issue 87](https://github.com/MassimoCimmino/pygfunction/issues/87) - Drop support for Python 2. All package requirements are updated to the latest conda version.
* [Issue 96](https://github.com/MassimoCimmino/pygfunction/issues/96) - Added a reference to the conference paper introducing pygfunction in the documentation.

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
