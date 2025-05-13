# pygfunction: A g-function calculator for Python

[![Tests](https://github.com/MassimoCimmino/pygfunction/actions/workflows/test.yml/badge.svg)](https://github.com/MassimoCimmino/pygfunction/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/100305705.svg)](https://zenodo.org/badge/latestdoi/100305705)

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
configuration (i.e. arbitrarily positioned in space), including fields of
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

*pygfunction* was developed and tested using Python 3.9. In addition, the
following packages are needed to run *pygfunction* and its examples:
- matplotlib (>= 3.9.2),
- numpy (>= 1.26.4)
- scipy (>= 1.13.1)
- SecondaryCoolantProps (>= 1.3)
- typing_extensions >= 4.11.0

The documentation is generated using [Sphinx](http://www.sphinx-doc.org). The
following packages are needed to build the documentation:
- sphinx (>= 7.3.7)
- numpydoc (>= 1.7.0)


## Quick start

**Users** - [Download pip](https://pip.pypa.io/en/latest/) and install the latest release:

```
pip install pygfunction[plot]
```

Alternatively, [download the latest release](https://github.com/MassimoCimmino/pygfunction/releases) and run the installation script:

```
pip install .
```

**Developers** - To get the latest version of the code, you can [download the
repository from github](https://github.com/MassimoCimmino/pygfunction) or clone
the project in a local directory using git:

```
git clone https://github.com/MassimoCimmino/pygfunction.git
```

Install *pygfunction* in development mode (this requires `pip >= 21.1`):
```
pip install --editable .
```

Once *pygfunction* is copied to a local directory, you can verify that it is
working properly by running the examples in `pygfunction/examples/`.


## Documentation

*pygfunction*'s documentation is hosted on
[ReadTheDocs](https://pygfunction.readthedocs.io).


## License

*pygfunction* is licensed under the terms of the 3-clause BSD-license.
See [pygfunction license](LICENSE.md).


## Contributing to *pygfunction*

You can report bugs and propose enhancements on the
[issue tracker](https://github.com/MassimoCimmino/pygfunction/issues).

To contribute code to *pygfunction*, follow the
[contribution workflow](CONTRIBUTING.md).


## Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.polymtl.ca/expertises/en/cimmino-massimo"><img src="https://avatars.githubusercontent.com/u/23085996?v=4?s=100" width="100px;" alt="Massimo Cimmino"/><br /><sub><b>Massimo Cimmino</b></sub></a><br /><a href="https://github.com/MassimoCimmino/pygfunction/commits?author=MassimoCimmino" title="Code">💻</a> <a href="https://github.com/MassimoCimmino/pygfunction/commits?author=MassimoCimmino" title="Documentation">📖</a> <a href="#example-MassimoCimmino" title="Examples">💡</a> <a href="http://www.ibpsa.org/proceedings/eSimPapers/2018/2-3-A-4.pdf" title="Founder">:rocket:</a> <a href="#ideas-MassimoCimmino" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-MassimoCimmino" title="Maintenance">🚧</a> <a href="https://github.com/MassimoCimmino/pygfunction/pulls?q=is%3Apr+reviewed-by%3AMassimoCimmino" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/j-c-cook"><img src="https://avatars.githubusercontent.com/u/39248734?v=4?s=100" width="100px;" alt="Jack Cook"/><br /><sub><b>Jack Cook</b></sub></a><br /><a href="https://github.com/MassimoCimmino/pygfunction/commits?author=j-c-cook" title="Code">💻</a> <a href="#example-j-c-cook" title="Examples">💡</a> <a href="#ideas-j-c-cook" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/MassimoCimmino/pygfunction/commits?author=j-c-cook" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mitchute"><img src="https://avatars.githubusercontent.com/u/2985979?v=4?s=100" width="100px;" alt="Matt Mitchell"/><br /><sub><b>Matt Mitchell</b></sub></a><br /><a href="https://github.com/MassimoCimmino/pygfunction/commits?author=mitchute" title="Code">💻</a> <a href="#ideas-mitchute" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://ghetool.eu"><img src="https://avatars.githubusercontent.com/u/52632307?v=4?s=100" width="100px;" alt="Wouter Peere"/><br /><sub><b>Wouter Peere</b></sub></a><br /><a href="https://github.com/MassimoCimmino/pygfunction/commits?author=wouterpeere" title="Code">💻</a> <a href="#ideas-wouterpeere" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/MassimoCimmino/pygfunction/issues?q=author%3Awouterpeere" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tblanke"><img src="https://avatars.githubusercontent.com/u/86232208?v=4?s=100" width="100px;" alt="Tobias Blanke"/><br /><sub><b>Tobias Blanke</b></sub></a><br /><a href="https://github.com/MassimoCimmino/pygfunction/commits?author=tblanke" title="Code">💻</a> <a href="#ideas-tblanke" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/MassimoCimmino/pygfunction/issues?q=author%3Atblanke" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/axelstudios"><img src="https://avatars.githubusercontent.com/u/411466?v=4?s=100" width="100px;" alt="Alex Swindler"/><br /><sub><b>Alex Swindler</b></sub></a><br /><a href="https://github.com/MassimoCimmino/pygfunction/commits?author=axelstudios" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
