# Contributing to pygfunction

*pygfunction* welcomes and appreciates bug reports, suggestions and
contributions from everyone.

This guide describes how to report bugs, suggest new features and
contribute code to *pygfunction*.


## Reporting bugs

Bugs are reported on the
[issue tracker](https://github.com/MassimoCimmino/pygfunction/issues).

Follow these steps when submitting a bug report:

1. **Make sure the bug has not been already reported.** Run a quick search
through the [issue tracker](https://github.com/MassimoCimmino/pygfunction/issues).
If an open issue is related to your problem, consider adding your input to that
issue. If you find a closed issue related to your problem, open an new issue
and link to the closed issue.
2. **Use a short and descriptive title** when creating a new issue.
3. **Provide detailed steps to reproduce the problem.** Explain, in details,
how to reproduce the problem and what should be the **expected result.** When
possible, include a simple code snippet that isolates and reproduces the
problem.

After submitting a bug report, if you wish to contribute code to fix the
problem, follow the steps outlined in the contribution workflow.


## Contribution workflow

This section outlines the steps for contributing to *pygfunction*. This workflow
is inspired by [Trunk Based Development](https://trunkbaseddevelopment.com/), with
short-lived feature branches and releases from tags on the `master`.

1. **Open a new [issue](https://github.com/MassimoCimmino/pygfunction/issues).**
2. **Use a short and descriptive title.** When proposing an enhancement,
describe in details what the enhancement would entail. If you plan to implement
the enhancement yourself, provide a step-by-step plan for the implementation.
3. **Explain how the enhancement benefits _pygfunction_.**
4. **Create (checkout) a new branch from the `master`.** The branch name should
follow the naming convention: `issue#_shortDescription`. For example:
issue1_loadAggregation.
5. Implement unit tests for new features. If necessary, update already
implement tests to cover the new features.
6. Before submitting a
[pull request](https://github.com/MassimoCimmino/pygfunction/pulls), **merge
the master to your branch.**
7. Once the branch is merged, **delete the branch and close the issue.**


## Styleguide

*pygfunction* follows the
[PEP8 style guide](https://www.python.org/dev/peps/pep-0008).
Docstrings are written following the
[numpydoc format](https://github.com/numpy/numpy/blob/master/doc/example.py),
see also an example
[here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
