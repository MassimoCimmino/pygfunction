.. install:

**********************
Setting up pygfunction
**********************

*pygfunction* uses Python 3.9, along with the following packages:
	- numpy (>= 1.26.4)
	- scipy (>= 1.13.1)
	- SecondaryCoolantProps (>= 1.3)
	- typing_extensions (>= 4.11.0)
	- (optionally) matplotlib (>= 3.9.2)

*pygfunction*'s- documentation is built using:
	- sphinx (>= 7.3.7)
	- numpydoc (>= 1.7.0)

**Users** - `Download pip <https://pip.pypa.io/en/latest/>`_ and install the
latest release:

.. code:: shell

	pip install pygfunction[plot]

Alternatively, `download the latest release
<https://github.com/MassimoCimmino/pygfunction/releases>`_ and run the
installation script:

.. code:: shell

	pip install .

**Developers** - To get the latest version of the code, you can `download the
repository from github <https://github.com/MassimoCimmino/pygfunction>`_ or
clone the project in a local directory using git:

.. code:: shell

	git clone https://github.com/MassimoCimmino/pygfunction.git

Install *pygfunction* in development mode (this requires `pip >= 21.1`):

.. code:: shell

	pip install --editable .

Test that *pygfunction* is running correctly by running any of the
provided examples in ``../pygfunction/examples/``
