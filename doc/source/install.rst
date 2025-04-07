.. install:

**********************
Setting up pygfunction
**********************

*pygfunction* uses Python 3.8, along with the following packages:
	- numpy (>= 1.21.5)
	- scipy (>= 1.7.3)
	- SecondaryCoolantProps (>= 1.1)
	- typing_extensions (>= 4.0.1)
	- (optionally) matplotlib (>= 3.8.4)

*pygfunction*'s- documentation is built using:
	- sphinx (>= 4.4.0)
	- numpydoc (>= 1.2.0)

**Users** - `Download pip <https://pip.pypa.io/en/latest/>`_ and install the
latest release:

.. code:: shell

	pip install pygfunction

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
