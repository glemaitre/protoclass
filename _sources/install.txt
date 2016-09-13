Getting Started
===============

Install
-------

The install of ``protoclass`` is almost straightforward. You need to clone it from GitHub_::

  $ git clone --recursive https://github.com/glemaitre/protoclass.git

There is a need to install the different dependencies::

  $ apt-get install libfftw3-dev libfftw3-3 eigen3
  $ conda install cython scikit-image
  $ conda install -c https://conda.anaconda.org/simpleitk SimpleITK
  $ cd third-party/UnbalancedDataset
  $ python setup.py install
  $ cd ../pyFFTW
  $ python setup.py install
  $ ../phasepack
  $ python setup.py install
  $ cd ../mahotas
  $ python setup.py install
  $ cd ../pyksvd
  $ python setup.py install
  $ ../scikit-learn
  $ python setup.py install

And finally::

  $ cd ../../
  $ python setup.py install

Test and coverage
-----------------

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

Contribute
----------

You can contribute to this code through Pull Request on GitHub_. Please, make sure that your code is coming with unit tests to ensure full coverage and continuous integration in the API.

.. _GitHub: https://github.com/glemaitre/protoclass
