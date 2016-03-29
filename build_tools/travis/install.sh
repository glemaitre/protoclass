#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone scikit-learn/scikit-learn repository in to a local repository.
# We use a cached directory with three scikit-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip


if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    pushd .
    cd
    mkdir -p download
    cd download
    echo "Cached in $HOME/download :"
    ls -l
    echo
    if [[ ! -f miniconda.sh ]]
        then
        wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
            -O miniconda.sh
        fi
    chmod +x miniconda.sh && ./miniconda.sh -b
    cd ..
    export PATH=/home/travis/miniconda2/bin:$PATH
    conda update --yes conda
    popd

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
        libgfortran nomkl
    source activate testenv

    # Install nose-timer via pip
    pip install nose-timer

    # Install libgfortran cython and simpleitk with conda
    conda install --yes cython scikit-image
    conda install --yes -c https://conda.anaconda.org/simpleitk SimpleITK
    conda install --yes libgfortran
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

if [ ! -d "$CACHED_BUILD_DIR" ]; then
    mkdir -p $CACHED_BUILD_DIR
fi

rsync -av --exclude '.git/' --exclude='testvenv/' \
      $TRAVIS_BUILD_DIR $CACHED_BUILD_DIR

cd $CACHED_BUILD_DIR/protoclass

# Build protoclass in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop

# Install Unbalanced-dataset
cd $CACHED_BUILD_DIR/protoclass/third-party/UnbalancedDataset
python setup.py install
# Install pyFFTW
cd $CACHED_BUILD_DIR/protoclass/third-party/pyFFTW
python setup.py install
# Install phasepack
cd $CACHED_BUILD_DIR/protoclass/third-party/phasepack
python setup.py install
# Install mahotas
cd $CACHED_BUILD_DIR/protoclass/third-party/mahotas
python setup.py install
# Install pyksvd
cd $CACHED_BUILD_DIR/protoclass/third-party/pyksvd
python setup.py install
# Install scikit-learn
cd $CACHED_BUILD_DIR/protoclass/third-party/scikit-learn
python setup.py install

# Go back to the protoclass directory to run the test
cd $CACHED_BUILD_DIR/protoclass
