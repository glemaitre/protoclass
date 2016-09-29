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

# Install libgfortran, cython, scikit-image, and simpleitk with conda
conda install --yes cython scikit-image
conda install --yes -c https://conda.anaconda.org/simpleitk SimpleITK
conda install --yes libgfortran
conda install --yes scikit-learn
conda install --yes -c glemaitre imbalanced-learn=0.1.8
conda install --yes seaborn
conda install --yes numpydoc
conda install --yes sphinx
conda install --yes sphinx_rtd_theme

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# Build scikit-cycling in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# Install pyFFTW
cd $TRAVIS_BUILD_DIR/third-party/pyFFTW
python setup.py install
# Install phasepack
cd $TRAVIS_BUILD_DIR/third-party/phasepack
python setup.py install
# Install mahotas
cd $TRAVIS_BUILD_DIR/third-party/mahotas
python setup.py install
# Install pyksvd
cd $TRAVIS_BUILD_DIR/third-party/pyksvd
python setup.py install
# Install statsmodel
cd $TRAVIS_BUILD_DIR/third-party/statsmodel
python setup.py install
# Install lmfit
cd $TRAVIS_BUILD_DIR/third-party/lmfit-py
python setup.py install


cd $TRAVIS_BUILD_DIR
python setup.py develop

# Build the doc
cd $TRAVIS_BUILD_DIR/doc
make html

cd $TRAVIS_BUILD_DIR
