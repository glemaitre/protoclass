.PHONY: all clean test
PYTHON=python
NOSETESTS=nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	rm -rf coverage
	rm -rf dist
	rm -rf build

test:
	$(NOSETESTS) -s -v protoclass

# doctest:
# 	$(PYTHON) -c "import protoclass, sys, io; sys.exit(protoclass.doctest_verbose())"

coverage:
	$(NOSETESTS) -s -v protoclass --with-coverage --cover-package=protoclass

html:
	conda install sphinx
	export SPHINXOPTS=-W; make -C doc html
