.PHONY: all clean test
PYTHON=python
NOSETESTS=nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build

test:
	$(NOSETESTS) -s -v protoclass

# doctest:
# 	$(PYTHON) -c "import protoclass, sys, io; sys.exit(protoclass.doctest_verbose())"

coverage:
	$(NOSETESTS) protoclass --with-coverage --cover-package=protoclass

html:
	conda install sphinx
	export SPHINXOPTS=-W; make -C doc html
