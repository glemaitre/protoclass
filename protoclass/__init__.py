"""Annonce something here.
"""

import os
import imp
import functools
import warnings
import sys

__version__ = '0.1.dev0'

pkg_dir = os.path.abspath(os.path.dirname(__file__))

# Logic for checking for improper install and importing while in the source
# tree when package has not been installed inplace.
# Code adapted from scikit-learn's __check_build module.
_INPLACE_MSG = """
It appears that you are importing a local scikit-cycling source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

_STANDARD_MSG = """
Your install of scikit-image appears to be broken. """


def _raise_build_error(e):
    import os.path as osp
    # Raise a comprehensible error
    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == "protoclass":
        # Picking up the local install: this will work only if
        # the install is an 'inplace build'
        msg = _INPLACE_MSG
    raise ImportError("""%s
    It seems that protoclass has not been built correctly.
    %s""" % (e, msg))

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __PROTOCLASS_SETUP__
except NameError:
    __PROTOCLASS_SETUP__ = False

if __PROTOCLASS_SETUP__:
    sys.stderr.write('Partial import of protoclass during the build process.\n')
    # We are not importing the rest of the scikit during the build
    # process, as it may not be compiled yet
else:
    __all__ = ['classification',
               'data_management',
               'extraction',
               'preprocessing',
               'selection',
               'tool',
               'validation']

del warnings, functools, os, imp, sys
