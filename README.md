Protoclass
=========

#### Manifesto

Because *Human* is **perfectible** and **error-prone**, because *Science* should be **open** and **flow** and because *cogito ergo sum*.

Goal
----

This toolbox will aggregate some machine learning tools from all the walk of lives in order to fast prototype.

File Structure
--------------

```
.
├── doc
├── LICENSE
├── protoclass
├── README.md
├── setup.py
└── third-party
```

Installation
------------

### Cloning

You can clone this repository with the usual `git clone`.

### Dependencies

This package needs the following dependencies:

* Numpy,
* Scipy,
* Scikit-learn,
* Scikit-image,
* UnbalancedDataset,
* Pyksvd.

The majority of the package are package which can be found in conda or pypi repositories.
The packages not available are present as submodule in the folder `third-party`.
To be able to install them, you need to:

1. Initialise the submodule using `git submodule init` at the root of the repository.
1. Download the source code using `git submodule update` at the root of the repository.
1. Enter in each sub-repository and follow the install instructions.

### Installation

You need to run the command `python setup.py install`.
