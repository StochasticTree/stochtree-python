# StochasticTree Python Package

**NOTE**: we are in the process of refactoring this project so that the R, Python, and C++ source code sits in the [same repo](https://github.com/StochasticTree/stochtree-cpp/).

## Getting started

The python package can be installed from source. Before you begin, make sure you have [conda](https://www.anaconda.com/download) installed.
Clone the repo recursively (including git submodules) by running 

```{bash}
git clone --recursive https://github.com/StochasticTree/stochtree-python.git
```

### Conda-based installation

Conda provides a straightforward experience in managing python dependencies, avoiding version conflicts / ABI issues / etc.

To build stochtree using a `conda` based workflow, first create and activate a conda environment with the requisite dependencies

```{bash}
conda create -n stochtree-dev -c conda-forge python=3.10 numpy scipy pytest pandas pybind11 scikit-learn matplotlib seaborn
conda activate stochtree-dev
pip install jupyterlab
```

Then, navigate to the main `stochtree-python` project folder (i.e. `cd /path/to/stochtree-python`) and install the package locally via pip

```{bash}
pip install .
```

### Pip-based installation

If you would rather avoid installing and setting up conda, you can alternatively setup the dependencies and install `stochtree` using only `pip` (caveat: this has not been extensively tested 
across platforms and python versions).

First, navigate to the main `stochtree-python` project folder (i.e. `cd /path/to/stochtree-python`) and create and activate a virtual environment as a subfolder of the repo

```{bash}
python -m venv venv
source venv/bin/activate
```

Install all of the package (and demo notebook) dependencies

```{bash}
pip install numpy scipy pytest pandas scikit-learn pybind11 matplotlib seaborn jupyterlab
```

Then install stochtree via

```{bash}
pip install .
```
