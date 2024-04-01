# StochasticTree Python Package

## Getting started

The python package can be installed from source. Before you begin, make sure you have [conda](https://www.anaconda.com/download) installed.
Clone the repo following the instructions in [the "cloning the repository" section](#cloning-the-repository) above.

Next, create and activate a conda environment with the requisite dependencies

```{bash}
conda create -n stochtree-dev -c conda-forge python=3.10 numpy scipy pytest pandas pybind11
conda activate stochtree-dev
conda install -c conda-forge matplotlib seaborn
pip install jupyterlab
```

Then, navigate to the main `stochtree-python` project folder (i.e. `cd /path/to/stochtree-python`) and install the package locally via pip

```{bash}
pip install .
```

