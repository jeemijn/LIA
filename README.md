# README

This repository contains the analysis code used in the paper:

**Scheen, Jeemijn and Stocker, Thomas F., "Effect of changing ocean circulation on deep ocean temperature in the last millennium", Earth System Dynamics Discussions, 2020**

It loads the model output data (which must be downloaded from a zenodo repository) and reproduces all figures.

All code is gathered in a well-documented ipython notebook (LIA.ipynb) with in addition a functions.py file, from where user-defined functions are loaded.  

N.B. Unfortunately, the history of this repository is hard to follow, since commits in LIA.ipynb also contain output cells. This choice has been made s.t. the figures are visible in the static preview.  

## A. For a static preview of the code, click [here](https://github.com/jeemijn/LIA/blob/master/LIA.ipynb)
This is only a snapshot and cannot be adjusted. If the preview doesn't work, reload or switch browser. 

## B. How to run the notebook yourself:
Make any changes you want, or reuse specific functions or figures in your own research ([license](https://github.com/jeemijn/LIA/blob/master/LICENSE)). Please consider to cite the doi of this code repository and/or the original paper. Final-version doi's for data and for code are cited in the paper mentioned above under Code and data availability.

Steps:
1. Download the simulation output data here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3878835.svg)](https://doi.org/10.5281/zenodo.3878835). You can choose to download either the small or large version. Note that this is not only about storage, but **this notebook loads 4 Gb into memory in the small version (default) or 22 Gb in the large version**. You only need the small version to run the notebook and reproduce the figures, but you are free to explore additional variables in the large version.
2. Download this repository on zenodo as zip or on github via the 'Clone or download' button at the top right and choose Download ZIP. Unzip.
3. Place the unzipped data folder (step 1) inside the repository (step 2). Now your folder contains: LIA.ipynb, functions.py, figures directory and a data_small or data_large directory (containing data; no subdirectories).
4. If you don't already have it: install jupyter lab or jupyter notebook [here](https://jupyter.org/install) with python 3 and run it from this folder ([jupyter lab](https://jupyterlab.readthedocs.io/en/latest/getting_started/starting.html); [jupyter notebook](https://jupyter.readthedocs.io/en/latest/running.html)). 
5. You are all set! More documentation is inside the notebook.  
