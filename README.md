# README

This repository contains the analysis code used in the paper:

Scheen, Jeemijn and Stocker, Thomas F., "Effect of changing ocean circulation on deep ocean temperature in the last millennium", submitted (2020)

It loads the model output data (which must be downloaded from a zenodo repository) and reproduces all figures.

All code is gathered in a well-documented ipython notebook with in addition a functions.py file, from where user-defined functions are loaded.

## Options:
A. Look at the static code **HERE**. This is only a snapshot and cannot be adjusted. This option is useful if you don't want to run the notebook yourself, since you can just read or copy the python code of the part you are interested in.
B. Run the notebook yourself and make any changes you want, or reuse specific functions or figures in your own research (**see license**).

## Required steps for option B:
1. Download the simulation output data from zenodo: **here**. You can choose to download either the small (4 Gb) or large (22 Gb) version of the dataset. You only need the small version to run the notebook and reproduce the figures, but you are free to explore additional variables in the large version. Unzip and rename the folder you chose to 'data'
2. Download this repository **TODO how**
3. Place the 'data' folder (step 1) inside the repository (step 2). Now your folder should contain: **name**.ipynb, functions.py, figures (empty folder), and data (folder containing data; no subfolders).
4. If you do not already have it: install python 3 and e.g. jupyter notebook or jupyter lab **todo links**to run the ipython notebook. 
5. You are all set! The rest of the documentation is inside the notebook. 
