# miniproject_OPTML
 
## Systematic Generation and Mitigation of Distribution Shifts in Federated Learning

This is the repository for the project of the course CS439 - Optimization for Machine Learning at EPFL in 2025. 

This repository contains all files used to create the results mentioned in the project report.
Please have a look at the results section to recreate our results.

## Installation Instructions

1. Install Anaconda if not already installed.

2. Create a new conda environment:
   ```bash
   conda create -n project_env python=3.11
   ```
3. Activate the conda environment:
    ```bash
   conda activate project_env
   ```

4. Install pip inside the environment:
    ```bash
   conda install pip
   ```
5. Install pytorch on your machine:
    Visit https://pytorch.org and follow the instructions to install pytorch on your system. Take care to install the CUDA version if you want to use GPU support locally.

6. Install project dependencies:
    After installing pytorch successfully install the project dependencies with `pip` by running the command:
    ```bash
   pip install -r requirements.txt
   ```

## Project Data

The directory structure of our data for this project is the following:
```
├── emnist-digits.mat <- EMNIST project data files 
```
## Project Structure 
```
├── features_shift.ipynb <- Notebook related to Results - Feature Distribution Shift
│
├── LabelShift.ipynb <- Notebook related to Results - Feature Distribution Shift
│
├── features_utils.py <- All functions related to Feature Distribution Shift
│
├── labels_utils.py <- All functions related to Label Distribution Shift
│
├── utils.py <- helpers functions
```
## Results

To recreate our results, you can simply run both notebooks i.e features_shift.ipynb and LabelShift.ipynb
