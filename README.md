# miniproject_OPTML
 
## Systematic Generation and Mitigation of Distribution Shifts in Federated Learning

This is the repository for the project of the course CS439 - Optimization for Machine Learning at EPFL in 2025. 

This repository contains all files used to create the results mentioned in the project report.
Please have a look at the results section to recreate our results.

## Project Setup Instructions
To use this project follow the setup instructions provided. You can use a conda environment (recommended) or a standard Python virtual environment.

Installation Instructions

Install Anaconda if not already installed.

Create a new conda environment:

conda create -n project_env python=3.11
Activate the conda environment:

conda activate project_env
Install pip inside the environment:

conda install pip
Install pytorch on your machine: Visit https://pytorch.org and follow the instructions to install pytorch on your system. Take care to install the CUDA version if you want to use GPU support locally.

## Project Data

The directory structure of our data for this project is the following:

├── emnist-digits.mat <- EMNIST project data files 

## Project Structure 

├── features_shift.ipynb <- Notebook related to Results - Feature Distribution Shift
├── LabelShift.ipynb <- Notebook related to Results - Feature Distribution Shift

## Results

To recreate our results, you can simply run both notebooks i.e features_shift.ipynb and LabelShift.ipynb






