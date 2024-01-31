# PACS Project of Andrea Bonifacio and Sara Gazzoni

## Reduced Order Models using Graph Neural Networks

### Introduction
This repository contains the code for a Python-based library for the implementation of Reduced Order Models (ROMs) using Graph Neural Networks (GNNs). The code is based on the paper [Learning Reduced-Order Models for Cardiovascular Simulations with Graph Neural Networks](https://arxiv.org/abs/2303.07310). We built the library with the aim of creating a fully working pipeline from the generation of the data to the training of the model and the evaluation of the results. 
We divided the code into three main folders:
- `scripts`: contains the scripts for the generation of the data;
- `gNN`: contains the code for the implementation of the GNN;
- `notebooks`: contains some example notebooks to test the library.

### Prerequisites
- Python 3.x
- Required packages: FEniCS, numpy, matplotlib, torch, dgl, gmsh, meshio, scipy, tqdm

### Installation
This installation procedure assumes that the user has already installed FEniCS in an Anaconda environment. If this is not the case, please refer to the [FEniCS installation guide](https://fenicsproject.org/download/archive/). 
1. Clone the repository
2. Activate the FEniCS environment
3. Install the required packages using `pip install -r requirements.txt`




