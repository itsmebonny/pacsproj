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
- Required packages: FEniCS, numpy, matplotlib, torch, dgl, gmsh, meshio, scipy, tqdm, jupyter 

### Installation
This installation procedure assumes that the user has already installed FEniCS in an Anaconda environment. If this is not the case, please refer to the [FEniCS installation guide](https://fenicsproject.org/download/archive/). 
1. Clone the repository
2. Activate the FEniCS environment
3. Install the required packages using `pip install -r requirements.txt`
4. Check if the installation was successful by running the script `python scripts/test_installation.py`

### Usage 
The library is able to create meshes, solve variational problems, save them in a suitable format and train a GNN on the data. The user can choose which part of the pipeline to run. We will now describe each step separately. If the user wants to run the whole pipeline, please refer to the subsection **Notebooks**.

#### Mesh generation
The user can generate a mesh using the script `scripts/MeshUtils.py`. Inside the script, one can modify the variables `filename` and `output_dir` to choose the name of the mesh and the directory where to save it. Then, the user can run the command `python scripts/MeshUtils.py --args` where `args` are the various parameters that can be modified to generate the meshes. The parameters are: 
- `--nmesh`: number of meshes to generate;
- `--nodes`: number of the interfaces inside the mesh;
- `--seed`: seed for the random generation of the meshes;
- `--hmax`: maximum length of interfaces;
- `--hmin`: minimum length of interfaces;
- `--wmax`: maximum spacing between interfaces;
- `--wmin`: minimum spacing between interfaces;
- `--lc`: characteristic length of the mesh;
- `--spacing`: boolean variable to choose if the interfaces are equally spaced or not (True = equally spaced).

#### Variational problem solver and data generation
The library is built to solve the following problems: 
- Heat diffusion
- Stokes problem
There is an abstract class `Solver` that the user can extend to solve other variational problems. We prepared two scripts to generate the data for the two problems mentioned above. The scripts are `scripts/HeatDatasetGen.py` and `scripts/StokesDatasetGen.py`. These two scripts solve the two problems and create the dataset. As for the section above, it is possible to modify the output directory which is stored in the variable `output_dir`, the mesh directory modifying the variable `mesh_dir` and the number of samples to generate which is stored in the variable `ngraphs`. In the same cell the user can modify the parameters of the problem. 
To run the script, the user can run the command `python scripts/HeatDatasetGen.py` or `python scripts/StokesDatasetGen.py`.
The class `MeshLoader` has a method `plot_mesh` that the user can use to see the mesh that is used to solve the problem.

#### GNN training




