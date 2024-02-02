# Solving PDEs using a Graph Neural Network

## PACS Project of Andrea Bonifacio and Sara Gazzoni

### Introduction

This repository contains the code for a Python-based library for solving PDEs using a Graph Neural Network. The code is based on the paper [Learning Reduced-Order Models for Cardiovascular Simulations with Graph Neural Networks](https://arxiv.org/abs/2303.07310). We built the library with the aim of creating a fully working pipeline from the generation of the data to the training of the model and the evaluation of the results.
We divided the code into three main folders:

- `scripts`: contains the scripts for the generation of the data;
- `gNN`: contains the code for the implementation of the GNN;
- `notebooks`: contains some example notebooks to test the library.
- `models`: contains some already trained models.

### Prerequisites

- Python <=3.10
- Required packages: FEniCS

### Installation

This installation procedure assumes that the user has already installed FEniCS in an Anaconda environment. If this is not the case, please refer to the [FEniCS installation guide](https://fenicsproject.org/download/archive/).
To install the library, please follow these steps:

1. Clone the repository 

```bash 
git clone https://github.com/itsmebonny/pacsproj.git
cd pacsproj
```

2. Activate the FEniCS environment
3. Install the required packages using `pip install -r requirements.txt`
4. Check if the installation was successful by running the script `python scripts/test_installation.py`
5. Store the `path/to/pacsproj/data` path in a file called `data_location.txt`, which needs to be saved in `gNN/tools/`

```bash
cd data
echo $(pwd) > ../gNN/tools/data_location.txt
cd ..
```

### Usage

The library is able to create meshes, solve variational problems, save them in a suitable format and train a GNN on the data. The user can choose which part of the pipeline to run. We will now describe each step separately. If the user wants to run the whole pipeline, please refer to the subsection **Notebooks**.

#### Mesh generation

The user can generate a mesh using the script `scripts/MeshUtils.py`. Inside the script, one can modify the variables `filename` (default: `mesh`) and `output_dir` (default: `data/mesh`) to choose the name of the mesh and the directory where to save it. Then, the user can run the command `python scripts/MeshUtils.py --args` where `args` are the various parameters that can be modified to generate the meshes. The parameters are:

- `--nmesh`: number of meshes to generate;
- `--interfaces`: number of the interfaces inside the mesh;
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

There is an abstract class `Solver` that the user can extend to solve other variational problems. We prepared two scripts to generate the data for the two problems mentioned above. The script is `scripts/HeatDatasetGen.py`. This script solves the problem and create the dataset. The Stokes problem solver is not implemented here as an executable script, in this case the user should refer to the `notebooks/StokesDatasetGen.ipynb` notebook.

To run the script, the user can run the command

```python
python scripts/HeatDatasetGen.py --args
```

Where `args` are the parameters of the problem:

- `--output_dir`: folder where to save the graphs (default: `data/graphs`);
- `--mesh_dir`: folder where the mesh are saved (default: `data/mesh_train`);
- `--ngraphs`: number of graphs to generate (default: 2);

The class `MeshLoader` has a method `plot_mesh` that the user can use to see the mesh that is used to solve the problem.

#### GNN training

For further information, please refer to the `README.md` file inside the `gNN` folder.

The script can be found in `gNN/network1d/training.py`. This script trains a GNN on the data generated before. To train a gNN, the user can run the command

```python
python gNN/network1d/training.py --args
```

Inside the `main` function, the user can modify the following variables:

- `graphs_folder`: path to the folder containing the graphs (default: `data/graphs_train`). Use the default setting to train the model on a large dataset already generated;
- `target_features`: features to predict;
- `nodes_features`: features of the nodes;
- `edges_features`: features of the edges;

The features chosen must correspond to the features that were used during the graph generation. For example, if the user wants to predict the heat flux, the target features must be `['flux']`.

The user can modify the parameters of the training in the `main` function by adding them as `--args`. The parameters are:

- `--latent_size_gnn`: size of the latent space of the GNN;
- `--latent_size_mlp`: size of the latent space of the MLP;
- `--process_iterations`: number of iterations of the GNN;
- `--number_hidden_layers_mlp`: number of hidden layers of the MLP;
- `--learning_rate`: learning rate of the optimizer;
- `--batch_size`: batch size;
- `--lr_decay`: learning rate decay;
- `--epochs`: number of epochs;
- `--weight_decay`: weight decay;
- `--rate_noise`: rate of noise to add to the target features;
- `--rate_noise_features`: rate of noise to add to the other features;
- `--stride`: stride of the time steps (how many time steps to consider);
- `--nout`: number of output features;
- `--bc_type`: type of boundary conditions;
- `--optimizer`: optimizer to use.


#### GNN testing

To test a GNN, the user can run the command

```python
python gNN/network1d/tester.py --args
```

where `args` are the parameters of the test. The parameters are:

- `--path`: path to the model folder (default: `models/trained_model`);
- `--graphs_folder`: name of folder containing graphs (default: `graphs_train`);
- `--data_location`: location of the "data" folder (default: `data`). In most of the cases, the default setting is fine.

If the user wants to test an already trained model, he can use the model that is already present in the `models` folder. In that case, simply run the command

```python
python gNN/network1d/tester.py
```

### Notebooks

To facilitate the use of the library, we prepared some notebooks that show how to use the library. The notebooks are:

- `notebooks/HeatDatasetGen.ipynb`: notebook that generates the data for the heat diffusion problem and creates the graphs;
- `notebooks/StokesDatasetGen.ipynb`: notebook that generates the data for the Stokes problem and creates the graphs;
- `notebooks/ModelTester.ipynb`: notebook that, given a model already trained, shows how to test it on the train and test geometries;

These notebooks can be used out of the box, without any modification by simply running all the cells. The dataset generation notebooks work the same as the scripts already presented, so the interested reader can refer to the previous sections for further information.
The notebook that solves Stokes is here just as a proof of concept, as it should need a few tweaks in order to solve the problem correctly.

The model tester notebook shows how to test a model on the train and test geometries. The user can modify the variable `path` to choose the model to test. The notebook will compute the errors for all the train and test geometries. It is also possible to test the network on a single geometry by modifying the variable `graphs_folder` to the path of the folder containing the graphs of the geometry to test and the variable `new_graph` to the name of the graph to test.

### Documentation

The documentation of the library is available in the `documentation.pdf` file. It is also possible to generate the documentation by running the command `doxygen Doxyfile` in the root folder of the repository.

### Authors

- Andrea Bonifacio
- Sara Gazzoni
