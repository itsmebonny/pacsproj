
# %%
#add scripts to path
import sys
import os
cwd = os.getcwd()
sys.path.append(f'{cwd}/scripts')
# sys.path.append('../scripts')

import GenerateGraph
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import GenerateData as gd
import MeshUtils as mutil
import gmsh
import GenerateGraph as gg

import json

# %%
# run this cell to load a mesh and plot it

mesh_load = mutil.MeshLoader("../data/mesh_test/test_mesh_0")
mesh = mesh_load.mesh
bounds = mesh_load.bounds
face = mesh_load.face
mesh_load.plot_mesh()

# %%
ngraphs = 1 # number of graphs to generate
output_dir = "../data/graphs_test/" # output directory for graphs 
mesh_dir = "../data/mesh_test/" # directory for mesh files
mesh_info = json.load(open(mesh_dir + '/mesh_info.json'))
mesh_name = mesh_info['mesh_name']
nmesh = mesh_info['nmesh']
nodes = mesh_info['nodes']

f = Constant(0.0)
g = Expression('a*exp(-(t-b)*(t-b)/c/c)',degree=2,a=5,b=2.5,c=1,t=0)
u0 = Expression('0.0',degree=0)
T = 5
timesteps = 50
dt = T/timesteps
kmax = 100 # maximum thermal conductivity
kmin = 1   # minimum thermal conductivity

for i in range(ngraphs):
    imesh = np.random.randint(0,nmesh)
    mesh_load = mutil.MeshLoader(mesh_dir + mesh_name + f"_{imesh}")
    mesh = mesh_load.mesh
    bounds = mesh_load.bounds
    face = mesh_load.face
    mesh_load.update_tags(nodes=nodes)
    mesh_load.measure_definition()
    
    V = FunctionSpace(mesh_load.mesh,"DG",1)
    set_log_active(False)
    k = round(np.random.uniform(kmin, kmax),2)
    heat_gaussian = gd.Heat(mesh_load,V,k,f,u0,dt,T,g)
    heat_gaussian.solve()
    data = gd.DataHeat(heat_gaussian,mesh_load)
    data.save_graph(output_dir)


