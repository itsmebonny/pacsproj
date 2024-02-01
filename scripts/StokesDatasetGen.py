from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import GenerateData as gd
import MeshUtils as mutil
import gmsh
import GenerateGraph as gg
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Generate heat dataset')
parser.add_argument('--ngraphs', type=int, default=2)
parser.add_argument('--output_dir', type=str, default="data/graphs/")
parser.add_argument('--mesh_dir', type=str, default="data/mesh/")

args = parser.parse_args()

ngraphs = args.ngraphs  # number of graphs to generate
output_dir = args.output_dir   # output directory for graphs 
mesh_dir = args.mesh_dir # directory for mesh files
mesh_info = json.load(open(mesh_dir + 'mesh_info.json'))
mesh_name = mesh_info['mesh_name']
nmesh = mesh_info['nmesh']
nodes = mesh_info['interfaces']


np.random.seed(50)
for i in range(ngraphs):
    it = np.random.randint(0,20)
    mesh_load = mutil.MeshLoader(f"data/mesh01/RandomMesh_{it}")
    mesh = mesh_load.mesh
    bounds = mesh_load.bounds
    face = mesh_load.face
    mesh_load.update_tags(nodes=nodes)
    mesh_load.measure_definition()
    set_log_active(False)
    V = VectorElement("P", mesh_load.mesh.ufl_cell(), 2)
    Q = FiniteElement("P", mesh_load.mesh.ufl_cell(), 1)
    rho = 1*1e3
    mu = 4*1e-6
    U0 = 0.001
    L0 = 0.001
    inflow = Expression(("(-1.0/4.0*x[1]*x[1] + 1)", " 0.0 "), degree=2)
    dt = 0.1
    T = 5
    f = Constant((0.0, 0.0))
    k = round(np.random.uniform(0, 10),5)
    ns = gd.Stokes(mesh_load, V, Q, rho, mu, U0, L0, inflow, f, dt, T, k)
    ns.solve()
    data = gd.DataNS(ns,mesh_load)
    data.save_graph(output_dir = "data/graphs_stokes/")
data.generate_json(output_dir)