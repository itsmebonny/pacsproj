# Copyright 2023 Stanford University

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

import sys
import os
sys.path.append(os.getcwd() + '/gNN')
import torch as th
import graph1d.generate_normalized_graphs as gng
import graph1d.generate_dataset as dset
import tools.io_utils as io
import time
from network1d.meshgraphnet import MeshGraphNet
import json
import shutil
import numpy as np
import pathlib
from network1d.rollout import rollout
import tools.plot_tools as pt
import matplotlib.pyplot as plt


def evaluate_all_models(dataset, split_name, gnn_model, params, doplot = False):
    """
    Runs the rollout phase for all models and computes errors.

    Arguments:
        dataset: dictionary containing two keys, 'train' and 'test', with
                 the different datasets
        split_name (string): either 'train' or 'test'
        gnn_model: the GNN
        params: dictionary of parameters
        doplot: if True, the functions creates and saves one video per simulation. Default -> False
    Returns:
        array containing average flux normalized errors
        array containing average flux rate errors, 
        Average continuity loss

    """
    print('==========' + split_name + '==========')
    
    dataset = dataset[split_name]
    if doplot:
        pathlib.Path('results/' + split_name).mkdir(parents=True, exist_ok=True)

    N = len(dataset.graphs)
    total_timesteps = 0
    total_time = 0
    tot_errs_normalized = 0
    tot_errs = 0
    tot_cont_loss = 0
    for i in range(0,len(dataset.graphs)):
        print('model name = {}'.format(dataset.graph_names[i]))
        fdr = 'results/' + split_name + '/' + dataset.graph_names[i] + '/'
        pathlib.Path(fdr).mkdir(parents=True, exist_ok=True)
        r_features, errs_normalized, \
        errs, diff, elaps = rollout(gnn_model, params, dataset.graphs[i])
        total_time = total_time + elaps
        total_timesteps = total_timesteps + r_features.shape[2]
        print('Errors')
        print(errs)
        #if doplot:
            #plot_rollout(r_features, dataset.graphs[i], params, fdr)
        tot_errs_normalized = tot_errs_normalized + errs_normalized
        tot_errs = tot_errs + errs

    print('-------------------------------------')
    print('Global statistics')
    print('Errors')
    print(tot_errs / N)
    # print('Average time = {:.2f}'.format(total_time / N))
    # print('Average n timesteps = {:.2f}'.format(total_timesteps / N))

    return tot_errs_normalized/N, tot_errs/N, tot_cont_loss/N
        #    total_time / N, total_timesteps / N

def get_gnn_and_graphs(path, graphs_folder = 'graphs_train', 
                       data_location = 'data'):

    """
    Get GNN and list of graphs given the path to a saved model folder.

    Arguments:
        path (string): path to the GNN model folder. This should be the output
                       generated when launching the 'network1d/training.py'
                       script
        graphs_folder: name of folder containing graphs
        data_location (string): location of the 'gROM_data' folder. If None, 
                                we take the default location (which must be 
                                specified in data_location.txt).
                                Default -> None
    Returns:
        GNN model
        List of graphs
        Dictionary containing parameters
    """
    
    params = json.load(open(path + '/parameters.json'))

    gnn_model = MeshGraphNet(params)
    gnn_model.load_state_dict(th.load(path + '/trained_gnn.pms'))

    if data_location == None:
        data_location = io.data_location()
        print(data_location)
    target_features = ['flux']
    nodes_features = [
            'k',
            'interface_length']

    edges_features = ['area', 'length']
    features = {'nodes_features': nodes_features, 
                'edges_features': edges_features,
                'target_features': target_features}
    graphpath = os.path.join(data_location, graphs_folder)
    graphs, _  = gng.generate_normalized_graphs(graphpath,
                                                params['statistics']
                                                      ['normalization_type'],
                                                params['bc_type'],
                                                statistics = params 
                                                             ['statistics'], features=features)

    return gnn_model, graphs, params

def get_dataset_and_gnn(path, graphs_folder = 'graphs_train', data_location = 'data'):
    """
    Get datasets and GNN given the path to a saved model folder.

    Arguments:
        path (string): path to the GNN model folder. This should be the output
                       generated when launching the 'network1d/training.py'
                       script
        graphs_folder: name of folder containing graphs
        data_location (string): location of the 'gROM_data' folder. If None, 
                                we take the default location (which must be 
                                specified in data_location.txt).
                                Default -> None
    Returns:
        Dictionary containing train and test datasets
        GNN model
        Dictionary containing parameters

    """
    gnn_model, graphs, params = get_gnn_and_graphs(path,
                                                   graphs_folder,
                                                   data_location)

    dataset = dset.generate_dataset_from_params(graphs, params)
    return dataset, gnn_model, params

def plot_predictions(dataset, split_name, gnn_model, params, graph_idx=-1):
    """
    Plot predictions of a saved model for a given graph in the dataset.

    Arguments:
        dataset: dictionary containing two keys, 'train' and 'test', with
                 the different datasets
        split_name (string): either 'train' or 'test'
        gnn_model: the GNN
        params: dictionary of parameters
        graph_idx (int): index of graph to plot. If -1, a random graph is
                         chosen. 
                         Default -> -1
    Returns:
        Rollout error
    """ 

    dataset = dataset[split_name]
    N = len(dataset.graphs)
    if graph_idx == -1 or graph_idx >= N:
        graph_n = np.random.uniform(0,N, 1).astype(int)[0]
    else:
        graph_n = graph_idx
    k = dataset.graphs[graph_n].ndata['k'][0][0][0]
    rfc, errs_normalized, \
        errs, _, _ = rollout(gnn_model, params, dataset.graphs[graph_n])
    # r_features[:,0,:] = gng.invert_normalize(r_features[:,0,:],'flux', params['statistics'], 'labels')
    true_data =gng.invert_normalize(dataset.graphs[graph_n].ndata['nfeatures'][:,0,:],'flux', params['statistics'], 'labels')

    print('k = ', round(k.item(), 2))
    for node in range(1,4):
        plt.plot(rfc[node,0,:], label = 'pred', linewidth = 3)
        plt.plot(true_data[node,:], label = 'real', linewidth = 3, linestyle = '--')
        plt.legend(['prediction', 'real'])
        plt.xlabel('time steps')
        plt.ylabel('heat flux')
        plt.title(f'node {node}')
        plt.show()

    return errs

def test_new_graphs(path, graph_name, graphs_folder = 'graphs_new', data_location = 'data'):
    """
    Test a saved model on new graphs.

    Arguments:
        path (string): path to the GNN model folder
        graph_name (string): name of the graph file to test on
        graphs_folder: name of folder containing graphs
        data_location (string): location of the 'gROM_data' folder. If None, 
                                we take the default location (which must be 
                                specified in data_location.txt).
                                Default -> None

    Returns:
        Rollout error

    """
    
    gnn_params = json.load(open(path + '/parameters.json'))

    gnn_model = MeshGraphNet(gnn_params)
    gnn_model.load_state_dict(th.load(path + '/trained_gnn.pms'))

    if data_location == None:
        data_location = io.data_location()
        print(data_location)
    target_features = ['flux']
    nodes_features = [ 'k', 'interface_length']
    edges_features = ['area', 'length']
    features = {'nodes_features': nodes_features, 
                'edges_features': edges_features,
                'target_features': target_features}
    pathgraphs = os.path.join(data_location, graphs_folder)
    info = json.load(open(pathgraphs + '/dataset_info.json'))
    graphs, params2  = gng.generate_normalized_graphs(pathgraphs,
                                                gnn_params['statistics']['normalization_type'],
                                                gnn_params['bc_type'],
                                                {'dataset_info' : info, 'types_to_keep': []}, features=features)
    
    params2['nout'] = gnn_params['nout']
    
    rfc, _, \
            errs, __, ___ = rollout(gnn_model, params2, graphs[graph_name])
    # r_features[:,0,:] = gng.invert_normalize(r_features[:,0,:],'flux', params2['statistics'], 'labels')
    true_data =gng.invert_normalize(graphs[graph_name].ndata['nfeatures'][:,0,:],'flux', params2['statistics'], 'labels')
    
    for node in range(1,4):
        plt.plot(rfc[node,0,:], label = 'pred', linewidth = 3)
        plt.plot(true_data[node,:], label = 'real', linewidth = 3, linestyle = '--')
        plt.legend(['prediction', 'real'])
        plt.xlabel('time steps')
        plt.ylabel('heat flux')
        plt.title(f'node {node}')
        plt.show()

    return errs

"""
This function expects the location of a saved model folder as command line
argument. This is typically located in 'models/' after launching
'network1d/training.py'.
"""
if __name__ == '__main__':
    path = sys.argv[1]

    dataset, gnn_model, params = get_dataset_and_gnn(path)

    if os.path.exists('results'):
        shutil.rmtree('results')

    evaluate_all_models(dataset, 'train', gnn_model, params)
    evaluate_all_models(dataset, 'test', gnn_model, params, True)
    