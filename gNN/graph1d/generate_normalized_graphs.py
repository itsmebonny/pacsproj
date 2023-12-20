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
sys.path.append(os.getcwd())
import tools.io_utils as io
import dgl
import torch as th
from tqdm import tqdm
from dgl.data.utils import load_graphs as lg
import numpy as np
import json
import random
import scipy

def normalize(field, field_name, statistics, norm_dict_label):
    """
    Normalize field.

    Normalize a field using statistics provided as input.

    Arguments:
        field: the field to normalize
        field_name (string): name of field
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'features' or 'labels'
    Returns:

        normalized field

    """
    if statistics['normalization_type'][norm_dict_label] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        if np.abs(delta) > 1e-5:
            field = (field - statistics[field_name]['min']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'][norm_dict_label] == 'normal':
        delta = statistics[field_name]['stdv']
        if np.abs(delta) > 1e-5 and not np.isnan(delta):
            field = (field - statistics[field_name]['mean']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'][norm_dict_label] == 'none':
        print('non normalizzo')
        pass
    else:
        raise Exception('Normalization type not implemented')
    return field

def invert_normalize(field, field_name, statistics, norm_dict_label):
    """
    Invert normalization over field.

    Invert normalization using statistics provided as input.

    Arguments:
        field: the field to normalize
        field_name (string): name of field
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'feature' or 'label'
    Returns:
        normalized field

    """
    if statistics['normalization_type'][norm_dict_label] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        field = statistics[field_name]['min'] + delta * field
    elif statistics['normalization_type'][norm_dict_label] == 'normal':
        delta = statistics[field_name]['stdv']
        if np.abs(delta) > 1e-5 and not np.isnan(delta):
            field = statistics[field_name]['mean'] + delta * field
        else:
            field = statistics[field_name]['mean']
    elif statistics['normalization_type'][norm_dict_label] == 'none':
        pass
    else:
        raise Exception('Normalization type not implemented')
    return field

def load_graphs(input_dir):
    """
    Load all graphs in directory.

    Arguments:
        input_dir (string): input directory path

    Returns:
        list of DGL graphs

    """
    print(input_dir)
    files = os.listdir(input_dir)
    random.seed(10)
    random.shuffle(files)

    graphs = {}
    for file in tqdm(files, desc = 'Loading graphs', colour='green'):
        if 'grph' in file:
            graphs[file] = lg(input_dir + file)[0][0]

    return graphs

def compute_statistics(graphs, fields, statistics):
    """
    Compute statistics on a list of graphs.

    The computet statistics are: min value, max value, mean, and standard
    deviation.

    Arguments:
        graphs: list of graphs
        fields: dictionary containing field names, divided into node and edge
                fields
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
    Returns:
        dictionary containining statistics (key: statistics name, value: value).
        New fields are appended to the input 'statistics' argument.

    """
    print('Compute statistics')
    for etype in fields:
            for field_name in fields[etype]:
                cur_statistics = {}
                minv = np.infty
                maxv = np.NINF
                Ns = []
                Ms = []
                means = []
                meansqs = []
                for graph_n in tqdm(graphs, desc = field_name, \
                                    colour='green'):
                    graph = graphs[graph_n]
                    if etype == 'node':
                        d = graph.ndata[field_name]
                    # elif etype == 'edge':
                    #     d = graph.edata[field_name]
                    # elif etype == 'outlet_node':
                    #     mask = graph.ndata['outlet_mask'].bool()
                    #     d = graph.ndata[field_name][mask]
                    # number of nodes
                    N = d.shape[0]
                    # number of times
                    M = d.shape[2]
                    minv = np.min([minv, th.min(d)])
                    maxv = np.max([maxv, th.max(d)])
                    mean = th.mean(d)
                    meansq = th.mean(d**2)

                    means.append(mean)
                    meansqs.append(meansq)
                    Ns.append(N)
                    Ms.append(M)

                ngraphs = len(graphs)
                MNs = 0
                for i in range(ngraphs):
                    MNs = MNs + Ms[i] * Ns[i]

                mean = 0
                meansq = 0
                for i in range(ngraphs):
                    coeff = Ms[i] * Ns[i] / MNs
                    mean = mean + coeff * means[i]
                    meansq = meansq + coeff * meansqs[i]

                cur_statistics['min'] = minv
                cur_statistics['max'] = maxv
                cur_statistics['mean'] = float(mean)
                cur_statistics['stdv'] = float(np.sqrt(meansq - mean**2))
                statistics[field_name] = cur_statistics

    graph_sts = {'nodes': []}

    for graph_n in graphs:
        graph = graphs[graph_n]
        graph_sts['nodes'].append(graph.ndata['x'].shape[0])

    for name in graph_sts:
        cur_statistics = {}

        cur_statistics['min'] = np.min(graph_sts[name])
        cur_statistics['max'] = np.max(graph_sts[name])
        cur_statistics['mean'] = np.mean(graph_sts[name])
        cur_statistics['stdv'] = np.std(graph_sts[name])

        statistics[name] = cur_statistics

    return statistics

def normalize_graphs(graphs, fields, statistics, norm_dict_label):
    """
    Normalize all graphs in a list.

    Arguments:
        graphs: list of graphs
        fields: dictionary containing field names, divided into node and edge
                fields
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'features' or 'labels'

    """
    print('Normalize graphs')
    for etype in fields:
            for field_name in fields[etype]:
                for graph_n in tqdm(graphs, desc = field_name,
                                    colour='green'):
                    graph = graphs[graph_n]
                    if etype == 'node':
                        d = graph.ndata[field_name]
                        graph.ndata[field_name] = normalize(d, field_name,
                                                            statistics,
                                                            norm_dict_label)
                    elif etype == 'edge':
                        d = graph.edata[field_name]
                        graph.edata[field_name] = normalize(d, field_name,
                                                            statistics,
                                                            norm_dict_label)
                    elif etype == 'outlet_node':
                        d = graph.ndata[field_name]
                        graph.ndata[field_name] = normalize(d, field_name,
                                                            statistics,
                                                            norm_dict_label)

def add_features(graphs, nodes_features = None, edges_features = None):
    """
    Add features to graphs.

    This function adds node and edge features to all graphs in
    the input list.

    Arguments:
        graphs: list of graphs.
        node_features: list of string of node features to include.
                       Default value -> None (keep all)
        edge_features: list of string of edge features to include.
                       Default value -> None (keep all)

    """
    if nodes_features == None:
        # pressure and flowrate are always included
        nodes_features = [
            'k',
            'T',
            'interface_length']

    if edges_features == None:
        edges_features = ['area', 'length']
        

    for graph_n in tqdm(graphs, desc = 'Add features', colour='green'):
        graph = graphs[graph_n]
        ntimes = graph.ndata['flux'].shape[2]

        cf = []

        def add_feature(tensor, desired_features, label):
            if label in desired_features:
                cf.append(tensor)

        # graph.ndata['dt'].repeat(1, 1, ntimes)
        # add_feature(graph.ndata['dt'].repeat(1, 1, ntimes), 
        #             nodes_features, 
        #             'dt')
        # print(cf)
        add_feature(graph.ndata['interface_length'].repeat(1, 1, ntimes), 
                    nodes_features, 
                    'interface_length')
        add_feature(graph.ndata['k'].repeat(1, 1, ntimes), 
                    nodes_features, 
                    'k')
        #print('flux',graph.ndata['flux'])
        # add_feature(graph.ndata['interface_length'].repeat(1, 1, ntimes), 
        #             nodes_features, 
        #             'interface_length')
        
        f = graph.ndata['flux'].clone()
        
        # add_feature(th.ones(f.shape[0],1,ntimes) * th.min(f), 
        #             nodes_features, 
        #             'dip')
        # add_feature(th.ones(f.shape[0],1,ntimes) * th.max(f), 
        #             nodes_features, 
        #             'sysp')
        # questo non serve stai aggiungendo loading 2 volte
        # add_feature(th.zeros(f.shape[0],1,ntimes), 
        #             nodes_features, 
        #             'loading')
        outmask = graph.ndata['outlet_mask'].bool()
        nnodes = outmask.shape[0]

        # r1 = th.zeros((nnodes,1,ntimes))
        # c = th.zeros((nnodes,1,ntimes))
        # r2 = th.zeros((nnodes,1,ntimes))
        # r1[outmask,0,:] = graph.ndata['resistance1'][outmask,0,:]
        # c[outmask,0,:] = graph.ndata['capacitance'][outmask,0,:]
        # r2[outmask,0,:] = graph.ndata['resistance2'][outmask,0,:]
        # add_feature(r1, nodes_features, 'resistance1')
        # add_feature(c, nodes_features, 'capacitance')
        # add_feature(r2, nodes_features, 'resistance2')

        cfeatures = th.cat(cf, axis = 1)

        if 'loading' in nodes_features:
            loading = graph.ndata['loading']
            graph.ndata['nfeatures'] = th.cat((f, cfeatures, loading), 
                                               axis = 1)
        else:
            graph.ndata['nfeatures'] = th.cat((f, cfeatures), axis = 1)

        cf = []
        add_feature(graph.edata['area'], edges_features, 'area')
        add_feature(graph.edata['length'], edges_features, 'length')
        graph.edata['efeatures'] = th.cat(cf, axis = 1)
def add_deltas(graphs):
    """
    Compute pressure and flowrate increments.

    The increments are computed from time t to t+1 and stored as node features
    labelled 'dp' and 'df'

    Arguments:
        graphs: list of graphs

    """
    for graph_n in tqdm(graphs, desc = 'Add deltas', colour='green'):
        graph = graphs[graph_n]

        # graph.ndata['dp'] = graph.ndata['flux'][:,:,1:] - \
        #                     graph.ndata['flux'][:,:,:-1]

        graph.ndata['df'] = graph.ndata['flux'][:,:,1:] - \
                            graph.ndata['flux'][:,:,:-1]

def save_graphs(graphs, output_dir):
    """
    Save all graphs contained in a list to file.

    Arguments:
        graphs: list of graphs
        output_dir: path of output directory

    """
    for graph_name in tqdm(graphs, desc = 'Saving graphs', colour='green'):
        dgl.save_graphs(output_dir + graph_name, graphs[graph_name])

def save_parameters(params, output_dir):
    """
    Save normalization parameters to file .

    Arguments:
        params: dictionary containing nornalization parameters
        output_dir: path of output directory

    """
    with open(output_dir + '/parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4)

def restrict_graphs(graphs, types, types_to_keep):
    """
    Restrict the list of graphs to the types that we are interested in.

    Arguments:
        graphs: list of graphs
        types: dictionary with all types (key: model name, value: type)
        types_to_keep: list with types to keep:
    
    Returns:
        restricted list of DGL graphs

    """
    selected_graphs = {}
    for graph in graphs:
        id = graph.replace('.grph','').split('.')
        if types[id[0] + '.' + id[1]]['model_type'] in types_to_keep:
            selected_graphs[graph] = graphs[graph]
    graphs = selected_graphs
    return graphs

def generate_normalized_graphs(input_dir, norm_type, bc_type,
                               types_to_keep = None,
                               n_graphs_to_keep = -1,
                               statistics = None,
                               features = None):
    """
    Generate normalized graphs.

    Arguments:
        input_dir: path to input directory
        norm_type: dictionary with keys: features/labels,
                   values: min_max/normal
        bc_type: boundary condition type. Currently supported: full_dirichlet
                 (pressure and flowrate imposed at boundary nodes) and
                 realistic dirichlet (flowrate imposed at inlet, pressure
                 imposed at outlets)
        types_to_keep: dictionary containing all graphs types, and list
                       containing types we want to keep. If None, keep all
                       types. Default value -> None.
        n_graphs_to_keep: number of graphs to keep. If -1, keep all graphs.
                          Default value -> -1.
        features: dictionary of features to include in graphs
                  Default value -> None (include all)

    Return:
        List of normalized graphs
        Dictionary of parameters

    """
    fields_to_normalize = {'node': ['flux'], 'edge': [], 'outlet_node': []}
    docompute_statistics = True
    if statistics != None:
        docompute_statistics = False

    if docompute_statistics:
        statistics = {'normalization_type': norm_type}
    graphs = load_graphs(input_dir)

    # if types_to_keep != None or types_to_keep['types_to_keep'] != None:
    #     graphs = restrict_graphs(graphs, types_to_keep['dataset_info'], 
    #                              types_to_keep['types_to_keep'])

    if n_graphs_to_keep != -1:
        graphs_ = {}
        graphs_names = []
        count = 0
        for key, value in graphs.items():
            # if count == n_graphs_to_keep:
            #     break
            # graph_name = key.replace('.graph','')
            # graph_name = graph_name[0:graph_name.find('.')]
            graph_name = ".".join(key.split(".", 2)[:2])
            if graph_name not in graphs_names and count != n_graphs_to_keep:
                graphs_names.append(graph_name)
                graphs_[key] = value
                count = count + 1
            elif graph_name in graphs_names:
                graphs_[key] = value
        graphs = graphs_

    if docompute_statistics:
        compute_statistics(graphs, fields_to_normalize, statistics)

    print(graphs['k_68.58.grph'].ndata['flux'])
    print(statistics)
    normalize_graphs(graphs, fields_to_normalize, statistics, 'features')
    print(graphs['k_68.58.grph'].ndata['flux'])
    add_deltas(graphs)
    if docompute_statistics:
        compute_statistics(graphs, {'node' : ['df']}, statistics)
    normalize_graphs(graphs, {'node' : ['df']}, statistics, 'labels')
    params = {'bc_type': bc_type}
    params['statistics'] = statistics
    if features == None:
        add_features(graphs)
    else:
        add_features(graphs, 
                     features['nodes_features'], 
                     features['edges_features'])

    return graphs, params