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
import argparse
import time
import tools.io_utils as io
import numpy as np
import torch.distributed as dist
import graph1d.generate_dataset as dset
import torch as th
from datetime import datetime
from network1d.meshgraphnet import MeshGraphNet
import pathlib
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from network1d.rollout import rollout
from network1d.rollout import perform_timestep
import json
import tools.plot_tools as ptools
import pickle
import signal
import graph1d.generate_normalized_graphs as gng
import random
import copy

class SignalHandler(object):
    """
    Signal handler. 

    We use the signal handler to listen for ctrl-c during training. When 
    triggered, we asked the user if training should be stopped and, if so,
    the current model is saved.

    """
    def __init__(self):
        """
        Constructor

        """
        self.should_exit = False
    def handle(self, sig, frm):
        """
        Handle the signals

        Arguments:
            sig: unused argument from overwritten method
            frm: unused argument from overwritten method

        """
        res = input("Do you want to exit training?" + \
                        "Model and statistics will be saved (y/n)")
        if res == "y":
            self.should_exit = True
        else:
            pass

def mse(input, target, mask = None):
    """
    Mean square error.

    This is defined as the ((input - target)**2).mean()

    Arguments:
        input: first tensor
        target: second tensor (ideally, the result we are trying to match)
        mask: tensor of 1 and 0 with same size as input and target. If not 
              None, selects only components for which it equals 1. 
              Default -> None
    
    Returns:
        The mean square error

    """
    if mask == None:
        return ((input - target) ** 2).mean()
    return (mask * (input - target) ** 2).mean() 

def mae(input, target, mask = None):
    """
    Mean average error.

    This is defined as the (abs(input - target)).mean()

    Arguments:
        input: first tensor
        target: second tensor (ideally, the result we are trying to match)
        mask: tensor of 1 and 0 with same size as input and target. If not 
              None, selects only components for which it equals 1. 
              Default -> None
    
    Returns:
        The mean average error

    """
    if mask == None:
        return (th.abs(input - target)).mean()
    return (mask * (th.abs(input - target))).mean()

def evaluate_model(gnn_model, train_dataloader, test_dataloader, optimizer,     
                   print_progress, params):
    """
    Evaluates and train a GNN model.

    Arguments:
        gnn_model: the GNN to train
        train_dataloader: dataloader containing train graphs
        test_dataloader: dataloader containing test graphs
        optimizer: a Pytorch optimizer
        print_progress: if True, prints the progress bar during epochs.
        params: dictionary of parameters
    Returns:
        Dictionary containing train results
        Dictionary containing test results
        Elapsed time in seconds

    """
    def loop_over(dataloader, label, c_optimizer = None):
        """
        Performs one epoch by looping over all batches in the dataloader.

        Arguments:
            dataloader: a DGL dataloader
            label (string): either 'test' or 'train' (used for progress bar)
            c_optimizer: pytorch optimizer. If not none, performs optimization
                         of the GNN. Default -> None.
        
        Returns:
            Dictionary containing values of loss, train metric, and continuity
                loss

        """
        global_loss = 0
        global_metric = 0
        count = 0

        def iteration(batched_graph, c_optimizer):
            """
            Performs one train iteration

            Arguments:
                batched_graph: a DGL batched graph
                c_optimizer: pytorch optimizer. If not None, performs
                             optimization of the GNN. Default -> None.
            
            Returns:
                Loss value
                Metric value
                Continuity loss value

            """
            batched_graph_c = copy.deepcopy(batched_graph)
            ns = batched_graph_c.ndata['next_steps']
            # print('batched_graph', batched_graph.ndata['nfeatures'])
            # print('ns', ns)

            loss_v = 0
            metric_v = 0
            mask = th.ones(ns[:,:,0:1].shape)
            inmask = batched_graph.ndata['inlet_mask'].bool()
            outmask = batched_graph.ndata['outlet_mask'].bool()

            bccoeff = 0
            mask[inmask,0] = mask[inmask,0] * bccoeff

            # flow rate is known
            # mask[outmask,0] = mask[outmask,0] * bccoeff
            #mask[outmask,1] = mask[outmask,1] * bccoeff
            for istride in range(params['stride']):
                # print('TRAINING')
                nf = perform_timestep(gnn_model, params, batched_graph_c, ns, 
                                      istride)


                batched_graph_c.ndata['nfeatures'][:,0:1] = nf
                
                # print(istride)
                # print(nf)

                # we follow https://arxiv.org/pdf/2206.07680.pdf for the
                # coefficient
                coeff = 0.5
                if istride == 0:
                    coeff = 1  

                #c_loss = th.tensor(0.0)
                    
                # print('real', ns[:,:,istride])
                # print('pred', nf)
                #print('mask', mask)
                loss_v = loss_v + coeff * mse(nf, ns[:,:,istride], mask=None)
                metric_v = metric_v + coeff * mae(nf, ns[:,:,istride], mask=None)

            if c_optimizer != None:
                optimizer.zero_grad()
                loss_v.backward()
                optimizer.step()
            
            return loss_v.detach().numpy(), metric_v.detach().numpy()


        if not print_progress:
            for batched_graph in dataloader:
                loss_v, metric_v = iteration(batched_graph, c_optimizer)
                global_loss = global_loss + loss_v
                global_metric = global_metric + metric_v
                count = count + 1
        else:
            for batched_graph in tqdm(dataloader, 
                                    desc = label, colour='green'):
                loss_v, metric_v = iteration(batched_graph, c_optimizer)
                global_loss = global_loss + loss_v
                global_metric = global_metric + metric_v
                count = count + 1
                # if count == 1:
                    #print(batched_graph.ndata['nfeatures'][:,:])

        return {'loss': global_loss / count, 
                'metric': global_metric / count}

    gnn_model.train()
    start = time.time()
    train_results = loop_over(train_dataloader, 'train', optimizer)
    test_results = loop_over(test_dataloader, 'test ')
    end = time.time()

    return train_results, test_results, end - start

def compute_rollout_errors(gnn_model, params, dataset, idxs_train, idxs_test):
    """
    Compute rollout errors

    Arguments:
        gnn_model: the GNN
        params: dictionary of parameters
        dataset: the dataset over which the errors are computed
        idxs_train: indices of graphs to use to evaluating the training
        idxs_test: indices of graphs to use to evaluate the test
    
    Returns:
        2D array containing the error for pressure and flow rate (train)
        2D array containing the error for pressure and flow rate (test)

    """
    train_errs = np.zeros(2)
    for idx in idxs_train:
        _, cur_train_errs, _, _, _ = rollout(gnn_model, params, 
                                             dataset['train'].graphs[idx])
        train_errs = cur_train_errs + train_errs
    
    train_errs = train_errs / len(idxs_train)

    test_errs = np.zeros(2)
    for idx in idxs_test:
        _, cur_test_errs, _, _, _ = rollout(gnn_model, params, 
                                            dataset['test'].graphs[idx])
        test_errs = cur_test_errs + test_errs

    test_errs = test_errs / (len(idxs_test))

    return train_errs, test_errs

def train_gnn_model(gnn_model, dataset, params, parallel, doprint = True):
    """
    Train GNN model

    Arguments:
        gnn_model: the GNN
        params: dictionary of parameters
        dataset: the dataset over which the errors are computed
        parallel (bool): must be set to True if we are using distributed 
                         training and false otherwise
        print (bool): if True, prints metrics during training. Default -> True.
    
    Returns:
        The trained GNN model
        Dictionary containing array with the training history of various 
            quantities. Keys: train_loss, train_metric, train_rollout, 
            train_cont, test_loss, test_metric, test_rollout, test_cont

    """
    batch_size = params['batch_size']
    rank = 0
    if parallel:
        rank = dist.get_rank()
        if rank != 0:
            doprint == False
        train_sampler = DistributedSampler(dataset['train'], 
                                           num_replicas = dist.get_world_size(),
                                           rank = rank)
        
        test_sampler = DistributedSampler(dataset['test'],      
                                          num_replicas = dist.get_world_size(), 
                                          rank = rank)
        # get smaller batch size to preserve scaling when parallel
        batch_size = int(np.floor(batch_size / dist.get_world_size()))
    else: 
        num_train = int(len(dataset['train']))
        train_sampler = SubsetRandomSampler(th.arange(num_train))
        num_test = int(len(dataset['test']))
        test_sampler = SubsetRandomSampler(th.arange(num_test))

    #print('index_map',dataset['train'].graphs[0].ndata['nfeatures'])
    
    train_dataloader = GraphDataLoader(dataset['train'], 
                                       sampler = train_sampler,
                                       batch_size = batch_size,
                                       drop_last = False)
    
    #print(train_dataloader.sampler)
    
    test_dataloader = GraphDataLoader(dataset['test'], 
                                      sampler = test_sampler,
                                      batch_size = batch_size,
                                      drop_last = False)

    lr = params['learning_rate']
    if parallel:
        print("my rank = %d, world = %d, train_dataloader_len = %d." \
        % (dist.get_rank(), dist.get_world_size(), len(train_dataloader)),\
        flush=True)
        # lr = params['learning_rate'] / dist.get_world_size()
    
    optimizer = th.optim.Adam(gnn_model.parameters(), lr,
                              weight_decay = params['weight_decay'])

    nepochs = params['nepochs']

    eta_min = lr * params['lr_decay']
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = nepochs,
                                                        eta_min = eta_min)

    countp = 0

    # sample train and test graphs for rollout
    np.random.seed(10)
    ngraphs = np.min((10, len(dataset['test'].graphs)))
    idxs_train = random.sample(range(len(dataset['train'].graphs)), ngraphs)
    idxs_test = random.sample(range(len(dataset['test'].graphs)), ngraphs)
    s = SignalHandler()
    history = {}
    history['train_loss'] = [[], []]
    history['train_metric'] = [[], []]
    history['train_rollout'] = [[], []]
    history['test_loss'] = [[], []]
    history['test_metric'] = [[], []]
    history['test_rollout'] = [[], []]  
    for epoch in range(nepochs):
        if doprint:
            print('================{}================'.format(epoch))

        signal.signal(signal.SIGINT, s.handle)
        train_results, test_results, elapsed = evaluate_model(gnn_model,
                                                              train_dataloader,
                                                              test_dataloader,
                                                              optimizer,
                                                              rank == 0,
                                                              params)


        msg = 'epoch {:.0f}, time = {:.2f} s \n'.format(epoch, elapsed)
        msg = msg + '\ttrain:\tloss = {:.2e}\t'.format(train_results['loss'])
        msg = msg + 'mae = {:.2e}\t'.format(train_results['metric'])
        msg = msg + '\ttest:\tloss = {:.2e}\t'.format(test_results['loss'])
        msg = msg + 'mae = {:.2e}\t'.format(test_results['metric'])

        if doprint:
            print("", flush = True)
            print(msg, flush = True)

        history['train_loss'][0].append(epoch)
        history['train_loss'][1].append(float(train_results['loss']))
        history['train_metric'][0].append(epoch)
        history['train_metric'][1].append(float(train_results['metric']))

        history['test_loss'][0].append(epoch)
        history['test_loss'][1].append(float(test_results['loss']))
        history['test_metric'][0].append(epoch)
        history['test_metric'][1].append(float(test_results['metric']))

        if rank == 0:
            if (epoch + 1) == 2**countp or epoch == (nepochs - 1):
                e_train, e_test = compute_rollout_errors(gnn_model, 
                                                        params, dataset, 
                                                        idxs_train, idxs_test)

                history['train_rollout'][0].append(epoch)
                history['train_rollout'][1].append(float(np.mean(e_train)))
                history['test_rollout'][0].append(epoch)
                history['test_rollout'][1].append(float(np.mean(e_test)))
                countp = countp + 1

        if doprint:
            msg = 'Rollout: {:.0f}\t'.format(epoch)
            print(msg, flush = True)
            print(history['train_rollout'][1])
            print(history['test_rollout'][1])

        scheduler.step()

        if s.should_exit:
            return gnn_model, history            

    return gnn_model, history

def launch_training(dataset, params, parallel, out_dir = 'models/'):
    """
    Launch training

    Arguments:
        dataset: dataset containing the graphs
        params: dictionary of parameters
        parallel (bool): must be set to True if we are using distributed 
                         training and false otherwise
        out_dir (bool): path of folder where data should be saved. 
                        Default-> 'models/'
    
    Returns:
        The trained GNN model

    """
    now = datetime.now()
    folder = out_dir + now.strftime("%d.%m.%Y_%H.%M.%S")

    gnn_model = MeshGraphNet(params)
    def save_model(filename):
        if parallel:
            th.save(gnn_model.module.state_dict(), folder + '/' + filename)
        else:
            th.save(gnn_model.state_dict(),  folder + '/' + filename)

    def default(obj):
        if isinstance(obj, th.Tensor):
            return default(obj.detach().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        print(obj)
        raise TypeError('Not serializable')

    save_data = True
    if parallel:
        gnn_model = th.nn.parallel.DistributedDataParallel(gnn_model)
        save_data = (dist.get_rank() == 0)

    if save_data:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        save_model('initial_gnn.pms')

    gnn_model, history = train_gnn_model(gnn_model, dataset, params, 
                                         parallel, save_data)

    if save_data:
        save_model('trained_gnn.pms')

    if save_data:
        final_rollout = history['test_rollout'][1][-1]
        print('Final rollout error on test = ' + str(final_rollout))

    if save_data:
        ptools.plot_history(history['train_loss'],
                        history['test_loss'],
                        'loss', folder)
        ptools.plot_history(history['train_metric'],
                        history['test_metric'],
                        'metric', folder)
        ptools.plot_history(history['train_rollout'],
                            history['test_rollout'],
                            'rollout', folder)

        with open(folder + '/history.bnr', 'wb') as outfile:
            pickle.dump(history, outfile)
        
        with open(folder + '/parameters.json', 'w') as outfile:
            json.dump(params, outfile, default=default, indent=4)

    return gnn_model


def parse_command_line_arguments():
    """
    Parse command line arguments.

    Returns:
        Data structure containing all the arguments
    """

    # parse arguments from command line
    parser = argparse.ArgumentParser(description='Graph Reduced Order Models')

    parser.add_argument('--bs', help='batch size', type=int, default=32)
    parser.add_argument('--epochs', help='total number of epochs', type=int,
                        default=10000)
    parser.add_argument('--lr_decay', help='learning rate decay', type=float,
                        default=0.001)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('--rate_noise', help='rate noise', type=float,
                        default=5)
    parser.add_argument('--rate_noise_features', help='rate noise features', 
                        type=float, default=1e-5)
    parser.add_argument('--weight_decay', help='l2 regularization', 
                        type=float, default=1e-5)
    parser.add_argument('--ls_gnn', help='latent size gnn', type=int,
                        default=8)
    parser.add_argument('--ls_mlp', help='latent size mlps', type=int,
                        default=16)
    parser.add_argument('--process_iterations', help='gnn layers', type=int,
                        default=3)
    parser.add_argument('--hl_mlp', help='hidden layers mlps', type=int,
                        default=2)
    parser.add_argument('--label_norm', help='0: min_max, 1: normal, 2: none',
                        type=int, default=1)
    parser.add_argument('--stride', help='stride for multistep training',
                        type=int, default=5
                        )
    parser.add_argument('--bcs_gnn', help='path to graph for bcs',
                        type=str, default='models_bcs/31.10.2022_01.35.31')
    args = parser.parse_args()

    # we create a dictionary with all the parameters
    t_params = {'latent_size_gnn': args.ls_gnn,
                'latent_size_mlp': args.ls_mlp,
                'process_iterations': args.process_iterations,
                'number_hidden_layers_mlp': args.hl_mlp,
                'learning_rate': args.lr,
                'batch_size': args.bs,
                'lr_decay': args.lr_decay,
                'nepochs': args.epochs,
                'weight_decay': args.weight_decay,
                'rate_noise': args.rate_noise,
                'rate_noise_features': args.rate_noise_features,
                'stride': args.stride,
                'bcs_gnn': args.bcs_gnn}

    return t_params, args

def get_graphs_params(label_normalization, types_to_keep, 
                      n_graphs_to_keep = -1,
                      graphs_folder = 'graphs_long/',
                      data_location = io.data_location(),
                      features = None):
    """
    Get normalized graphs and associated parameters

    Arguments:
        label_normalization: type of label normalization ('normal' or 'min_max')
        types_to_keep: list with strings of types of graphs we want to restrict
                       training to.
        n_graphs_to_keep: number of unique graphs to use for training.
                          Default: -1 (keep all)None
        graphs_folder: name of folder containing graphs
        data_location: path of folder containing 'graphs/' folder
        features: dictionary with node and edge features to include
                        Default value -> None (keep all)
    Returns:
        Graphs
        Dictionary of parameters
        Dictionary containing dataset_info
    """

    input_dir = data_location + graphs_folder
    norm_type = {'features': 'normal', 'labels': label_normalization}
    info = json.load(open(input_dir + '/dataset_info.json'))

    t2k = types_to_keep
    ngtk = n_graphs_to_keep
    graphs, params  = gng.generate_normalized_graphs(input_dir, norm_type, 
                                                    'physiological',
                                                    {'dataset_info' : info,
                                                    'types_to_keep': t2k},
                                                    n_graphs_to_keep=ngtk,
                                                    features=features)

    return graphs, params, info

def training(parallel, rank = 0, graphs_folder = 'graphs_long/', 
             data_location = io.data_location(),
             types_to_keep = None,
             features = None):
    """
    Run GNN training

    Arguments:
        parallel (bool): True if we are using distributed training
        rank: rank of the processor
        graphs_folder: name of folder containing graphs
        data_location: path to folder containing graph_folder
        types_to_keep: list of graph types to keep
                       Default value -> None (keep all)
        features: dictionary with node and edge features to include
                  Default value -> None (keep all)

    """
    t_params, args = parse_command_line_arguments()

    if args.label_norm == 0:
        label_normalization = 'min_max'
    elif args.label_norm == 1:
        label_normalization = 'normal'
    elif args.label_norm == 2:
        label_normalization = 'none'
    
    graphs, params, info = get_graphs_params(label_normalization,
                                             types_to_keep, -1,
                                             graphs_folder, data_location,
                                             features)
    graph = graphs[list(graphs)[0]]


    infeat_nodes = graph.ndata['nfeatures'].shape[1] + 1
    infeat_edges = graph.edata['efeatures'].shape[1]
    nout = 1

    t_params['infeat_nodes'] = infeat_nodes
    t_params['infeat_edges'] = infeat_edges
    t_params['out_size'] = nout

    if features != None and features['nodes_features'] != None:
        params['node_features'] = nodes_features


    params.update(t_params)

    datasets = dset.generate_dataset(graphs, params, info, nchunks = 1)

    start = time.time()
    for _, dataset in enumerate(datasets):
        # print('train', dataset['train'].graph_names)
        # print('test',dataset['test'].graph_names)
        dataset['test'].graph_names.sort()
        params['train_split'] = dataset['train'].graph_names
        params['test_split'] = dataset['test'].graph_names
        _ = launch_training(dataset, params, parallel)
    end = time.time()
    elapsed_time = end - start

    if rank == 0:
        print('Training time = ' + str(elapsed_time))

"""
The main function launches the training by reading the graphs contained into 
data_location/graphs. The trained model is saved by default in a folder called 
'models'.
"""
if __name__ == "__main__":
    rank = 0
    try:
        parallel = True
        dist.init_process_group(backend='mpi')
        rank = dist.get_rank()
        print("my rank = %d, world = %d." % (rank, dist.get_world_size()), flush=True)
        th.backends.cudnn.enabled = False
    except RuntimeError:
        parallel = False
        print("MPI not supported. Running serially.")

    # 'synthetic' refers to the bcs, not the geometry
    types_to_keep = []
    nodes_features = [
            'flux', 
            'k',
            'interface_length']

    edges_features = ['area', 'length']

    features = {'nodes_features': nodes_features, 
                'edges_features': edges_features}
    training(parallel, rank, 
             graphs_folder = 'graphs_long/', 
             types_to_keep = types_to_keep, 
             features = features)
    sys.exit()
