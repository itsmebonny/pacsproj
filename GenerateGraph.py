import dgl 
import torch as th
import GenerateData as gd
import numpy as np

def generate_graph(point_data, points, edges_data, edges1, edges2):
    """
    Generate graph.

    Generate DGL graph out of data obtained from a vtp file.

    Arguments:
        point_data: dictionary containing point data (key: name, value: data)
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        add_boundary_edges (bool): decide whether to add boundary edges
        rcr_values: dictionary associating each branch id outlet to values
                    of RCR boundary conditions

    Returns:
        DGL graph
        dictionary containing indices of inlet and outlet nodes
        n x 3 numpy array of point coordinates
        n-dimensional array containin junction ids
        numpy array containing indices of source nodes for every edge
        numpy array containing indices of dist nodes for every edge
    """

    inlet = [0]
    outlets = [4] #find_outlets(edges1, edges2)

    indices = {"inlet": inlet, "outlets": outlets}

    graph = dgl.graph((edges1, edges2), idtype=th.int32)
    k = max(point_data['k'])
    area = max(edges_data['area'])
    length = max(edges_data['length'])
    graph.ndata["x"] = th.tensor(points, dtype=th.float32)
    graph.ndata["k"] = th.reshape(th.ones(graph.num_nodes(), dtype=th.float32) * k, (-1, 1, 1))
    graph.ndata["NodeId"] = th.tensor(point_data['NodeId'], dtype=th.float32)
    graph.ndata["inlet_mask"] = th.tensor(point_data['inlet_mask'], dtype=th.float32)
    graph.ndata["outlet_mask"] = th.tensor(point_data['outlet_mask'], dtype=th.float32)
    graph.edata["EdgeId"] = th.tensor(edges_data['edgeId'], dtype=th.float32)
    graph.edata["area"] = th.reshape(th.ones(graph.num_edges(), dtype=th.float32) * area, (-1, 1, 1))
    graph.edata["length"] = th.reshape(th.ones(graph.num_edges(), dtype=th.float32) * length, (-1, 1, 1))
    return graph


def add_field(graph, field, field_name, offset=0):
    """
    Add time-dependent fields to a DGL graph.

    Add time-dependent scalar fields as graph node features. The time-dependent
    fields are stored as n x 1 x m Pytorch tensors, where n is the number of
    graph nodes and m the number of timesteps.

    Arguments:
        graph: DGL graph
        field: dictionary with (key: timestep, value: field value)
        field_name (string): name of the field
        offset (int): number of timesteps to skip.
                      Default: 0 -> keep all timesteps
    """
    timesteps = [float(t) for t in field]
    timesteps.sort()
    dt = timesteps[1] - timesteps[0]
    T = timesteps[-1]

    # we use the third dimension for time
    field_t = th.zeros(len(list(field.values())[0]), 1, len(timesteps) - offset)

    times = [t for t in field]
    times.sort()
    times = times[offset:]

    for i, t in enumerate(times):
        f = th.tensor(field[t], dtype=th.float32)
        field_t[:, 0, i] = f
    if field_name == "flux":
        graph.ndata[field_name] = field_t
        graph.ndata["dt"] = th.reshape(
            th.ones(graph.num_nodes(), dtype=th.float32) * dt, (-1, 1, 1)
        )
        graph.ndata["T"] = th.reshape(
            th.ones(graph.num_nodes(), dtype=th.float32) * T, (-1, 1, 1)
        )
        
    elif field_name == "mean_temp":
        graph.edata[field_name] = field_t
        graph.edata["dt"] = th.reshape(
            th.ones(graph.num_edges(), dtype=th.float32) * dt, (-1, 1, 1)
        )
        graph.edata["T"] = th.reshape(
            th.ones(graph.num_edges(), dtype=th.float32) * T, (-1, 1, 1)
        )
def save_graph(graph, filename, output_dir = "data/graphs/"):
    """
    Save graph to disk.

    Save graph to disk as a DGL graph.

    Arguments:
        graph: DGL graph
        output_dir (string): path to output directory
    """
    
    dgl.save_graphs(output_dir+filename+".grph", graph)
    print("Graph saved to disk.")