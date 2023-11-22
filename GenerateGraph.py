import dgl 
import torch as th
import GenerateData as gd
import numpy as np

def generate_graph(point_data, points, edges1, edges2):
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

    edges1_copy = edges1.copy()
    edges1 = np.concatenate((edges1, edges2))
    edges2 = np.concatenate((edges2, edges1_copy))

    graph = dgl.graph((edges1, edges2), idtype=th.int32)
    graph.ndata["k"] = th.tensor(point_data['k'], dtype=th.float32)
    graph.ndata["x"] = th.tensor(points, dtype=th.float32)
    graph.ndata["NodeId"] = th.tensor(point_data['NodeId'], dtype=th.float32)

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

    graph.ndata[field_name] = field_t
    graph.ndata["dt"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * dt, (-1, 1, 1)
    )
    graph.ndata["T"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * T, (-1, 1, 1)
    )
def save_graph(graph, filename):
    """
    Save graph to disk.

    Save graph to disk as a DGL graph.

    Arguments:
        graph: DGL graph
        output_dir (string): path to output directory
    """
    output_dir = "data/graphs/"
    dgl.save_graphs(output_dir+filename+".grph", graph)
    print("Graph saved to disk.")