
import dgl 
import torch as th
import GenerateData as gd
import numpy as np
import os
import json


def generate_graph(point_data, points, edges_data, edges1, edges2):
    """
    Generate DGL graph.

    Arguments:
        point_data: dictionary containing point data (key: name, value: data)
        points: n x 2 numpy array of point coordinates
        edges_data: dictionary containing edge data (key: name, value: data)
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge

    Returns:
        DGL graph
    """

    # inlet = [0]
    # outlets = [4] #find_outlets(edges1, edges2)

    # indices = {"inlet": inlet, "outlets": outlets}

    graph = dgl.graph((edges1, edges2), idtype=th.int32)
    # k = max(point_data['k'])
    # length = max(edges_data['length'])
    graph.ndata["x"] = th.tensor(points, dtype=th.float32)
    graph.ndata["k"] = th.reshape(th.ones(graph.num_nodes(), dtype=th.float32) * point_data['k'], (-1, 1, 1))
    # graph.ndata["k"] = th.reshape(th.tensor(point_data['k'], dtype=th.float32), (-1, 1, 1))
    graph.ndata["NodeId"] = th.tensor(point_data['NodeId'], dtype=th.float32)
    graph.ndata["inlet_mask"] = th.tensor(point_data['inlet_mask'], dtype=th.float32)
    graph.ndata["outlet_mask"] = th.tensor(point_data['outlet_mask'], dtype=th.float32)
    graph.ndata["interface_length"] = th.reshape(th.tensor(point_data['interface_length'], dtype=th.float32), (-1, 1, 1))
    graph.edata["EdgeId"] = th.tensor(edges_data['edgeId'], dtype=th.float32)
    graph.edata["area"] = th.reshape(th.tensor(edges_data['area'], dtype=th.float32), (-1, 1, 1))
    graph.edata["length"] = th.reshape(th.tensor(edges_data['length'], dtype=th.float32), (-1, 1, 1))
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
    # if field_name == "flux":
    graph.ndata[field_name] = field_t
    graph.ndata["dt"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * dt, (-1, 1, 1))
    graph.ndata["T"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * T, (-1, 1, 1))
        
def save_graph(graph, filename, output_dir = "data/graphs/"):
    """
    Save graph to disk as a DGL graph.

    Arguments:
        graph: DGL graph
        filename (string): name of the file
        output_dir (string): path to output directory
    """
    
    dgl.save_graphs(output_dir+filename+".grph", graph)
    print("Graph saved to disk.")

def generate_json(output_dir):

    input_directory = os.path.expanduser(output_dir)

    input_directory = os.path.realpath(input_directory)

    # Initialize an empty dictionary to store JSON objects for each file
    json_dict = dict()

    # Iterate through each .grph file in the directory
    for graph_file in os.listdir(input_directory):
        if graph_file.endswith(".grph"):
            # Extract the filename without extension
            filename_no_extension = os.path.splitext(graph_file)[0]

            # # Extract the mu parameter from the filename (assuming it is in the format k_11.3.grph)
            # mu_match = re.search(r'_(\d+(\.\d+)?)', filename_no_extension)
            # mu = mu_match.group(1) if mu_match else None

            # Create a dictionary for each .grph file
            json_dict[filename_no_extension] = {
                "model_type": "heat_eq"
            }

    # Save the dictionary as a JSON file
    json_file_path = os.path.join(input_directory, "dataset_info.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=2)

    print(f"Created {json_file_path}")