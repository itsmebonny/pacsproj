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

    #bif_id = point_data["BifurcationId"]

    # we manually make the graph bidirected in order to have the relative
    # position of nodes make sense (xj - xi = - (xi - xj)). Otherwise, each edge
    # will have a single feature
    edges1_copy = edges1.copy()
    edges1 = np.concatenate((edges1, edges2))
    edges2 = np.concatenate((edges2, edges1_copy))

    #rel_position, distance = generate_edge_features(points, edges1, edges2)

    #types, inlet_mask, outlet_mask = generate_types(bif_id, indices)

    # npoints = points.shape[0]
    # rcr = np.zeros((npoints, 3))
    # for ipoint in range(npoints):
    #     if outlet_mask[ipoint] == 1:
    #         if rcr_values["bc_type"] == "RCR":
    #             id = find_closest_point_in_rcr_file(points[ipoint])
    #             rcr[ipoint, :] = rcr_values[id]["RCR"]
    #         elif rcr_values["bc_type"] == "R":
    #             id = find_closest_point_in_rcr_file(points[ipoint])
    #             rcr[ipoint, 0] = rcr_values[id]["RP"][0]
    #         else:
    #             raise ValueError("Unknown type of boundary conditions!")
    # etypes = [0] * edges1.size
    # # we set etype to 1 if either of the nodes is a junction
    # for iedge in range(edges1.size):
    #     if types[edges1[iedge], 1] == 1 or types[edges2[iedge], 1] == 1:
    #         etypes[iedge] = 1

    # if add_boundary_edges:
    #     bedges1, bedges2, brel_position, bdistance, btypes = generate_boundary_edges(
    #         points, indices, edges1, edges2
    #     )
    #     edges1 = np.concatenate((edges1, bedges1))
    #     edges2 = np.concatenate((edges2, bedges2))
    #     etypes = etypes + btypes
    #     distance = np.concatenate((distance, bdistance))
    #     rel_position = np.concatenate((rel_position, brel_position), axis=0)

    # jmasks = {}
    # jmasks["inlets"] = np.zeros(bif_id.size)
    # jmasks["all"] = np.zeros(bif_id.size)

    graph = dgl.graph((edges1, edges2), idtype=th.int32)

    graph.ndata["x"] = th.tensor(points, dtype=th.float32)
    graph.ndata["flux"] = th.tensor(point_data['flux'], dtype=th.float32)
    graph.ndata["NodeId"] = th.tensor(point_data['NodeId'], dtype=th.float32)
    #tangent = th.tensor(point_data["tangent"], dtype=th.float32)
    #graph.ndata["tangent"] = th.unsqueeze(tangent, 2)
    # graph.ndata["area"] = th.reshape(th.tensor(area, dtype=th.float32), (-1, 1, 1))

    # graph.ndata["type"] = th.unsqueeze(types, 2)
    # graph.ndata["inlet_mask"] = th.tensor(inlet_mask, dtype=th.int8)
    # graph.ndata["outlet_mask"] = th.tensor(outlet_mask, dtype=th.int8)
    # graph.ndata["jun_inlet_mask"] = th.tensor(jmasks["inlets"], dtype=th.int8)
    # graph.ndata["jun_mask"] = th.tensor(jmasks["all"], dtype=th.int8)
    # graph.ndata["branch_mask"] = th.tensor(
    #     types[:, 0].detach().numpy() == 1, dtype=th.int8
    # )
    # graph.ndata["branch_id"] = th.tensor(point_data["BranchId"], dtype=th.int8)

    # graph.ndata["resistance1"] = th.reshape(
    #     th.tensor(rcr[:, 0], dtype=th.float32), (-1, 1, 1)
    # )
    # graph.ndata["capacitance"] = th.reshape(
    #     th.tensor(rcr[:, 1], dtype=th.float32), (-1, 1, 1)
    # )
    # graph.ndata["resistance2"] = th.reshape(
    #     th.tensor(rcr[:, 2], dtype=th.float32), (-1, 1, 1)
    # )

    # graph.edata["rel_position"] = th.unsqueeze(
    #     th.tensor(rel_position, dtype=th.float32), 2
    # )
    # graph.edata["distance"] = th.reshape(
    #     th.tensor(distance, dtype=th.float32), (-1, 1, 1)
    # )
    # etypes = th.nn.functional.one_hot(th.tensor(etypes), num_classes=5)
    # graph.edata["type"] = th.unsqueeze(etypes, 2)

    return graph