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

import os
import numpy as np
import xml.etree.cElementTree as ET
import meshio

def data_location():
    """
    Return data location, which by default needs to be stored in 
    data_location.txt
  
    Returns:
        Location of the data (string)

    """
    if not os.path.exists(os.getcwd() + '/data_location.txt'):
        return '/home/bonni/Documents/pacs/pacsproj/data/'
    f = open(os.getcwd() + '/data_location.txt', 'r')
    location = f.readline().strip()
    f.close()
    return location + '/'

def create_directory(fdr_name):
    """
    Create a directory.
    
    Arguments:
        fdr_name (string): Name of the directory
  
    """
    try:
        os.mkdir(fdr_name)
    except OSError:
        print('Directory ' + fdr_name + ' exists')

def collect_arrays(celldata, components = None):
    """  
    Collect arrays from a cell data or point data object.
  
    Arguments:
        celldata: Input data
        components (int): Number of array components to keep. 
                          Default: None -> keep all
    Returns:
        A dictionary of arrays (key: array name, value: numpy array)
  
    """
    res = {}
    for i in range(celldata.GetNumberOfArrays()):
        name = celldata.GetArrayName(i)
        data = celldata.GetArray(i)
        if components == None:
            res[name] = v2n(data).astype(np.float32)
        else:
            res[name] = v2n(data)[:components].astype(np.float32)
    return res


def gather_array(arrays, arrayname, mintime = 1e-12):
    """
    Given a dictionary of numpy arrays, this method gathers all the arrays
    containing a certain substring in the array name.
  
    Arguments:
        arrays: Arrays look into.
        arrayname (string): Substring to look for.
        mintime (float): Minimum time to consider. Default value = 1e-12.
    Returns:
        Dictionary of arrays (key: time, value: numpy array)
  
    """
    out   = {}
    for array in arrays:
        if arrayname in array:
            time = float(array.replace(arrayname + "_",""))
            if time > mintime:
                out[time] = arrays[array]

    return out

def write_graph(graph, outfile):
    """
    Write a vtk file given a graph. 
    The file can be opened in Paraview.
  
    Arguments:
        graph: A DGL graph.
        outfile (string): Output file name

    """

    points = graph.ndata['x'].detach().numpy()
    edges0 = graph.edges()[0].detach().numpy()
    edges1 = graph.edges()[1].detach().numpy()

    cells = {
        'line': np.vstack((edges0, edges1)).transpose()
    }

    type = np.argmax(graph.ndata['type'].detach().numpy(),
                     axis = 1)

    point_data = {
        'type': type
    }

    meshio.write_points_cells(outfile, points, cells,
                              point_data = point_data)

    bpoints = np.where(type > 1)[0]

    points = points[bpoints,:]
    print(points)
    x1 = np.arange(points.shape[0])
    x2 = (x1 + 1)
    x2[-1] = 0
    cells = {
        'line': np.vstack((x1, x2)).transpose()
    }

    point_data = {
        'type': type[bpoints]
    }

    print(cells)

    meshio.write_points_cells('boundary.vtk', points, cells,
                              point_data = point_data)
    
def write_solution(graph, solution, outfile, outdir = '.'):
    """
    Write vtk files (one per timestep) given a graph and a solution.
    The file can be opened in Paraview.
  
    Arguments:
        graph: A DGL graph.
        solution: Tuple containing two n x m tensors, where n is the number of
                  nodes and m the number of timesteps. The first tensor contains
                  the pressure solution, the second contains 
                  the flow rate solution
        outfile (string): name of the ouput file
        outdir (string): directory where results should be stored

    """

    ntimesteps = solution[0].shape[2]

    if outdir != '.':
        create_directory(outdir)

    for t in range(ntimesteps):

        points = graph.ndata['x'].detach().numpy()
        edges0 = graph.edges()[0].detach().numpy()
        edges1 = graph.edges()[1].detach().numpy()

        type = np.argmax(graph.edata['type'].detach().numpy(),
                     axis = 1)
        
        p_edges = np.where(type < 2)[0]

        cells = {
            'line': np.vstack((edges0[p_edges], edges1[p_edges])).transpose()
        }

        point_data = {
            'pressure': solution[0][:,0,t],
            'flowrate': solution[1][:,0,t]
        }

        o = '/' + outfile + '_' + "%04d" % (t,) + '.vtk'
        meshio.write_points_cells(outdir + o, points, cells,
                                  point_data = point_data)
    







