#!/usr/bin/env python3

## @package MeshUtils
#  @brief This file contains utilities for creating and loading meshes.
#
#  The MeshUtils file contains two classes: MeshCreator and MeshLoader. 
#  The MeshCreator class is used to generate meshes using Gmsh and to convert meshes from msh to xml format. 
#  The MeshLoader class is used to load meshes from xml files to be used in FEniCS. 
#  The file also contains a main function to call the MeshCreator class. 
#  The mesh parameters can be set in the main function or using the command line.
#
#  @authors Andrea Bonifacio and Sara Gazzoni

import gmsh
import argparse
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import GenerateGraph as gg
import os
import sys
import json

class MeshCreator:
    """
    A class for creating mesh using Gmsh.

    Attributes:
        nmesh: Number of meshes to create.
        seed: Seed for random number generation.
        hmax: Maximum value for the interfaces height.
        hmin: Minimum value for the interfaces height.
        lc: Characteristic length.
        wmax: Maximum value for the distance between nodes.
        wmin: Minimum value for the distance between nodes.
        spacing: Flag indicating whether the nodes are equispaced or not.
        nodes: Number of nodes.

    """

    def __init__(self, args):
        """
        Constructor for the MeshCreator class.

        Args:
            args: The arguments for the mesh creation.
        """
        self.nmesh = args.nmesh
        self.seed = args.seed
        self.hmax = args.hmax
        self.hmin = args.hmin
        self.lc = args.lc
        self.wmax = args.wmax
        self.wmin = args.wmin
        self.spacing = args.spacing
        self.nodes = args.nodes

    def create_mesh(self, filename, output_dir):
        """
        Create a mesh using Gmsh.

        Args:
            filename: The name of the output file.
            output_dir: The directory to save the output file.
            plot: A flag indicating whether to plot the mesh.

        """
        #check existence of output_dir 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        np.random.seed(self.seed)
        for it in range(self.nmesh):
            gmsh.initialize()
            model = gmsh.model
            w = self.wmax
            wold = 0
            points = []

            # add points 
            for i in range(self.nodes):
                h = round(np.random.uniform(self.hmin, self.hmax),2)
                points += [gmsh.model.geo.addPoint(wold, -h, 0, self.lc)]
                points += [gmsh.model.geo.addPoint(wold, h, 0, self.lc)]
                if self.spacing:
                    w = round(np.random.uniform(self.wmin, self.wmax),2)
                wold += w

            # add lines
            list_lines_x=[]
            list_lines_y = []
            for i in range(len(points)-2):
                if i % 2 == 0:
                    list_lines_x.append(gmsh.model.geo.addLine( points[i], points[i+2] ))
                else:
                    list_lines_x.append(gmsh.model.geo.addLine( points[i+2], points[i] ))

            for i in range(0,len(points)-1,2):
                list_lines_y.append(gmsh.model.geo.addLine( points[i], points[i+1] ))

            # add faces
            faces = []
            for i in range(0,len(list_lines_x)-1,2):
                face = gmsh.model.geo.addCurveLoop([-list_lines_y[i//2], list_lines_x[i], list_lines_y[i//2+1], list_lines_x[i+1]])
                faces.append(face)
                gmsh.model.geo.addPlaneSurface([face])

            # add physical group
            gmsh.model.geo.addPhysicalGroup(1, list_lines_x, 1) # horizontal
            gmsh.model.geo.addPhysicalGroup(1, [list_lines_y[0]], 2) # left wall
            gmsh.model.geo.addPhysicalGroup(1, [list_lines_y[-1]], 3) # right wall
            for i in range(1,len(list_lines_y)-1): # interfaces
                model.addPhysicalGroup(1, [list_lines_y[i]], i+3)
            for j in range(len(faces)): # faces
                gmsh.model.addPhysicalGroup(2, [faces[j]], i+j+4)

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate()
            gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
            gmsh.write(output_dir + "/" + filename + f"_{it}.msh")
            gmsh.finalize()

            self.create_info_file(output_dir, filename)

    def convert_mesh(self, output_dir):
        """
        Convert the meshes in a given directory from msh to xml format.

        Args:
            output_dir: The directory containing the meshes to convert.
        """

        # Check if the provided path is a directory
        if not os.path.isdir(output_dir):
            print("Error: '{}' is not a valid directory.".format(output_dir))
            sys.exit(1)

        # Iterate over all files in the folder and run dolfin-convert
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".msh"):
                # Extract the filename without extension
                filename = os.path.splitext(file_name)[0]

                # Run dolfin-convert
                file_path = os.path.join(output_dir, file_name)
                output_path = os.path.join(output_dir, "{}.xml".format(filename))
                
                print("Converting file: {}".format(file_path))
                os.system("dolfin-convert {} {}".format(file_path, output_path))

        print("Conversion complete.")

    def create_info_file(self, output_dir, meshname):
        """
        Create a json file containing the mesh information.

        Args:
            output_dir: The directory to save the json file.
        """

        json_dict = dict()
        json_dict["nmesh"] = self.nmesh
        json_dict["hmax"] = self.hmax
        json_dict["hmin"] = self.hmin
        json_dict["lc"] = self.lc
        json_dict["wmax"] = self.wmax
        json_dict["wmin"] = self.wmin
        json_dict["spacing"] = self.spacing
        json_dict["nodes"] = self.nodes
        json_dict["mesh_name"] = meshname

        json_file_path = os.path.join(output_dir, "mesh_info.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, indent=2)
        
        print("Info file created.")


class MeshLoader:
    """
    A class for loading mesh from a xml file to be used in FEniCS.

    Attributes:
        meshfile: The name of the mesh file.
        mesh: The FEniCS mesh.
        bounds: FEniCS MeshFunction for the boundaries of the mesh.
        face: FEniCS MeshFunction for the faces of the mesh.
        n: The normal vector of the mesh.
        h: The characteristic length of the mesh.
        tags: A dictionary containing the tags of the mesh.
        rename_boundaries: FEniCS MeshFunction for the boundaries of the mesh with the tags.
        rename_faces: FEniCS MeshFunction for the faces of the mesh with the tags.
        dS: Measure for integration over external boundaries (inlet and outlet).
        ds: Measure for integration over internal boundaries (interface).
        dx: Measure for integration over faces.
    """

    def __init__(self,filename):
        """
        Constructor for the MeshLoader class.

        Args:
            filename: The name of the mesh file without extension.
        """

        self.meshfile = filename
        self.mesh = Mesh(self.meshfile + ".xml")
        self.bounds = MeshFunction("size_t", self.mesh, self.meshfile + "_facet_region.xml")
        self.face = MeshFunction("size_t", self.mesh, self.meshfile + "_physical_region.xml")
        self.n = FacetNormal(self.mesh)
        self.h = self.mesh.hmin()
    
    
    def update_tags(self,tags={},nodes = -1):
        """
        Method to save the tags of the boundaries and faces of the mesh.
        If the number of nodes are provided, the tags are created automatically.
        If the number of nodes are not provided, the tags must be provided.

        Args:
            tags: A dictionary containing the tags of the mesh.
            nodes: The number of nodes of the mesh.

        Returns:
            A dictionary containing the tags of the mesh.
        """
        if nodes == -1 and tags != {}:
            self.tags = tags
        elif nodes == -1 and tags == {}:
            print("Error: tags not provided")
            sys.exit(1)
        else:
            while nodes <= 0:
                nodes = int(input("Insert the number of nodes (MUST BE POSITIVE): "))
            self.tags = {"walls":[1],"inlet":[2],"outlet":[3],"interface":[],"faces":[]}
            for i in range(4,4+nodes-2):
                self.tags["interface"].append(i)
            for j in range(i+1,i+nodes):
                self.tags["faces"].append(j)

        self.rename_boundaries = MeshFunction("size_t", self.mesh,1)
        self.rename_boundaries.set_all(0)
        self.rename_faces = MeshFunction("size_t", self.mesh, 2)
        self.rename_faces.set_all(0)
        for j in self.tags:
            if j != "faces":
                for i in self.tags[j]:
                    self.rename_boundaries.array()[self.bounds.array()==i] = i
            else:
                for i in self.tags[j]:
                    self.rename_faces.array()[self.face.array()==i] = i
        return self.tags

    def measure_definition(self):
        """
        Method to define the measures for the integration.
        """

        self.dS = Measure("dS",domain=self.mesh, subdomain_data=self.rename_boundaries) # measure for integration over external boundaries (inlet and outlet)
        self.ds = Measure("ds",domain=self.mesh, subdomain_data=self.rename_boundaries) # measure for integration over internal boundaries (interface)
        self.dx = Measure("dx",domain=self.mesh, subdomain_data=self.rename_faces)      # measure for integration over faces
    
    def plot_mesh(self):
        """
        Method to plot the mesh.
        """
        plot(self.mesh)


"""
The main function launches the mesh creation and conversion. The mesh parameters can be set using the command line and the output directory and the filename can be set in the main function.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mesh creation')
    parser.add_argument('--nodes', help='number of nodes', type=int, default=5)
    parser.add_argument('--lc', help='mesh size', type=float, default=0.1)
    parser.add_argument('--hmax', help='max height of the domain', type=float, default=5.0)
    parser.add_argument('--hmin', help='min height of the domain', type=float, default=0.5)
    parser.add_argument('--nmesh', help='number of meshes', type=int, default=20)
    parser.add_argument('--seed', help='random seed', type=int, default=42)
    parser.add_argument('--spacing', help='equispaced nodes', type=bool, default=False)
    parser.add_argument('--wmax', help='max node distance', type=float, default=5.0)
    parser.add_argument('--wmin', help='min node distance', type=float, default=1.0)
    args = parser.parse_args()

    filename = "test_mesh"
    output_dir = "data/mesh_test/"

    mesh_creator = MeshCreator(args)
    mesh_creator.create_mesh(filename, output_dir)
    mesh_creator.convert_mesh(output_dir)
    