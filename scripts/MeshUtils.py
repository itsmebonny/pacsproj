import gmsh
import argparse
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import GenerateGraph as gg
import os
import sys

class MeshCreator:
    """
    A class for creating mesh using Gmsh.

    Args:
        args: An object containing the arguments for mesh creation.

    Attributes:
        args: The arguments for mesh creation.
        nmesh: The number of meshes to create.
        seed: The seed for random number generation.
        hmax: The maximum value for h.
        hmin: The minimum value for h.
        lc: The characteristic length.
        wmax: The maximum value for w.
        wmin: The minimum value for w.
        spacing: A flag indicating whether to use random spacing.
        nodes: The number of nodes.

    """

    def __init__(self, args):
        self.args = args
        self.nmesh = args.nmesh
        self.seed = args.seed
        self.hmax = args.hmax
        self.hmin = args.hmin
        self.lc = args.lc
        self.wmax = args.wmax
        self.wmin = args.wmin
        self.spacing = args.spacing
        self.nodes = args.nodes

        # aggiungere se fare interfaccia random o no

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
            # Initialize Gmsh
            gmsh.initialize()

            # Create a new model
            model = gmsh.model
            w = self.wmax
            wold = 0
            points = []
            for i in range(self.nodes):
                h = round(np.random.uniform(self.hmin, self.hmax),2)
                
                points += [gmsh.model.geo.addPoint(wold, -h, 0, self.lc)]
                points += [gmsh.model.geo.addPoint(wold, h, 0, self.lc)]
                if self.spacing:
                    w = round(np.random.uniform(self.wmin, self.wmax),2)
                wold += w
            # Define the rectangle coordinates
            print(points)

            # Add the rectangle to the model
            list_lines_x=[]
            list_lines_y = []
            for i in range(len(points)-2):
                if i % 2 == 0:
                    list_lines_x.append(gmsh.model.geo.addLine( points[i], points[i+2] ))
                else:
                    list_lines_x.append(gmsh.model.geo.addLine( points[i+2], points[i] ))

            for i in range(0,len(points)-1,2):
                list_lines_y.append(gmsh.model.geo.addLine( points[i], points[i+1] ))

            faces = []
            for i in range(0,len(list_lines_x)-1,2):
                face = gmsh.model.geo.addCurveLoop([-list_lines_y[i//2], list_lines_x[i], list_lines_y[i//2+1], list_lines_x[i+1]])
                faces.append(face)
                gmsh.model.geo.addPlaneSurface([face])

            #physical group
            gmsh.model.geo.addPhysicalGroup(1, list_lines_x, 1) # horizontal
            gmsh.model.geo.addPhysicalGroup(1, [list_lines_y[0]], 2) # left wall
            gmsh.model.geo.addPhysicalGroup(1, [list_lines_y[-1]], 3) # right wall
            for i in range(1,len(list_lines_y)-1): # interfaces
                model.addPhysicalGroup(1, [list_lines_y[i]], i+3)
            for j in range(len(faces)): # faces
                gmsh.model.addPhysicalGroup(2, [faces[j]], i+j+4)

            gmsh.model.geo.synchronize()
            # Generate mesh:
            gmsh.model.mesh.generate()

            # Write mesh data:
            gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
            gmsh.write(output_dir + "/" + filename + f"_{it}.msh")

            # Creates graphical user interface
            #if 'close' not in sys.argv:
            # if plot:
            # gmsh.fltk.run()

            # It finalize the Gmsh API
            gmsh.finalize()
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

class MeshLoader:

    def __init__(self,filename):
        self.meshfile = filename
        self.mesh = Mesh(self.meshfile + ".xml")
        self.bounds = MeshFunction("size_t", self.mesh, self.meshfile + "_facet_region.xml")
        self.face = MeshFunction("size_t", self.mesh, self.meshfile + "_physical_region.xml")
        self.n = FacetNormal(self.mesh)
        self.h = self.mesh.hmin()
    
    
    def update_tags(self,tags):
        """
        Method to save the tags of the boundaries and faces of the mesh

        The method takes as input a dictionary with the following keys: 
        'walls', 'inlet', 'outlet', 'interface', 'faces' and as values a list of the corresponding tags
        """
        self.tags = tags
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

    # method to define measures for integration over boundaries and faces
    def measure_definition(self):

        # Define measure for integration over external boundaries (inlet and outlet)
        self.dS = Measure("dS",domain=self.mesh, subdomain_data=self.rename_boundaries)

        # Define measure for integration over internal boundaries (interface)
        self.ds = Measure("ds",domain=self.mesh, subdomain_data=self.rename_boundaries)

        # Define measure for integration over faces
        self.dx = Measure("dx",domain=self.mesh, subdomain_data=self.rename_faces)

        return self.dS, self.ds, self.dx

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

    mesh_creator = MeshCreator(args)
    mesh_creator.create_mesh( "TestMeshes","data/mesh_test2")
    