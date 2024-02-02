#!/usr/bin/env python3

## @package GenerateData
#  @brief This file implements a solver class to solve a variational problem and a data generator class to store the data and the solutions of the problem solved in a dgl graph.
# 
#  This file contains the implementation of two abstract classes: Solver and DataGenerator. 
#  The Solver class is used to solve a variational problem on a given mesh, while the DataGenerator class is used to store the data and the solutions of the problem at each time step in a dgl graph. 
#  The Solver class is then inherited by two subclasses: Stokes and Heat, which are used to solve the Stokes equation and the heat equation, respectively. 
#  The DataGenerator class is also inherited by two subclasses: DataNS and DataHeat, which are used to generate the data for the Stokes equations and the heat equation, respectively. 
#  This class needs to be initialized with the proper solver object and mesh object.
#
#  @authors Andrea Bonifacio and Sara Gazzoni



from dolfin import *
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import GenerateGraph as gg
import torch as th
import os
import json
import re
            
class Solver(ABC):
    """
    This class represents a solver for a variational problem on a given mesh.

    This class is an abstract base class (ABC) and cannot be instantiated.

    Attributes:
        mesh (Mesh): Mesh object.
    """
    
    def __init__(self,mesh):
        """
        Initialize the Solver class.

        Args: 
            mesh (Mesh): Mesh object.
        """

        self.mesh = mesh

    @abstractmethod 
    def set_parameters(self):
        pass 

    @abstractmethod 
    def solve(self):
        pass 

    @abstractmethod 
    def plot_solution(self):
        pass 


class Stokes(Solver):
    """
    Class representing a Stokes solver.

    This class inherits from the Solver class and provides 
    methods to solve the Stokes equation.

    Attributes:
        mesh (Mesh): Mesh object.
        V (FunctionSpace): Velocity function space.
        Q (FunctionSpace): Pressure function space.
        Re (float): Reynolds number.
        inflow (float): Inflow rate of the fluid.
        f (Expression): Source term.
        dt (float): Time step.
        T (float): Final time.
        k (float): Reynolds number.
        doplot (bool): Flag indicating whether to 
                       plot the solution at each time step.
        ts (numpy.ndarray): Array of time steps.
        ut (numpy.ndarray): Array of velocity solutions at each time step.
        pt (numpy.ndarray): Array of pressure solutions at each time step.
    """

    def __init__(self, mesh, V, Q, Re, inflow, f, dt, T, doplot=False):
        """
        Initialize the GenerateData class.

        Args:
            V (FunctionSpace): Velocity function space.
            Q (FunctionSpace): Pressure function space.
            Re (float): Reynolds number.
            inflow (float): Inflow rate of the fluid.
            f (Expression): Source term.
            dt (float): Time step.
            T (float): Final time.
            doplot (bool): Flag indicating whether to plot
                           the solution at each time step.
        """
        super().__init__(mesh)
        self.V = V
        self.Q = Q
        self.k = Re
        self.inflow = inflow
        self.f = f
        self.dt = dt
        self.T = T
        self.doplot = doplot


    def set_parameters(self, V, Q, Re, inflow, f, dt, T):
        """
        Set the parameters for the simulation.

        Args:
            V (FunctionSpace): Velocity function space.
            Q (FunctionSpace): Pressure function space.
            Re (float): Reynolds number.
            inflow (float): Inflow rate of the fluid.
            f (Expression): Source term.
            dt (float): Time step.
            T (float): Final time.
        """
        self.V = V
        self.Q = Q
        self.k = Re
        self.inflow = inflow
        self.f = f
        self.dt = dt
        self.T = T

    def solve(self):
        """
        Solve the Stokes equation.

        """
        TH = self.V *self.Q
        W = FunctionSpace(self.mesh.mesh, TH)

        # No-slip boundary condition for velocity
        bcs = []
        noslip = Constant((0.0, 0.0))
        for i in self.mesh.tags['walls']:
            bcs.append(DirichletBC(W.sub(0), noslip, self.mesh.bounds, i))
        
        # Inflow boundary condition for velocity
        for i in self.mesh.tags['inlet']:
            bcs.append(DirichletBC(W.sub(0), self.inflow, self.mesh.bounds, i))
        for i in self.mesh.tags['outlet']:
            bcs.append(DirichletBC(W.sub(1), Constant(0.0), self.mesh.bounds, i))

        # Define variational problem
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
        f = Constant((0, 0))
        P = W.sub(0).collapse()
        Z = W.sub(1).collapse()
        fa = FunctionAssigner([P,Z],W)
        u0 = Function(P)
        p0 = Function(Z)

        # self.k = self.U0 * self.L0 / self.nu
        # self.k = round(self.k,2)
        a = (1/self.dt)*inner(u,v)*dx + (1/self.k)*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx
        L = inner(f,v)*dx + (1/self.dt)*inner(u0,v)*dx

        U = Function(W)
        self.ts = np.arange(0,self.T,self.dt)
        self.ut = np.empty(len(self.ts), dtype=object)
        self.pt = np.empty(len(self.ts), dtype=object)
        nsteps = int(self.T/self.dt)
        t = 0
        while(t<nsteps):
            solve(a == L, U, bcs)
            u, p = U.split()
            fa.assign([u0,p0],U)
            temp1 = Function(P)
            temp1.vector()[:] = u0.vector()[:]
            self.ut[t] = temp1
            temp2 = Function(Z)
            temp2.vector()[:] = p0.vector()[:]
            self.pt[t] = temp2
            
            if self.doplot:
                self.plot_solution(u,p)

            t+=1

        self.u = u
        self.p = p

    def plot_solution(self, u, p):
        """
        Plot the solution u and p.

        Args:
            u: The velocity solution.
            p: The pressure solution.
        """
        sol1 = plot(u)
        plt.colorbar(sol1)
        plt.show()
        sol2 = plot(p)
        plt.colorbar(sol2)
        plt.show()


class Heat(Solver):
    """
    Class representing a heat solver.

    This class inherits from the Solver class and provides methods to solve the heat equation 
    using the Discontinuous Galerkin method with non-homogeneous Neumann boundary conditions.

    Attributes:
        V (FunctionSpace): Function space for the solution.
        k (float): Thermal conductivity.
        f (Expression): Source term.
        u0 (Expression): Initial condition.
        dt (float): Time step.
        T (float): Final time.
        g (Expression): Neumann boundary condition at the inlet.
        doplot (bool): Flag indicating whether to plot the solution at each time step.
        ts (numpy.ndarray): Array of time steps.
        ut (numpy.ndarray): Array of solutions at each time step.
    """

    def __init__(self, mesh, V, k, f, u0, dt, T, g, doplot=False):
        """
        Initialize the Heat class.

        Args:
            mesh (Mesh): Mesh object.
            V (FunctionSpace): Function space for the solution.
            k (float): Thermal conductivity.
            f (Expression): Source term.
            u0 (Expression): Initial condition.
            dt (float): Time step.
            T (float): Final time.
            g (Expression): Neumann boundary condition at the inlet.
            doplot (bool): Flag indicating whether to plot
                           the solution at each time step.
        """

        super().__init__(mesh)
        self.V = V
        self.k = k
        self.f = f
        self.u0 = u0
        self.dt = dt
        self.T = T
        self.g = g
        self.doplot = doplot

    def set_parameters(self, V, k, f, u0, dt, T, g):
        """
        Method to set different parameters for the heat solver.

        Args:
            V (FunctionSpace): Function space for the solution.
            k (float): Thermal conductivity.
            f (Expression): Source term.
            u0 (Expression): Initial condition.
            dt (float): Time step.
            T (float): Final time.
            g (Expression): Neumann boundary condition at the inlet.
        """
        self.V = V
        self.k = k
        self.f = f
        self.u0 = u0
        self.dt = dt
        self.T = T
        self.g = g

    def solve(self):
        """
        Method to solve the heat equation.

        The problem is solved using the Discontinuous Galerkin method 
        and imposing non-homogeneous Neumann boundary condition at the inlet 
        and homogeneous Neumann boundary condition at the outlet and walls.

        Returns:
            numpy.ndarray: Array of solutions at each time step.
        """
        t = float(self.dt)
        u0 = interpolate(self.u0,self.V)
        U = Function(self.V)

        # Define variational problem
        u=TrialFunction(self.V)
        v=TestFunction(self.V)
        a_int=u*v*dx+self.dt*self.k*inner(grad(u),grad(v))*dx 
        a_facet = self.k*(10/avg(self.mesh.h)*dot(jump(v,self.mesh.n),jump(u,self.mesh.n))*dS - dot(avg(grad(v)), jump(u, self.mesh.n))*dS - dot(jump(u, self.mesh.n), avg(grad(v)))*dS)

        a = a_int + a_facet
        L=u0*v*dx+self.dt*self.f*v*dx + self.g*v*self.dt*self.mesh.ds(self.mesh.tags['inlet'][0])

        self.ts = np.arange(0,self.T,self.dt)
        self.ut = np.empty(len(self.ts), dtype=object)

        # Solve the heat equation at each time step
        nsteps = int(self.T/self.dt)
        t=0
        while(t<nsteps):
            temp = Function(self.V)
            temp.vector()[:] = u0.vector()[:]
            self.ut[t] = temp
            self.g.t=t*self.dt
            solve(a==L,U)

            # Update
            u0.assign(U)
            t+=1

            # Plot solution at each time step
            if self.doplot:
                self.plot_solution(U)
        return self.ut 


    def plot_solution(self, u):
        """
        Plot the solution.

        Args:
            u: The solution to be plotted.
        """
        sol = plot(u)
        plot(u)
        plt.colorbar(sol)
        plt.show()
           

class DataGenerator(ABC):
    """
    This class represents a data generator given a mesh and a solver object. 
    It stores the data and the solutions of a variational problem in a dgl graph.

    This class is an abstract base class (ABC) and cannot be instantiated.

    Attributes:
        solver: The solver object.
        mesh: The mesh object.
        NNodes: The number of interfaces in the mesh.
        edges1 (numpy.ndarray): Source nodes of the edges.
        edges2 (numpy.ndarray): Destination nodes of the edges.
        center_line (numpy.ndarray): The centerline coordinates of the inlet, 
                                     interfaces and outlet of the mesh.
        NodesData (dict): A dictionary containing the nodes features.
        EdgesData (dict): A dictionary containing the edges features.

    """

    def __init__(self, solver, mesh):
        """
        Initializes the DataGenerator object.

        Args:
            solver: The solver object.
            mesh: The mesh object.
        """
        self.solver = solver
        self.mesh = mesh
        self.NNodes = len(self.mesh.tags['interface']) + len(self.mesh.tags['inlet']) + len(self.mesh.tags['outlet'])

    @abstractmethod
    def flux(self):
        pass

    @abstractmethod
    def inlet_flux(self,tag, u):
        pass

    def area(self,tag):
        """
        Calculates the area of the face with the specified tag.

        Args:
            tag: The tag of the face.

        Returns:
            The area of the face with the specified tag.
        """
        area = assemble(Constant(1.0)*self.mesh.dx(tag))
        return area

    def create_edges(self):
        """
        Creates the edges of the mesh.

        Returns:
            edges1 (numpy.ndarray): Source nodes of the edges.
            edges2 (numpy.ndarray): Destination nodes of the edges.
        """
        if not hasattr(self, 'NodesData'):
            self.nodes_data()
        self.edges1 = self.NodesData['NodeId'][:-1]
        self.edges2 = self.NodesData['NodeId'][1:]
        return self.edges1,self.edges2
    
    def edges_data(self):
        """
        Stores the edges data in a dictionary.

        The data stored for each edge are the edge ID, the area of the face 
        where the edge is located and the length of the edge.

        Returns: 
            The dictionary containing the edges data.
        """
        if not hasattr(self, 'NodesData'):
            self.nodes_data()
        dict_e = {'area': np.zeros(self.NNodes-1), 'length': np.zeros(self.NNodes-1)}
        dict_e['edgeId'] = np.arange(0,self.NNodes-1)
        for j in range(self.NNodes-1):
            dict_e['area'][j] = self.area(self.mesh.tags['faces'][j])
            dict_e['length'][j] = self.center_line[j+1][0]-self.center_line[j][0]
        self.EdgesData = dict_e
        return dict_e

    def nodes_data(self):
        """
        Stores the data of the nodes in a dictionary.

        The data stored are the thermal conductivity, the node IDs, 
        the inlet mask, the outlet mask and the length of the interface 
        where the node is located.

        Returns:
            The dictionary containing the nodes data.
        """
        if not hasattr(self, 'center_line'):
            self.centerline()
        dict = {'k': self.solver.k, 'inlet_mask': np.zeros(self.NNodes,dtype=bool), 'outlet_mask':np.zeros(self.NNodes,dtype=bool), 'interface_length': np.zeros(self.NNodes)}
        dict['NodeId'] = np.arange(0,self.NNodes)
        dict['inlet_mask'][0] = 1
        dict['outlet_mask'][-1] = 1
        it = 0
        for i in self.mesh.tags['inlet']:
            dict['interface_length'][it] = round(assemble(Constant(1.0)*self.mesh.ds(i)),2)
            it+=1
        for i in self.mesh.tags['interface']:
            dict['interface_length'][it] = round(assemble(Constant(1.0)*self.mesh.dS(i)),2)
            it+=1  
        for i in self.mesh.tags['outlet']:
            dict['interface_length'][it] = round(assemble(Constant(1.0)*self.mesh.ds(i)),2)  
            it+=1

        self.NodesData = dict

        return dict

    @abstractmethod
    def td_nodes_data(self):
        pass

    def centerline(self):
        """
        Computes the centerline coordinates of the mesh corresponding 
        to the inlet, interfaces and outlet.

        Returns:
            A n x 2 numpy array with the centerline coordinates of the mesh, 
            where n is the number of nodes.
        """
        center_line = np.zeros((self.NNodes,2)) # should be an array of arrays (commonly known as list of lists)
        tags_list = ['inlet','interface','outlet']
        it=0
        for j in tags_list:
            for i in self.mesh.tags[j]:
                edge_coord =[]
                # Extract the coordinates of the vertices for each edge with the specified tag
                for edge in edges(self.mesh.mesh):
                    if self.mesh.bounds.array()[edge.index()] == i:
                        for vertex in vertices(edge):
                           coordinate = vertex.point().array()
                           edge_coord.append(coordinate)
                edge_coord = np.stack(edge_coord, axis=0)

                # Calculate the midpoint of the edge and append to the centerline list
                center_line[it] = (np.max(edge_coord[:,0])+np.min(edge_coord[:,0]))/2,(np.max(edge_coord[:,1])+np.min(edge_coord[:,1]))/2
                it+=1
        self.center_line = center_line
        return center_line

    def save_graph(self, output_dir, fields_names):
        """
        Saves the graph with the specified fields.

        Args:
            fields_names: The names of the fields.
            output_dir: The output directory to save the graph.

        Returns:
            The saved graph.
        """
        if not hasattr(self, 'NodesData'):
            self.nodes_data()
        if not hasattr(self, 'TDNodesData'):
            self.td_nodes_data()
        if not hasattr(self, 'edges1'):
            self.create_edges()
        if not hasattr(self, 'center_line'):
            self.centerline()
        if not hasattr(self, 'EdgesData'):
            self.edges_data()
        self.graph = gg.generate_graph(self.NodesData, self.center_line, self.EdgesData, self.edges1, self.edges2)
        for i,key in enumerate(fields_names):
            gg.add_field(self.graph, self.TDNodesData[i], key)
        gg.save(self.graph, f"k_{self.solver.k}", output_dir)
        return self.graph
    
    def generate_json(self,output_dir,model_type):
        """
        Generate JSON file containing information about the dataset.

        Arguments:
            output_dir (string): path to output directory
            model_type (string): equation type (e.g. "heat")
        """

        input_directory = os.path.expanduser(output_dir)

        input_directory = os.path.realpath(input_directory)

        # Initialize an empty dictionary to store JSON objects for each file
        json_dict = dict()

        # Iterate through each .grph file in the directory
        for graph_file in os.listdir(input_directory):
            if graph_file.endswith(".grph"):
                # Extract the filename without extension
                filename_no_extension = os.path.splitext(graph_file)[0]

                # Create a dictionary for each .grph file
                json_dict[filename_no_extension] = {
                    "model_type": model_type
                }

        # Save the dictionary as a JSON file
        json_file_path = os.path.join(input_directory, "dataset_info.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, indent=2)

        print(f"Created {json_file_path}")
    

class DataNS(DataGenerator):
    """
    This class represents a data generator for a Stokes solver. 
    It inherits from the DataGenerator class.

    Attributes:
        solver (Solver): The solver object used for solving the Stokes equations.
        mesh (Mesh): The mesh object representing the computational domain.
        model_type (string): The type of model (stokes).
        target_fields (string): The fields that will be predicted 
                                by the graph neural network (flowrate and pressure).
        TDNodesData (dict): A dictionary containing the time-dependent data 
                            at each node.
    """

    def __init__(self, solver, mesh):
        """
        Constructor for the DataNS class.

        Args:
            solver (Solver): The solver object used to solve the Stokes equations.
            mesh (Mesh): The mesh object representing the computational domain.
        """
        super().__init__(solver, mesh)
        self.model_type = "stokes"
        self.target_fields = ['flowrate','pressure']

    def flux(self,tag,u):
        """
        Computes the flowrate at a given interface.

        Args:
            tag: The tag representing the interface.
            u: The velocity variable.
        """
        flux = dot(u, self.mesh.n('+'))*self.mesh.dS(tag)
        total_flux = assemble(flux)
        return total_flux

    def inlet_flux(self,tag, u):
        """
        Computes the flowrate at the inlet.

        Args:
            tag: The tag representing the inlet.
            u: The velocity variable.
        """

        flux = -dot(u, self.mesh.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux

    def outlet_flux(self,tag, u):
        """
        Computes the flowrate at the outlet.

        Args:
            tag: The tag representing the outlet.
            u: The velocity variable.
        """
        flux = dot(u, self.mesh.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux
    
    def mean_pressure_interface(self,tag,p):
        """
        Calculates the mean pressure on a specified interface.

        Args:
            tag: The tag representing the interface.
            p: The pressure variable.
        """
        mean_p = p*self.mesh.dS(tag)
        length = assemble(Constant(1.0)*self.mesh.dS(tag))
        mean_p = assemble(mean_p)
        return mean_p/length

    def mean_pressure_boundaries(self,tag,p):
        """
        Calculates the mean pressure on a specified boundary.

        Args:
            tag: The tag representing the boundary (inlet or outlet).
            p: The pressure variable.
        """
        mean_p = p*self.mesh.ds(tag)
        length = assemble(Constant(1.0)*self.mesh.ds(tag))
        mean_p = assemble(mean_p)
        return mean_p/length
            
    def td_nodes_data(self):
        """
        Stores the time-dependent data in a dictionary.

        The data stored are the flowrate and the pressure at each node and at each time step.

        Returns:
            A dictionary containing the flowrate and a dictionary containing 
            the pressure, both at each node and at each time step.
        """
        td_dict_u = dict()
        td_dict_p = dict()
        for t in range(len(self.solver.ts)):
            td_dict_u[self.solver.ts[t]] = np.zeros(self.NNodes)
            td_dict_p[self.solver.ts[t]] = np.zeros(self.NNodes)
            it = 0
            for i in self.mesh.tags['inlet']:
                td_dict_u[self.solver.ts[t]][it] = self.inlet_flux(i,self.solver.ut[t])
                td_dict_p[self.solver.ts[t]][it] = self.mean_pressure_boundaries(i,self.solver.pt[t])
                it+=1
            for i in self.mesh.tags['interface']:
                td_dict_u[self.solver.ts[t]][it] = self.flux(i,self.solver.ut[t])     
                td_dict_p[self.solver.ts[t]][it] = self.mean_pressure_interface(i,self.solver.pt[t]) 
                it+=1 
            for i in self.mesh.tags['outlet']:
                td_dict_u[self.solver.ts[t]][it] = self.outlet_flux(i,self.solver.ut[t])
                td_dict_p[self.solver.ts[t]][it] = self.mean_pressure_boundaries(i,self.solver.pt[t])
                it+=1

        self.TDNodesData = [td_dict_u, td_dict_p]
        return td_dict_u, td_dict_p

    def save_graph(self, output_dir = "../data/graphs"):    
        """
        Saves the graph in a specified directory.

        This function calls the save_graph method of the super class, 
        passing the model_type and target_fields attributes.

        Args:
            output_dir: The directory to save the graph in.
        """
        return super().save_graph(output_dir,self.target_fields)
    
    def generate_json(self,output_dir = "../data/graphs"):
        """
        Generate JSON file containing information about the dataset.

        The function calls the generate_json method of the super class, 
        passing the model_type attributes.

        Arguments:
            output_dir (string): path to output directory
        """
        return super().generate_json(output_dir,self.model_type)
    

# Define a subclass DataHeat that inherits from the DataGenerator abstract base class
class DataHeat(DataGenerator):
    """
    This class represents a data generator for a heat solver.
    It inherits from the DataGenerator class.

    Attributes:
        solver: The solver object for heat equation.
        mesh: The mesh object representing the computational domain.
        model_type (string): The type of model (heat).
        fields_names (string): The fields that will be predicted by the graph neural network (flux).
        TDNodesData (dict): A dictionary containing the time-dependent data at each node.
    """

    def __init__(self, solver, mesh):
        """
        Constructor for the DataHeat class.

        Args:
            solver: The solver object used to solve the heat equation.
            mesh: The mesh object representing the computational domain.
        """
        super().__init__(solver, mesh)
        self.model_type = "heat"
        self.target_fields = ['flux']

    
    def flux(self, interface, u):
        """
        Calculates the heat flux at a specified interface.

        Args:
            interface: The tag representing the interface.
            u: The temperature field.

        Returns:
            the heat flux at the specified interface.
        """
        flux = -self.solver.k * dot(grad(u)('+'), self.mesh.n('+')) * self.mesh.dS(interface)
        total_flux = assemble(flux)
        return total_flux
    
    def inlet_flux(self, tag, u):
        """
        Calculates the heat flux at the inlet.

        Args:
            tag: The tag representing the inlet.
            u: The temperature field.

        Returns:
            the heat flux at the inlet.
        """
        
        flux = self.solver.k * dot(grad(u), self.mesh.n) * self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux

    def td_nodes_data(self):
        """
        Stores the time-dependent data in a dictionary.

        The data stored are the heat flux at each node and at each time step.

        Returns:
            A dictionary containing the heat flux at each node and at each time step.
        """
        td_dict = dict()
        for t in range(len(self.solver.ts)):
            td_dict[self.solver.ts[t]] = np.zeros(self.NNodes)
            it = 0
            for i in self.mesh.tags['inlet']:
                td_dict[self.solver.ts[t]][it] = self.inlet_flux(i, self.solver.ut[t])
                it += 1
            for i in self.mesh.tags['interface']:
                td_dict[self.solver.ts[t]][it] = self.flux(i, self.solver.ut[t])  
                it += 1    

        self.TDNodesData = [td_dict]
        return td_dict
    
    def save_graph(self, output_dir = "../data/graphs"):
        """
        Saves the graph in a specified directory.

        This function calls the save_graph method of the super class, 
        passing the model_type and fields_names attributes.

        Args:
            output_dir: The directory to save the graph in.
        """
        return super().save_graph(output_dir, self.target_fields)

    def generate_json(self,output_dir = "../data/graphs"):
        """
        Generate JSON file containing information about the dataset.

        The function calls the generate_json method of the super class, 
        passing the model_type attributes.

        Arguments:
            output_dir (string): path to output directory
        """
        return super().generate_json(output_dir, self.model_type)