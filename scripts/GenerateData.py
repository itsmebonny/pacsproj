from dolfin import *
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import GenerateGraph as gg
import torch as th

import matplotlib as mpl  #temporaneo


            
class Solver(ABC):
    
    def __init__(self,mesh):
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

    def __init__(self, mesh, V, Q, rho, mu, U0, L0, inflow, f, dt, T, k, doplot=False):
        """
        Initialize the GenerateData class.

        Parameters:
        - mesh: the mesh object
        - V: velocity function space
        - Q: pressure function space
        - rho: density
        - mu: dynamic viscosity
        - U0: characteristic velocity
        - L0: characteristic length
        - inflow: inflow boundary condition
        - f: source term
        - dt: time step
        - T: final time
        - doplot: flag to plot solution at each time step (default: False)
        """
        super().__init__(mesh)
        self.V = V
        self.Q = Q
        self.rho = rho
        self.mu = mu
        self.U0 = U0
        self.L0 = L0
        self.inflow = inflow
        self.f = f
        self.dt = dt
        self.T = T
        self.k = k
        self.doplot = doplot


    def set_parameters(self, V, Q, rho, mu, U0, L0, inflow, f, dt, T):
        """
        Set the parameters for the simulation.

        Args:
            V (float): Velocity of the fluid.
            Q (float): Flow rate of the fluid.
            rho (float): Density of the fluid.
            mu (float): Viscosity of the fluid.
            U0 (float): Initial velocity of the fluid.
            L0 (float): Initial length of the fluid.
            inflow (float): Inflow rate of the fluid.
            f (float): Force applied to the fluid.
            dt (float): Time step for the simulation.
            T (float): Total time for the simulation.
        """
        self.V = V
        self.Q = Q
        self.rho = rho
        self.mu = mu
        self.U0 = U0
        self.L0 = L0
        self.inflow = inflow
        self.f = f
        self.dt = dt
        self.T = T

    def solve(self):
        """
        Solves the variational problem for velocity and pressure.

        Returns:
            None
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

        # self.k = self.rho * self.U0 * self.L0 / self.mu
        # self.k = round(self.k,2)
        a = (1/self.dt)*inner(u,v)*dx + (1/self.k)*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx
        L = inner(f,v)*dx + (1/self.dt)*inner(u0,v)*dx

        U = Function(W)
        # t = self.dt
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

        self.u = u*self.U0
        self.p = p*self.rho*self.U0*self.U0

    def plot_solution(self, u, p):
        """
        Plot the solution u and p.

        Parameters:
        u (array-like): The solution u.
        p (array-like): The solution p.
        """
        sol1 = plot(u)
        plt.colorbar(sol1)
        plt.show()
        sol2 = plot(p)
        plt.colorbar(sol2)
        plt.show()


# Define a subclass Heat that inherits from the Solver abstract base class to solve Heat equation
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
    """

    def __init__(self, mesh, V, k, f, u0, dt, T, g, doplot=False):
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


    # Method to plot the solution
    def plot_solution(self, u):
        """
        Plot the solution.

        Args:
            u: The solution to be plotted.

        Returns:
            None
        """
        sol = plot(u)
        plot(u)
        plt.colorbar(sol)
        plt.show()
           

class DataGenerator(ABC):
    """
    This class represents a data generator for a solver on a given mesh.
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
        """
        Abstract method to calculate the flux.
        """
        pass

    @abstractmethod
    def inlet_flux(self,tag, u):
        """
        Abstract method to calculate the inlet flux.

        Args:
            tag: The tag of the inlet.
            u: The velocity.
        """
        pass

    def area(self,tag):
        """
        Calculates the area of a given tag.

        Args:
            tag: The tag of the area.

        Returns:
            The area of the tag.
        """
        area = assemble(Constant(1.0)*self.mesh.dx(tag))
        return area

    def create_edges(self):
        """
        Creates the edges of the mesh.

        Returns:
            The edges of the mesh.
        """
        if not hasattr(self, 'NodesData'):
            self.nodes_data()
        self.edges1 = self.NodesData['NodeId'][:-1]
        self.edges2 = self.NodesData['NodeId'][1:]
        return self.edges1,self.edges2
    
    def edges_data(self):
        """
        Calculates the data of the edges.

        Returns:
            The data of the edges.
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
        Calculates the data of the nodes.

        Returns:
            The data of the nodes.
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
        """
        Abstract method to calculate the time-dependent data of the nodes.
        """
        pass

    def centerline(self):
        """
        Calculates the centerline coordinates of the mesh.

        Returns:
            The centerline coordinates of the mesh.
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
        # self.NNodes = len(center_line)
        return center_line

    def save_graph(self, fields_names, output_dir = "data/graphs/"):
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
        gg.save_graph(self.graph, f"k_{self.solver.k}", output_dir)
        return self.graph

class DataNS(DataGenerator):
    """
    This class represents a data generator for solving the Navier-Stokes equations.
    It inherits from the DataGenerator class.
    """

    def __init__(self, solver, mesh):
        """
        Constructor for the DataNS class.

        Parameters:
        - solver: The solver object used for solving the Navier-Stokes equations.
        - mesh: The mesh object representing the computational domain.
        """
        super().__init__(solver, mesh)

    def flux(self,tag,u):
        """
        Calculates the flux of a given variable across a specified tag.

        Parameters:
        - tag: The tag representing the boundary or interface.
        - u: The variable to calculate the flux for.

        Returns:
        - total_flux: The total flux of the variable across the tag.
        """
        flux = dot(u, self.mesh.n('+'))*self.mesh.dS(tag)
        total_flux = assemble(flux)
        return total_flux

    def inlet_flux(self,tag, u):
        """
        Calculates the inlet flux of a given variable across a specified tag.

        Parameters:
        - tag: The tag representing the inlet boundary.
        - u: The variable to calculate the flux for.

        Returns:
        - total_flux: The total inlet flux of the variable across the tag.
        """
        flux = -dot(u, self.mesh.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux

    def outlet_flux(self,tag, u):
        """
        Calculates the outlet flux of a given variable across a specified tag.

        Parameters:
        - tag: The tag representing the outlet boundary.
        - u: The variable to calculate the flux for.

        Returns:
        - total_flux: The total outlet flux of the variable across the tag.
        """
        flux = dot(u, self.mesh.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux
    
    def mean_pressure_interface(self,tag,p):
        """
        Calculates the mean pressure across a specified interface.

        Parameters:
        - tag: The tag representing the interface.
        - p: The pressure variable.

        Returns:
        - mean_p: The mean pressure across the interface.
        """
        mean_p = p*self.mesh.dS(tag)
        length = assemble(Constant(1.0)*self.mesh.dS(tag))
        mean_p = assemble(mean_p)
        return mean_p/length

    def mean_pressure_boundaries(self,tag,p):
        """
        Calculates the mean pressure across a specified boundary.

        Parameters:
        - tag: The tag representing the boundary.
        - p: The pressure variable.

        Returns:
        - mean_p: The mean pressure across the boundary.
        """
        mean_p = p*self.mesh.ds(tag)
        length = assemble(Constant(1.0)*self.mesh.ds(tag))
        mean_p = assemble(mean_p)
        return mean_p/length
            
    def td_nodes_data(self):
        """
        Calculates the time-dependent data for the nodes.

        Returns:
        - td_dict_u: A dictionary containing the time-dependent data for the velocity.
        - td_dict_p: A dictionary containing the time-dependent data for the pressure.
        """
        td_dict_u = {}
        td_dict_p = {}
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

    def save_graph(self,fields_names=['flux','pressure'], output_dir = "data/graphs/"):
        """
        Saves the graphs of specified fields.

        Parameters:
        - fields_names: A list of field names to save the graphs for. Default is ['flow_rate', 'pressure'].
        - output_dir: The directory to save the graphs in. Default is "data/graphs/".

        Returns:
        - The result of the super class's save_graph method.
        """
        return super().save_graph(fields_names,output_dir)
    

# Define a subclass DataHeat that inherits from the DataGenerator abstract base class
class DataHeat(DataGenerator):
    """
    This class represents a data generator for solving heat transfer problems.
    It inherits from the DataGenerator class.
    """

    def __init__(self, solver, mesh):
        """
        Constructor for the DataHeat class.

        Parameters:
        - solver: The solver object for heat transfer equations.
        - mesh: The mesh object representing the computational domain.
        """
        super().__init__(solver, mesh)

    
    def flux(self, interface, u):
        """
        Calculates the heat flux across an interface.

        Parameters:
        - interface: The interface tag.
        - u: The temperature field.

        Returns:
        - total_flux: The total heat flux across the interface.
        """
        flux = -self.solver.k * dot(grad(u)('+'), self.mesh.n('+')) * self.mesh.dS(interface)
        total_flux = assemble(flux)
        return total_flux
    
    def inlet_flux(self, tag, u):
        """
        Calculates the heat flux at an inlet boundary.

        Parameters:
        - tag: The boundary tag.
        - u: The temperature field.

        Returns:
        - total_flux: The total heat flux at the inlet boundary.
        """
        flux = self.solver.k * dot(grad(u), self.mesh.n) * self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux

    def td_nodes_data(self):
        """
        Calculates the time-dependent data for the nodes.

        Returns:
        - td_dict: A dictionary containing the time-dependent data for the temperature.
        """
        td_dict = {}
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
    
    def save_graph(self, fields_names=['flux'], output_dir="data/graphs/"):
        """
        Saves the graphs of specified fields.

        Parameters:
        - fields_names: A list of field names to save the graphs for. Default is ['flux'].
        - output_dir: The directory to save the graphs in. Default is "data/graphs/".

        Returns:
        - The result of the super class's save_graph method.
        """
        return super().save_graph(fields_names, output_dir)
