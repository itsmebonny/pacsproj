from dolfin import *
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import GenerateGraph as gg
import torch as th

import matplotlib as mpl  #temporaneo


class MeshLoader:

    def __init__(self,filename):
        # store the input mesh file name
        self.meshfile = filename

        # load the mesh from xml file 
        self.mesh = Mesh(self.meshfile + ".xml")

        # create a MeshFunction for boundaries and physical regions
        self.bounds = MeshFunction("size_t", self.mesh, self.meshfile + "_facet_region.xml")
        self.face = MeshFunction("size_t", self.mesh, self.meshfile + "_physical_region.xml")
    
    
    def update_tags(self,tags):
        """
        Method to save the tags of the boundaries and faces of the mesh

        The method takes as input a dictionary with the following keys: 
        'walls', 'inlet', 'outlet', 'interface', 'faces'
        and as values a list of the corresponding tags
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

# class Stokes(Solver):

#     def __init__(self, mesh):
#         super().__init__(mesh)
#         self.V = VectorFunctionSpace(self.mesh.mesh,"P", 2)
#         self.Q = FunctionSpace(self.mesh.mesh,"P", 1)
#         self.rho = 1*1e3
#         self.mu = 4*1e-3
#         self.U0 = 0.1
#         self.L0 = 0.1
#         self.inflow = Expression(("-1.0/4 * x[1] * x[1] + 1", "0.0"), degree=0)
#         self.dt = 0.5
#         self.T = 10
#         self.f = Constant((0.0, 0.0))

#     def set_parameters(self,V,Q,rho,mu,U0,L0,inflow,f):
#         self.V = V
#         self.Q = Q
#         self.rho = rho
#         self.mu = mu
#         self.U0 = U0
#         self.L0 = L0
#         self.inflow = inflow
#         self.f = f

#     def solve(self):

#         u = TrialFunction(self.V)
#         p = TrialFunction(self.Q)
#         v = TestFunction(self.V)
#         q = TestFunction(self.Q)

#         # No-slip boundary condition for velocity
#         bcsu = []
#         bcsp = []
#         noslip = Constant((0.0, 0.0))
#         for i in self.mesh.tags['walls']:
#             bcsu.append(DirichletBC(self.V, noslip, self.mesh.bounds, i))
        
#         # Inflow boundary condition for velocity
#         for i in self.mesh.tags['inlet']:
#             bcsu.append(DirichletBC(self.V, self.inflow, self.mesh.bounds, i))
#         for i in self.mesh.tags['outlet']:
#             bcsp.append(DirichletBC(self.Q, Constant(0.0), self.mesh.bounds, i))
#         bcs = bcsu + bcsp
#         # Create functions
#         # u0 = Function(self.V)
#         u1 = Function(self.V)
#         p1 = Function(self.Q)
        

#         Re = Constant(self.rho * self.U0 * self.L0 / self.mu)

#         TH = self.V *self.Q
#         W = FunctionSpace(self.mesh.mesh, TH)
#         w0 = Function(W)
#         a0 = (1/Re)*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx 
#         L0 = inner(self.f,v)*dx 
#         solve(a0 == L0, w0, bcs)
#         u0, p0 = w0.split()

#         # Step 1
#         a1 = (1/self.dt)*inner(u,v)*dx + (1/Re)*inner(grad(u), grad(v))*dx + inner(dot(grad(u),u0),v)*dx # potrebbe essere dot(grad(u),u0)
#         L1 = (1/self.dt)*inner(u0,v)*dx + inner(self.f,v)*dx 
        
#         # step 2
#         a2 = inner(grad(p), grad(q))*dx
#         L2 = -(1/self.dt)*div(u1)*q*dx

#         # Velocity update
#         a3 = inner(u, v)*dx
#         L3 = inner(u1, v)*dx - self.dt*inner(grad(p1), v)*dx

#         # Assemble matrices
#         A1 = assemble(a1)
#         A2 = assemble(a2)
#         A3 = assemble(a3)

#         # Use amg preconditioner if available
#         prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

#         # Time-stepping
#         t = self.dt
#         self.ut = []
#         self.pt = []
#         while t < self.T + DOLFIN_EPS:

#             # Update pressure boundary condition
#             self.inflow.t = t

#             # Compute tentative velocity step
         
#             b1 = assemble(L1)
#             [bc.apply(A1, b1) for bc in bcsu]
#             solve(A1, u1.vector(), b1, "gmres", "default")
            

#             # Pressure correction
            
#             b2 = assemble(L2)
#             [bc.apply(A2, b2) for bc in bcsp]
#             solve(A2, p1.vector(), b2, "cg", prec)
            

#             # Velocity correction
   
#             b3 = assemble(L3)
#             [bc.apply(A3, b3) for bc in bcsu]
#             solve(A3, u1.vector(), b3, "gmres", "default")

#             #norm = mpl.colors.Normalize(vmin=0.,vmax=15.,clip=False)
#             vel = plot(u1, title = 'Velocity')#, norm=norm)
#             plt.colorbar(vel,label='m/s')
#             plt.show()
#             pres = plot(p1, title = 'Pressure')
#             plt.colorbar(pres,label='Pa')
#             plt.show()

#             self.ut.append(u1*self.U0)
#             self.pt.append(p1*self.rho*self.U0*self.U0) #non va dovrebbe essere pt.vector()[:] = p1.vector()[:]*self.rho*self.U0*self.U0 e dovremmo salvarlo in un array

#             # Move to next time step
#             u0.assign(u1)
#             t += self.dt

#     def plot_solution(self):
#         vel = plot(self.ut[-1], title = 'Velocity')
#         plt.colorbar(vel,label='m/s')
#         plt.show()
#         pres = plot(self.pt[-1], title = 'Pressure')
#         plt.colorbar(pres,label='Pa')
#         plt.show()
        


class Stokes(Solver):

    def __init__(self, mesh):
        super().__init__(mesh)
        self.V = VectorElement("P", self.mesh.mesh.ufl_cell(), 2)
        self.Q = FiniteElement("P", self.mesh.mesh.ufl_cell(), 1)
        self.rho = 1*1e3
        self.mu = 4*1e-6
        self.U0 = 0.001
        self.L0 = 0.001
        self.inflow = Expression(("(-1.0/4.0*x[1]*x[1] + 1)", " 0.0 "), degree=2)
        self.dt = 0.5
        self.T = 10
        self.f = Constant((0.0, 0.0))

    def set_parameters(self,V,Q,rho,mu,U0,L0,inflow, f, dt, T):
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

        self.k = self.rho * self.U0 * self.L0 / self.mu
        self.k = round(self.k,2)
        a = (1/self.dt)*inner(u,v)*dx + (1/self.k)*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx
        L = inner(f,v)*dx + (1/self.dt)*inner(u0,v)*dx

        U = Function(W)
        t = self.dt
        self.ts = []
        self.ut = []
        self.pt = []
        while(t<self.T):
            solve(a == L, U, bcs)
            u, p = U.split()
            fa.assign([u0,p0],U)
            # plot(u)
            # plt.colorbar(plot(u))
            # plt.show()
            temp1 = Function(P)
            temp1.vector()[:] = u0.vector()[:]
            self.ut.append(temp1)
            temp2 = Function(Z)
            temp2.vector()[:] = p0.vector()[:]
            self.pt.append(temp2)
            self.ts.append(t)

            t+=self.dt

        self.u = u*self.U0
        self.p = p*self.rho*self.U0*self.U0

    def plot_solution(self):
        pass


# Define a subclass Heat that inherits from the Solver abstract base class to solve Heat equation
class Heat(Solver):

    def __init__(self, mesh, V, k, f, u0, dt, T, g, doplot=False):
        super().__init__(mesh)
        self.V = V # FunctionSpace
        self.k = k # Thermal conductivity
        self.f = f # Source term
        self.u0 = u0 # Initial condition
        self.dt = dt # Time step
        self.T = T # Final time
        self.g = g # Neumann boundary condition at inlet
        self.doplot = doplot # Plot solution at each time step

    # Method to set different parameters 
    def set_parameters(self,V,k,f,u0,dt,T,g):
        self.V = V
        self.k = k
        self.f = f
        self.u0 = u0
        self.dt = dt
        self.T = T
        self.g = g

    def solve(self):

        """
        Method to solve the Heat equation.

        The problem is solved using Discontinuous Galerkin method 
        and imposing non-homogeneuos Neumann boundary condition at the inlet 
        and homogeneous Neumann boundary condition at the outlet and walls.
        """

        t = float(self.dt)
        u0 = interpolate(self.u0,self.V)
        U = Function(self.V)

        # Define variational problem
        u=TrialFunction(self.V)
        v=TestFunction(self.V)
        h = self.mesh.mesh.hmin()
        n = FacetNormal(self.mesh.mesh)
        a_int=u*v*dx+self.dt*self.k*inner(grad(u),grad(v))*dx 
        a_facet = self.k*(10/avg(h)*dot(jump(v,n),jump(u,n))*dS - dot(avg(grad(v)), jump(u, n))*dS - dot(jump(u, n), avg(grad(v)))*dS)

        a = a_int + a_facet
        L=u0*v*dx+self.dt*self.f*v*dx + self.g*v*self.dt*self.mesh.ds(self.mesh.tags['inlet'][0])

        # lists to store the solution at each time step and the corresponding time
        # possiamo usare degli array? 
        self.ut = []
        self.ts = []

        # Solve the heat equation at each time step
        while(t<=self.T):
            temp = Function(self.V)
            temp.vector()[:] = u0.vector()[:]
            self.ut.append(temp)
            self.g.t=t
            solve(a==L,U)

            # Update
            u0.assign(U)
            self.ts.append(t)
            t+=float(self.dt)

            # Plot solution at each time step
            if self.doplot:
                self.plot_solution(U)

        return self.ut 


    # Method to plot the solution
    def plot_solution(self,u):
        sol = plot(u)
        plot(u)
        plt.colorbar(sol)
        plt.show()
           

class DataGenerator(ABC):
    
    def __init__(self, solver, mesh):
        self.solver = solver
        self.mesh = mesh
        self.n = FacetNormal(self.mesh.mesh)
        self.h = self.mesh.mesh.hmin()

    @abstractmethod
    def flux(self):
        pass

    @abstractmethod
    def inlet_flux(self,tag, u):
        pass

    def area(self,domain, u):
        area = assemble(Constant(1.0)*self.mesh.dx(domain))
        return area

    def create_edges(self):
        if not hasattr(self, 'NodesData'):
            self.nodes_data()
        self.edges1 = self.NodesData['NodeId'][:-1]
        self.edges2 = self.NodesData['NodeId'][1:]
        return self.edges1,self.edges2
    
    def edges_data(self):
        if not hasattr(self, 'NodesData'):
            self.nodes_data()
        dict_e = {'edgeId':[], 'area':[], 'length':[]}
        for i in range(len(self.NodesData['NodeId'])-1):
            dict_e['edgeId'].append(i)
        
        for i in self.mesh.tags['faces']:
            dict_e['area'].append(self.area(i,self.solver.ut[0]))

        for j in range(len(self.center_line)-1):
            dict_e['length'].append(self.center_line[j+1][0]-self.center_line[j][0])
        self.EdgesData = dict_e
        return dict_e

    def nodes_data(self):
        if not hasattr(self, 'center_line'):
            self.centerline()
        dict = {'k':[], 'NodeId':[], 'inlet_mask':[], 'outlet_mask':[], 'interface_length': []}
        for j in range(len(self.center_line)):
                dict['NodeId'].append(j)
                if j == 0:
                    dict['inlet_mask'].append(1)
                    dict['outlet_mask'].append(0)
                elif j == len(self.center_line)-1:
                    dict['inlet_mask'].append(0)
                    dict['outlet_mask'].append(1)
                else:
                    dict['inlet_mask'].append(0)
                    dict['outlet_mask'].append(0)
                dict['k'].append(self.solver.k)
        for i in self.mesh.tags['inlet']:
            dict['interface_length'].append(round(assemble(Constant(1.0)*self.mesh.ds(i)),2)) 
        for i in self.mesh.tags['interface']:
            dict['interface_length'].append(round(assemble(Constant(1.0)*self.mesh.dS(i)),2))  
        for i in self.mesh.tags['outlet']:
            dict['interface_length'].append(round(assemble(Constant(1.0)*self.mesh.ds(i)),2))  

        self.NodesData = dict
        return dict

    @abstractmethod
    def td_nodes_data(self):
        pass

    # Method to calculate the centerline coordinates of the mesh, which are the coordinates of the graph nodes
    def centerline(self):
        center_line = [] # dovrebbe essere un array di array (comunemente detto lista di liste)
        tags_list = ['inlet','interface','outlet']
        for j in tags_list:
            for i in self.mesh.tags[j]:
                edge_coord =[]
                # Extract the coordinates of the vertices for each edge with the specified tag
                for edge in edges(self.mesh.mesh):
                    if self.mesh.bounds.array()[edge.index()] == i:
                        for vertex in vertices(edge):
                           coordinate = vertex.point().array()
                           edge_coord.append(coordinate)

                edge_coord = np.array(edge_coord)
                # Calculate the midpoint of the edge and append to the centerline list
                center_line.append([(np.max(edge_coord[:,0])+np.min(edge_coord[:,0]))/2,(np.max(edge_coord[:,1])+np.min(edge_coord[:,1]))/2])
        self.center_line = center_line
        self.NNodes = len(center_line)
        return center_line

    def save_graph(self, fields_names, output_dir = "data/graphs/"):
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
        print(fields_names)
        for i,key in enumerate(fields_names):
            gg.add_field(self.graph, self.TDNodesData[i], key)
        gg.save_graph(self.graph, f"k_{self.solver.k}", output_dir)
        return self.graph

class DataNS(DataGenerator):

    def __init__(self, solver, mesh):
        super().__init__(solver, mesh)

    def flux(self,tag,u):
        flux = dot(u, self.n('+'))*self.mesh.dS(tag)
        total_flux = assemble(flux)
        return total_flux

    def inlet_flux(self,tag, u):
        flux = -dot(u, self.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux

    def outlet_flux(self,tag, u):
        flux = dot(u, self.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux
    
    def mean_pressure_interface(self,tag,p):
        mean_p = p*self.mesh.dS(tag)
        length = assemble(Constant(1.0)*self.mesh.dS(tag))
        mean_p = assemble(mean_p)
        return mean_p/length

    def mean_pressure_boundaries(self,tag,p):
        mean_p = p*self.mesh.ds(tag)
        length = assemble(Constant(1.0)*self.mesh.ds(tag))
        mean_p = assemble(mean_p)
        return mean_p/length
            
    def td_nodes_data(self):
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

    def save_graph(self,fields_names=['flow_rate','pressure'], output_dir = "data/graphs/"):
        return super().save_graph(fields_names,output_dir)
    
    # for further improvements 

# Define a subclass DataHeat that inherits from the DataGenerator abstract base class
class DataHeat(DataGenerator):

    def __init__(self, solver, mesh):
        super().__init__(solver, mesh)

    
    def flux(self,interface, u):
        flux = -self.solver.k*dot(grad(u)('+'), self.n('+'))*self.mesh.dS(interface)
        total_flux = assemble(flux)
        return total_flux
    
    def inlet_flux(self,tag, u):
        flux = self.solver.k*dot(grad(u), self.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux
    
    def area(self,domain, u):
        area = assemble(Constant(1.0)*self.mesh.dx(domain))
        return area

    def td_nodes_data(self):
        td_dict = {}
        for t in range(len(self.solver.ts)):
            
            td_dict[self.solver.ts[t]] = np.zeros(self.NNodes)
            it = 0
            for i in self.mesh.tags['inlet']:
                td_dict[self.solver.ts[t]][it] = self.inlet_flux(i,self.solver.ut[t])
                it+=1
            for i in self.mesh.tags['interface']:
                td_dict[self.solver.ts[t]][it] = self.flux(i,self.solver.ut[t])  
                it+=1    
            # for i in self.mesh.tags['outlet']:
            #     td_dict[self.solver.ts[t]].append(0.0)

        self.TDNodesData = [td_dict]
        return td_dict
    
    def save_graph(self, fields_names=['flux'],output_dir = "data/graphs/"):
        return super().save_graph(fields_names,output_dir)