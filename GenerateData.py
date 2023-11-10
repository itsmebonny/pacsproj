from dolfin import *
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class MeshLoader:

    def __init__(self,filename):
        self.meshfile = filename
        self.mesh = Mesh(self.meshfile + ".xml")
        self.bounds = MeshFunction("size_t", self.mesh, self.meshfile + "_facet_region.xml")
        self.face = MeshFunction("size_t", self.mesh, self.meshfile + "_physical_region.xml")
    
    def update_tags(self,tags):
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

    def measure_definition(self):
        self.dS = Measure("dS",domain=self.mesh, subdomain_data=self.rename_boundaries)
        self.ds = Measure("ds",domain=self.mesh, subdomain_data=self.rename_boundaries)
        self.dx = Measure("dx",domain=self.mesh, subdomain_data=self.rename_faces)

        return self.dS, self.ds, self.dx

            
class Solver(ABC):
    
    def __init__(self,mesh,equation):
        self.mesh = mesh
        self.equation = equation

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

    def __init__(self, mesh, equation):
        super().__init__(mesh, equation)
        self.V = VectorElement("P", self.mesh.mesh.ufl_cell(), 2)
        self.Q = FiniteElement("P", self.mesh.mesh.ufl_cell(), 1)
        self.rho = 1*1e3
        self.mu = 4*1e-3
        self.U0 = 0.1
        self.L0 = 0.01
        self.inflow = Expression(("0.0", " -( 0.25 * x[0] * x[0] + 9.0) "), degree=2)

    def set_parameters(self,V,Q,rho,mu,U0,L0,inflow):
        self.V = V
        self.Q = Q
        self.rho = rho
        self.mu = mu
        self.U0 = U0
        self.L0 = L0
        self.inflow = inflow

    # potremmo creare solve CG e solve DG
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

        Re = Constant(self.rho * self.U0 * self.L0 / self.mu)
        a = (1/Re)*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx 
        L = inner(f,v)*dx 

        U = Function(W)

        solve(a == L, U, bcs)
        u, p = U.split()

        self.u = u*self.U0
        self.p = p*self.rho*self.U0*self.U0

    def plot_solution(self):
        vel = plot(self.u, title = 'Velocity')
        plt.colorbar(vel,label='m/s')
        plt.show()
        pres = plot(self.p, title = 'Pressure')
        plt.colorbar(pres,label='Pa')
        plt.show()
        

class Heat(Solver):

    def __init__(self, mesh, equation):
        super().__init__(mesh, equation)
        self.V = FunctionSpace(self.mesh.mesh,"CG",1)
        self.k = Constant(1000.0)
        self.f = Constant(12 - 2 - 2*3000)
        self.bc = Expression('1+x[0]*x[0]+3000*x[1]*x[1]+12*t',t=0,degree=2)
        self.u0 = self.bc
        self.dt = Constant(30)
        self.T = 100

    # exact solution u(x,y,t) = 1 + x^2 + alpha*y^2 + beta*t
    # source f(x,y,t) = beta - 2 - 2*alpha
    # alpha = 30, beta = 12
    # bcs are the exact solution at time t

    def set_parameters(self,V,k,f,bc,u0,dt,T):
        self.V = V
        self.k = k
        self.f = f
        self.bc = bc
        self.u0 = u0
        self.dt = dt
        self.T = T
        

    # potremmo creare solve CG e solve DG
    def solve(self):
        t = float(self.dt)
        u0 = interpolate(self.u0,self.V)
        U = Function(self.V)

        #Variationalproblemateachtime
        u=TrialFunction(self.V)
        v=TestFunction(self.V)
        a_int=u*v*dx+self.dt*self.k*inner(grad(u),grad(v))*dx 
        #a_facet = 10/avg(h)*dot(jump(v,n),jump(u,n))*dS - dot(avg(grad(v)), jump(u, n))*dS - dot(jump(u, n), avg(grad(v)))*dS
        a = a_int # + a_facet
        L=u0*v*dx+self.dt*self.f*v*dx

        bcs = []
        tags_list = ['walls','inlet','outlet']
        for j in tags_list:
            for i in self.mesh.tags[j]:
                bcs.append(DirichletBC(self.V,self.bc,self.mesh.bounds,i))

        self.ut = []
        self.ts = []
        while(t<=self.T):
            #Solve
            self.ut.append(u0)
            self.bc.t=t
            solve(a==L,U,bcs)
            #Update
            u0.assign(U)
            self.ts.append(t)
            t+=float(self.dt)
            sol= plot(U)
            plt.colorbar(sol)
            plt.show()

        self.u = U

    def plot_solution(self):
        sol = plot(self.u)
        plot(self.u)
        plt.colorbar(sol)
        plt.show()
           

class DataGenerator(ABC):
    
    def __init__(self, solver, mesh):
        self.solver = solver
        self.mesh = mesh
        self.n = FacetNormal(self.mesh.mesh)
        self.h = self.mesh.mesh.hmin()
        #da pensare come gestire più soluzioni se vuoi pensarci

    @abstractmethod
    def flux(self):
        pass

    def centerline(self):
        center_line = []
        tags_list = ['inlet','interface','outlet']
        for j in tags_list:
            for i in self.mesh.tags[j]:
                edge_coord =[]
                for edge in edges(self.mesh.mesh):
                    if self.mesh.bounds.array()[edge.index()] == i:
                        for vertex in vertices(edge):
                           coordinate = vertex.point().array()
                           edge_coord.append(coordinate)

                edge_coord = np.array(edge_coord)
                center_line.append([(np.max(edge_coord[:,0])+np.min(edge_coord[:,0]))/2,(np.max(edge_coord[:,1])+np.min(edge_coord[:,1]))/2])
        self.center_line = center_line
        return center_line

class DataNS(DataGenerator):

    def __init__(self, solver, mesh):
        super().__init__(solver, mesh)

    def flux(self,interface):
        flux = dot(self.solver.u, self.n('+'))*self.mesh.dS(interface)
        total_flux = assemble(flux)
        return total_flux
    
    def mean_pressure(self,interface):
        mean_p = self.solver.p*self.mesh.dS(interface)
        length = assemble(Constant(1.0)*self.mesh.dS(interface))
        mean_p = assemble(flux)
        return mean_p/length
    
    def centerline(self):
        return super().centerline()

    
class DataHeat(DataGenerator):

    def __init__(self, solver, mesh):
        super().__init__(solver, mesh)

    def flux(self,interface, u):
        flux = -dot(grad(u)('+'), self.n('+'))*self.mesh.dS(interface)
        total_flux = assemble(flux)
        return total_flux
    
    def boundary_flux(self,tag, u):
        flux = -dot(grad(u), self.n)*self.mesh.ds(tag)
        total_flux = assemble(flux)
        return total_flux
    
    def mean_temp(self,domain, u):
        mean_temp = u*self.mesh.dx(domain)
        area = assemble(Constant(1.0)*self.mesh.dx(domain))
        mean_temp = assemble(mean_temp)
        return mean_temp/area
    
    def centerline(self):
        return super().centerline()
    
    def nodes_data(self):
        dict = {'NodeId':[]}
        td_dict = {}
        for t in range(len(self.solver.ts)):
            td_dict[self.solver.ts[t]] = []
            for i in self.mesh.tags['inlet']:
                td_dict[self.solver.ts[t]].append(self.boundary_flux(i,self.solver.ut[t]))
            for i in self.mesh.tags['interface']:
                td_dict[self.solver.ts[t]].append(self.flux(i,self.solver.ut[t]))
            for i in self.mesh.tags['outlet']:
                td_dict[self.solver.ts[t]].append(self.boundary_flux(i,self.solver.ut[t]))
        for j in range(len(self.center_line)):
                dict['NodeId'].append(j)
        self.NodesData = dict
        self.TDNodesData = td_dict
        return dict, td_dict
    
    
    def create_edges(self):
        self.edges1 = self.NodesData['NodeId'][:-1]
        self.edges2 = self.NodesData['NodeId'][1:]

        return self.edges1,self.edges2