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
        self.k = Constant(1.0)
        self.f = Constant(12 - 2 - 2*30)
        self.bc = Expression('1+x[0]*x[0]+30*x[1]*x[1]+12*t',t=0,degree=2)
        self.u0 = self.bc
        self.dt = Constant(0.3)
        self.T = 1.8

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
        for i in self.mesh.tags['walls']:
            bcs.append(DirichletBC(self.V,self.bc,self.mesh.bounds,i))


        while(t<=self.T):
            #Solve
            self.bc.t=t
            solve(a==L,U,bcs)
            #Update
            u0.assign(U)
            t+=float(self.dt)

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
    
class DataHeat(DataGenerator):

    def __init__(self, solver, mesh):
        super().__init__(solver, mesh)

    def flux(self,interface):
        flux = -dot(grad(self.solver.u)('+'), self.n('+'))*self.mesh.dS(interface)
        total_flux = assemble(flux)
        return total_flux
    
    def mean_temp(self,domain):
        mean_temp = self.solver.u*self.mesh.dx(domain)
        area = assemble(Constant(1.0)*self.mesh.dx(domain))
        mean_temp = assemble(mean_temp)
        return mean_temp/area
    

        