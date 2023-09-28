from dolfin import *
import matplotlib.pyplot as plt

def solve_stokes(fd,mesh):
    #define element spaces
    P2 = VectorElement("P", mesh.ufl_cell(), 2)
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    TH = P2* P1
    W = FunctionSpace(mesh, TH)

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0))
    bc1 = DirichletBC(W.sub(0), noslip, fd, 2)
    
    # Inflow boundary condition for velocity
    inflow = Expression(("0.0", " -( 0.5 * x[0] * x[0] + 8.0) "), degree=2)
    #inflow = Expression(("0.0", " 10000000000000 "), degree=2)
    bc0 = DirichletBC(W.sub(0), inflow, fd, 1)
    bc_outlet = DirichletBC(W.sub(1), Constant(0.0), fd, 3)

    # Collect boundary condition
    bcs = [bc0, bc1]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0, 0))

    Re1 = Constant(5*36/0.035)
    Re2 = Constant(32226)
    Re = Expression('(x[0] < 0) ? Re1 : Re2',Re1=Re1,Re2=Re2,degree=1)
    delta = 1
    a = (1/Re)*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx 
    L = inner(f,v+ delta*grad(q))*dx #+inner(g,v)*ds # in case of emergency: + delta*grad(q)

    U = Function(W)
    solve(a == L, U, bcs,solver_parameters={"linear_solver": "mumps"})
    u, p = U.split()

    return u,p

def plot_solution(u,p):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4)) # 1 riga, 2 colonne
    vv = axs[0].imshow(u*36)
    cbar = plt.colorbar(vv, ax=axs[0])
    cbar.ax.set_ylabel('cm/s') 
    axs[0].set_title('Velocity') 
    pp = axs[1].imshow(p*36*36*10)
    cbar = plt.colorbar(pp, ax=axs[1])
    cbar.ax.set_ylabel('Pa') 
    axs[1].set_title('Pressure') 
    plt.show()