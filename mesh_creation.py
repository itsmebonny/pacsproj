import gmsh
import numpy as np
import argparse

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

np.random.seed(args.seed)
for it in range(args.nmesh):
    # Initialize Gmsh
    gmsh.initialize()

    # Create a new model
    model = gmsh.model
    lc = args.lc
    w = args.wmax
    points = []
    for i in range(args.nodes):
        h = round(np.random.uniform(args.hmin, args.hmax),2)
        # if args.spacing:
        #     w = round(np.random.uniform(args.wmin, args.wmax),2)
        points += [gmsh.model.geo.addPoint(i*w, -h, 0, lc)]
        points += [gmsh.model.geo.addPoint(i*w, h, 0, lc)]

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
    gmsh.model.addPhysicalGroup(1, list_lines_x, 1) # horizontal
    gmsh.model.addPhysicalGroup(1, [list_lines_y[0]], 2) # left wall
    gmsh.model.addPhysicalGroup(1, [list_lines_y[-1]], 3) # right wall
    for i in range(1,len(list_lines_y)-1): # interfaces
        model.addPhysicalGroup(1, [list_lines_y[i]], i+3)
    for j in range(len(faces)): # faces
        gmsh.model.addPhysicalGroup(2, [faces[j]], i+j+4)

    gmsh.model.geo.synchronize()
    # Generate mesh:
    gmsh.model.mesh.generate()

    # Write mesh data:
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.write(f"data/mesh_long/RandomMesh_{it}.msh")

    # Creates graphical user interface
    #if 'close' not in sys.argv:
    gmsh.fltk.run()

    # It finalize the Gmsh API
    gmsh.finalize()


