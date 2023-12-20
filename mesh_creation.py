import gmsh
import numpy as np

for it in range(20):
    # Initialize Gmsh
    gmsh.initialize()

    # Create a new model
    model = gmsh.model
    lc = 1
    points = []
    #center_points = []
    for i in range(5):
        h = round(np.random.uniform(0.5, 5),2)
        points += [gmsh.model.geo.addPoint(i*5, -h, 0, lc)]
        #center_points += [gmsh.model.geo.addPoint(i*5, 2.5, 0, lc)]
        points += [gmsh.model.geo.addPoint(i*5, h, 0, lc)]

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

    #list_lines_y.append(gmsh.model.geo.addLine( points[1], points[0] ))
    for i in range(0,len(points)-1,2):
        list_lines_y.append(gmsh.model.geo.addLine( points[i], points[i+1] ))
    #list_lines_y.append(gmsh.model.geo.addLine( points[-2], points[-1] ))

    faces = []
    print(len(list_lines_x))
    # faces of cube:
    for i in range(0,len(list_lines_x)-1,2):
        face = gmsh.model.geo.addCurveLoop([-list_lines_y[i//2], list_lines_x[i], list_lines_y[i//2+1], list_lines_x[i+1]])
        faces.append(face)
        gmsh.model.geo.addPlaneSurface([face])

    # surfaces of cube:

    #physical group
    gmsh.model.addPhysicalGroup(1, list_lines_x, 5) # horizontal
    gmsh.model.addPhysicalGroup(1, [list_lines_y[0]], 6) # left wall
    gmsh.model.addPhysicalGroup(1, [list_lines_y[-1]], 7) # right wall
    for i in range(1,len(list_lines_y)-1): # interfaces
        model.addPhysicalGroup(1, [list_lines_y[i]], i+7)
    for i in range(len(faces)): # faces
        gmsh.model.addPhysicalGroup(2, [faces[i]], i+11)

    gmsh.model.geo.synchronize()
    # Generate mesh:
    gmsh.model.mesh.generate()

    # Write mesh data:
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.write(f"data/mesh/DDrandom_{it}.msh")

    # Creates graphical user interface
    #if 'close' not in sys.argv:
    gmsh.fltk.run()

    # It finalize the Gmsh API
    gmsh.finalize()


