import gmsh

# Initialize Gmsh
gmsh.initialize()

# Create a new model
model = gmsh.model
lc = 1
points = []
inlet = []
interface = []
outlet = []
walls = []
faces = []

l1 = 4
l2 = 10

points += [model.geo.addPoint(l1/2, l2, 0, lc)]
points += [model.geo.addPoint(-l1/2, l2, 0, lc)]
points += [model.geo.addPoint(-l1/2, 0, 0, lc)]
points += [model.geo.addPoint(0, 0, 0, lc)]
points += [model.geo.addPoint(l1/2, 0, 0, lc)]
model.geo.synchronize()

inlet.append(model.geo.addLine( points[0], points[1] ))
walls.append(model.geo.addLine( points[1], points[2] ))
outlet.append(model.geo.addLine( points[2], points[3] ))
outlet.append(model.geo.addLine( points[3], points[4] ))
walls.append(model.geo.addLine( points[4], points[0] ))


face = model.geo.addCurveLoop([inlet[0], walls[0], outlet[0], outlet[1],walls[1]])
faces.append(face)
model.geo.addPlaneSurface([face])

def add_bifurcation(model,interface,outlet,faces,point1_tag,point2_tag, mid_point_tag, drop,points):

    interface.append(outlet[-2])
    interface.append(outlet[-1])
    outlet.pop()
    outlet.pop()
    
    point1 = model.getValue(0,point1_tag,[])
    point2 = model.getValue(0,point2_tag,[])
    mid_point = model.getValue(0,mid_point_tag,[])

    # left bifurcation
    model.geo.synchronize()
    
    points_left = [mid_point_tag,point1_tag]
    points_left += [model.geo.addPoint(point1[0]-drop, point1[1]-drop, 0, lc)]
    points_left += [model.geo.addPoint((point1[0]+mid_point[0])/2-drop, (point1[1]+mid_point[1])/2-drop, 0, lc)]
    points_left += [model.geo.addPoint(mid_point[0]-drop, mid_point[1]-drop, 0, lc)]

    walls_left = []
    walls_left.append(model.geo.addLine( points_left[1], points_left[2] ))
    outlet.append(model.geo.addLine( points_left[2], points_left[3] ))
    outlet.append(model.geo.addLine( points_left[3], points_left[4] ))
    walls_left.append(model.geo.addLine( points_left[4], points_left[0] ))

    face = model.geo.addCurveLoop([-interface[-2], walls_left[0], outlet[-2], outlet[-1], walls_left[1]])
    faces.append(face)
    model.geo.addPlaneSurface([face])

    #right bifurcation 

    points_right = [point2_tag,mid_point_tag]
    points_right += [model.geo.addPoint(mid_point[0]+drop, mid_point[1]-drop, 0, lc)]
    points_right += [model.geo.addPoint((point2[0]+mid_point[0])/2+drop, (point2[1]+mid_point[1])/2-drop, 0, lc)]
    points_right += [model.geo.addPoint(point2[0]+drop, point2[1]-drop, 0, lc)]

    walls_right = []
    walls_right.append(model.geo.addLine( points_right[1], points_right[2] ))
    outlet.append(model.geo.addLine( points_right[2],points_right[3] ))
    outlet.append(model.geo.addLine( points_right[3], points_right[4] ))
    walls_right.append(model.geo.addLine( points_right[4], points_right[0] ))

    face = model.geo.addCurveLoop([-interface[-1], walls_right[0], outlet[-2], outlet[-1], walls_right[1]])
    faces.append(face)
    model.geo.addPlaneSurface([face])

    points += points_left[2:]
    points += points_right[2:]
    model.geo.synchronize()

    
add_bifurcation(model,interface,outlet,faces,points[2],points[4],points[3],8,points)
add_bifurcation(model,interface,outlet,faces,points[8],points[10],points[9],8,points)

# #physical group
model.addPhysicalGroup(1, inlet, 1) 
model.addPhysicalGroup(1, walls, 2) 
model.addPhysicalGroup(1, outlet, 3) 
model.addPhysicalGroup(2, faces, 4)
for i in range(len(interface)):
    model.addPhysicalGroup(1, [interface[i]], i+4)

# Create the relevant Gmsh data structures from Gmsh model.
model.geo.synchronize()

# Generate mesh:
model.mesh.generate()

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.write(f"data/mesh/bifurcation.msh")

# Creates graphical user interface
#if 'close' not in sys.argv:
gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()
