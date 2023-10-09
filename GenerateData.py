from dolfin import *
import numpy as np

class MeshLoader:

    def __init__(self,filename,equation):
        self.meshfile = filename
        self.mesh = Mesh(self.meshfile + ".xml")
        self.bounds = MeshFunction("size_t", self.mesh, self.meshfile + "_facet_region.xml")
        self.face = MeshFunction("size_t", self.mesh, self.meshfile + "_physical_region.xml")
        self.equation = equation
    
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
        dS = Measure("dS",domain=self.mesh, subdomain_data=self.rename_boundaries)
        ds = Measure("ds",domain=self.mesh, subdomain_data=self.rename_boundaries)
        dx = Measure("dx",domain=self.mesh, subdomain_data=self.rename_faces)

        return dS, ds, dx

            
    