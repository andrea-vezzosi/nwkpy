import numpy as np
import copy
from nwkpy.fem import FemSpace
from nwkpy.physics import DopingChargeDensity, ElectrostaticPotential
from nwkpy.fem.solver import LinearSystem
from nwkpy.fem.problem import Poisson
from nwkpy import _constants


class PoissonProblem:
    def __init__(self, 
                mesh,
                shape_class_name="LagrangeQuadratic",
                dirichlet=None,
                electric_field=(0.0,np.pi*0.5),
                user_defined_parameters = None,
                rho_doping=None,
                **rho_free,
                ):
        
        self.mesh = mesh
        self.fs = FemSpace(copy.copy(mesh), shape_class_name=shape_class_name)
        
        
        self.dirichlet = dirichlet
        
        # list of free charge density object from k dot p problem
        self.rho_free = rho_free
        self.rho_doping = rho_doping            
            
        # electrostatic potential energy
        self.epot = ElectrostaticPotential(self.fs, V=None, electric_field=electric_field)
        
        # an external electric field
        self.electric_field = electric_field

        if user_defined_parameters is not None:
            self.parameters = user_defined_parameters
            
    def run(self):
        # create solver
        solver = LinearSystem()
        
        # solve for potential energy
        v = Poisson(
            rho_dop = self.rho_doping,
            user_defined_params = self.parameters,
            **self.rho_free
        )
        
        solver.assembly( self.fs, v, dirichlet_borval=self.dirichlet)
        
        x = solver.solve()
        if solver.pure_neumann:
            V = x[:-1]
            c = x[-1]
        else:
            V = x
            c = None

        self.c = c

        self.epot = ElectrostaticPotential(self.fs, V=V, electric_field=self.electric_field)

    
    
    #def write_potential(self, path, idn=''):
    #    file_name = path+'/OUTPUT_DATA_EPOT'+str(idn)
    #    np.save(file_name, self.epot.epot)
