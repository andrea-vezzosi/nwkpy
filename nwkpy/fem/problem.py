import numpy as np
from nwkpy import _constants
from nwkpy import _database
def build_dict(keys, values):
    zip_ite = zip(keys, values)
    return dict(zip_ite)


############################ SCHRODINGER PROBLEM ##########################
class Schrodinger:
    
    """
    Implements a multiband k dot p model
    for the electronic band structure of the system
    """
    
    def __init__(
        self,
        kz,
        hamiltonian_dict,
        **epot
    ):
        # set number of components of the solution
        self.ncom = 8

        self.hamiltonian_dict = hamiltonian_dict
        
        for key in hamiltonian_dict.keys():
            hamiltonian_dict[key].quantize1D(kz)
        
        # the epot argument is a set of potential energy objects with keyword
        self.epot = epot
        
        # flag for uncoupled band approximation
        #self.use_ub = use_ub

    def get_aelm(self, fel):
        ham = self.hamiltonian_dict[fel.material]
        n = fel.shape.ndofV 

        # second order 
        aelm2or = BlockMultiply( fel.ddx  , ham.Hxx , n )    \
               + BlockMultiply( fel.ddy  , ham.Hyy , n)    \
               + BlockMultiply( fel.dxdy , ham.Hxy, n )    \
               + BlockMultiply( fel.dydx , ham.Hyx, n )
        
        # first order and zeroth order
        aelm1or0or = BlockMultiply( fel.dxl  , -ham.HxR, n )    \
                     + BlockMultiply( fel.dxr  , ham.HxL, n )    \
                     + BlockMultiply( fel.dyl  , -ham.HyR, n )    \
                     + BlockMultiply( fel.dyr  , ham.HyL , n)    \
                     + BlockMultiply( fel.I    , ham.H0 , n ) 
        
        V = np.zeros(fel.shape.gpar.n)
        gauss_coords = fel.gauss_coords
        for epot in self.epot.values():
            # V [eV] is a vector containing the potential energy on gauss pts of this element
            # since the kp hamiltonian is in atomic units, express also V in the same units
            V += epot.interp(gauss_coords) / _constants.energy_scale
        # build element matrix
        epotmat = fel.int_psi_f_psi(V)
        # expand el mat to ncom
        epotmat = BlockMultiply(epotmat ,  np.eye(self.ncom) , n  )

        aelm =  aelm2or + aelm1or0or + epotmat

        return aelm      
    
    def get_belm(self, fel):
        n = fel.shape.ndofV
        belm = BlockMultiply(fel.I , np.eye(self.ncom), n)
        return belm


############################ POISSON PROBLEM ##########################

"""
The Poisson equation is solved in cgs units. We follow the book by Ram-Mohan.
The vacuum permittity is equal to 1/(4 pi). Instead of solving for the electrostatic potential \phi,
this code solves for the electrostatic potential energy V = - |e| \phi.
Also, as we have done for the Schrodinger equation, we work in adimensional length units: we set x = \Tilde{x} l0
and y = \Tilde{x} l0. Here l0 has to be expressed in cm, since we are in cgs units, and it is equal to the bohr radius l0 = 0.521 * 10^{-8} cm.
In this way the Poisson equation reads
\Grad^2 V = 4 pi l0^2 e^2 n = S
where n is the carrier density of electrons, holes or doping with its corresponding sign. 
Now, we have 4 pi l0^2 e^2 = 1.8095128174304022e-22 * (0.521)^2 eV cm^3 
The source term S is given by this constant multiplied by the carrier density.
In the code (0.521)^2 is represented by _constants.length_scale**2

To be consistent, the charge density has to be expressed in cm^{-3} into this equation. 
So the source term on the RHS of the Poisson equation has the unit of eV.
Also the LHS of the Poisson equation has the units of eV, as the second derivatives are in adimensional units.
"""

class Poisson:
    def __init__(self, user_defined_params=None, rho_dop = None, **rho_free):
        
        # doping charge density
        self.rho_dop = rho_dop
        
        # free charge density
        # the rho_free argument is a set of charge objects with keyword
        self.rho_free = rho_free
        self.ncom = 1

        if user_defined_params is not None:
            self.parameters = user_defined_params
        else:
            self.parameters = _database.params
        
    def get_elmat(self, fel):
        
        # no charge
        S = np.zeros(fel.shape.gpar.n) 
        gauss_coords = fel.gauss_coords
        if self.rho_dop is not None:
            # add doping density
            Sd = self.rho_dop.interp(gauss_coords)
            Sd *= 1.8095128174304022e-22  * _constants.length_scale**2
            S += Sd 
                
        #if self.rho_free is not None:
        for rho in self.rho_free.values():
            # add free charge density
            Se = rho.interp(gauss_coords)
            Se = Se * 1.8095128174304022e-22  * _constants.length_scale**2
            S += Se 
        

        # set dielectric constant
        #eps = _database.params[fel.material]['eps']
        eps = self.parameters[fel.material]['eps']
        #eps = fel.material.p['eps']
        
        # stiffness matrix (laplacian of V times dielectric constant)
        melm = ( fel.ddx + fel.ddy ) * eps
        
        # load vector (S is called source term)
        felm = - fel.int_psi_f( S )

        # constrain vector
        celm = fel.int_psi_f( np.full(fel.shape.gpar.n,1))
        
        return melm, felm, celm    
    

############################ TOTAL ANGULAR MOMENTUM PROBLEM ##########################

class TotAngMom:
    """
    Implements total angular momentum problem 
    """
    
    def __init__(
        self,
        comps=np.s_[0:8]
    ):
        # set number of components of the solution
        self.ncom = comps.stop-comps.start

        # spin angular momentum
        Qmat = np.diag([1.,-1.j, -1,-1.j,1.j,1,-1,-1.j])
        Jz = np.diag(v = np.array([0.5, -0.5, 1.5, -1.5, 0.5, -0.5, 0.5, -0.5]))
        Jz = np.conjugate(Qmat) @ Jz @ Qmat.T
        self.Jz = Jz[comps,comps]

        self.epot = None

    def get_aelm(self, fel):
        n = fel.shape.ndofV
        aelm = BlockMultiply(fel.Lz , np.eye(self.ncom), n) + BlockMultiply(fel.I, self.Jz, n)
        return aelm
    
    def get_belm(self, fel):
        n = fel.shape.ndofV
        belm = BlockMultiply(fel.I , np.eye(self.ncom), n)
        return belm
    
######################################################################################

def BlockMultiply(A, B, n):
    AB = np.block([
        [np.kron(A[:n,:n], B[:2,:2]), np.kron(A[:n,n:], B[:2,2:])],
        [np.kron(A[n:,:n], B[2:,:2]), np.kron(A[n:,n:] , B[2:,2:])]
    ])
    return AB
    
