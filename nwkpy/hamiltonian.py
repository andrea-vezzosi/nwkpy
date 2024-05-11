import numpy as np
from scipy.linalg import block_diag
from nwkpy import _constants, _database
import copy
class HamiltonianZB():
    """
    Hamiltonian object
    
    Init Parameters
    ----------
    material : str, material name
    ev : float, valence band edge, default is 0 eV
    principal_axis_direction : str, principal axis direction, default is '111'
    rescaling : str, default is 'S=0'
    temp : float, temperature of the system, default is 0 K 
    rembands : bool, include effects of remote bands, default True 
    user_defined_params : dict, a different paremeter set, defined by the user, default is None
    
    Attributes
    ----------
    p : dict, material parameters
    theta : float, rotation angle
    phi : float, rotation angle
    
    rotmat : ndarray, rotation matix + basis change (see Xu article)
    is_rotated : bool, True if a roation has been performed
    
    C2 : ndarray, second order coefficients
    C1L : ndarray, first order left coefficients
    C1R : ndarray, first order right coefficients
    C0 : ndarray, zeroth order coefficients
    
    Hij : ndarray, coefficient matrices
    
    
    """
    def __init__(
        self,
        material,
        valence_band_edge=0.0,
        principal_axis_direction='001',
        rescaling='S=0',
        temperature=0.0,
        rembands=True,
        spherical_approx = False,
        decouple_split_off = None,
        decouple_conduction = None,
        user_defined_params = None
    ):

        # definition of the parameter set of the hamiltonian
        self.material = material

        if user_defined_params is not None:
            p = copy.copy(user_defined_params[material])
        else:
            p = copy.copy(_database.params[material])
    
        # the energy gap depend upon temperature...
        p['Eg'] = p['Eg'] - p['alpha'] * temperature**2 / (temperature + p['beta'])

        # the lattice constant depends upon temperature
        p['alc'] = p['alc'] + p['alcTpar'] * (temperature - 300.0)
        
        # modify Ep to eliminate spurious solutions
        if rescaling is not None:
            if type(rescaling)==float or type(rescaling)==int:
                # Ep is lowered
                p['Ep'] = p['Ep'] - rescaling * p['Ep']
            elif rescaling=='S=0':
                # Ep is such that S = 0
                p['Ep'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) / p['me']
            elif rescaling=='S=1':
                # Ep is such that S = 0 (completely neglect remote bands)
                p['Ep'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) *(1./ p['me'] - 1.0)
            elif rescaling=='P=0':
                p['Ep'] = 0.0
            else:
                raise ValueError('Invalid rescaling parameter')
    
            # Ep is such that Ac = 1
            #p['Ep'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) *(1./ p['me'] - 2.0)
    
        # energy in atomic units
        p['Eg'] /= _constants.energy_scale
        p['Ep'] /= _constants.energy_scale
        p['delta'] /= _constants.energy_scale
    
        # compute P0
        p['P'] = np.sqrt(p['Ep'] * 0.5)
        
        # compute conduction band effective mass
        if rembands:
            p['Ac'] = 0.5 / p['me'] - 2. / 3. * p['P']**2 / p['Eg'] - 1. / 3. * p['P']**2 / (p['Eg']+p['delta'])
        else:
            p['Ac'] = 0.5

        # decouple condunction band from the rest
        if decouple_conduction is not None:
            p['Eg'] = decouple_conduction / _constants.energy_scale 

        # modify lu params for 8 band kp
        p['lu1'] = p['lu1'] - p['Ep']/(3.0*p['Eg'])
        p['lu2'] = p['lu2'] - p['Ep']/(6.0*p['Eg'])
        p['lu3'] = p['lu3'] - p['Ep']/(6.0*p['Eg'])
    
        # compute DKK params
        # store luttinger params
        if spherical_approx is True:
            lu_spher = (p['lu2']+p['lu3'])*0.5
            p['lu2'] = lu_spher
            p['lu3'] = lu_spher

        lu1 = p['lu1']
        lu2 = p['lu2']
        lu3 = p['lu3']
        
        # compute DKK params (au) (do not include free electron term)
        L = (-lu1 - 4. * lu2 - 1.) * 0.5
        M = (2. * lu2 - lu1 - 1.) * 0.5
        Nm = M
        Np = (- 6. * lu3 - (2. * lu2 - lu1 - 1.)) * 0.5
        N = Np + Nm
        
        # save into param dict
        p['L'] = L
        p['M'] = M
        p['N'] = N
        p['Nm'] = Nm
        p['Np'] = Np

        # valence and conduction band edges
        ev = valence_band_edge / _constants.energy_scale
        ec = ev + p['Eg']

        # decouple SO bands from the rest
        if decouple_split_off is not None:
            p['delta'] *= decouple_split_off

        self.p = p

        ############ BUILD HAMILTONIAN #############
        # 2nd order coefficients
        self.C2 = np.block([[Hxx(p) , Hxy(p) , Hxz(p)],
                           [Hyx(p) , Hyy(p) , Hyz(p)],
                           [Hzx(p) , Hzy(p) , Hzz(p)]])
        
        # first order coefficients
        self.C1L = np.block([[HxL(p)],
                            [HyL(p)],
                            [HzL(p)]])
        
        self.C1R = np.block([[HxR(p)],
                            [HyR(p)],
                            [HzR(p)]])
        self.C0 = Hd(p, ec, ev) + Hso(p)
        
        # store coeffients matrices
        self._get_hij()
        
        # rotation of the hamiltonian + change of basis
        if principal_axis_direction == '111':
            theta = np.arccos(1./np.sqrt(3.))
            phi = np.pi/4.
        elif principal_axis_direction =='100':
            theta = np.pi*0.5
            phi = 0.0
            
        elif principal_axis_direction =='001':
            theta = 0.0
            phi = 0.0    
        else:
            raise ValueError('Invalid principal axis direction')  

        self.theta = theta
        self.phi = phi 

        self._rotate(theta, phi)
        self.is_rotated=True   

        self.is_quantized=False  

        
    def _rotate(self, theta , phi ):
        """
        Rotate + basis change Hamiltonian
        
        Parameters
        ----------
        theta : float, rotationa angle
        phi : float, rotation angle
        """
        rotmat = P( theta , phi )
        
        # effettua la rotazione della matrice dei coefficienti
        rotham = np.kron( np.eye(3) , rotmat)
        rotx = np.kron( R( theta , phi ) , np.eye(8) )

        # transform 1st order coeff in block diag form
        C1Lbd = block_diag(self.HxL, self.HyL, self.HzL)
        C1Rbd = block_diag(self.HxR, self.HyR, self.HzR)
        
        # rotation plus change of basis
        C2tmp = np.conjugate(rotham) @ self.C2 @ np.transpose(rotham)
        C1Lbd = np.conjugate(rotham) @ C1Lbd @ np.transpose(rotham)
        C1Rbd = np.conjugate(rotham) @ C1Rbd @ np.transpose(rotham)
        C0tmp = np.conjugate(rotmat) @ self.C0 @ np.transpose(rotmat)
        
        self._C2tmp = C2tmp
        self._C1Ltmp = C1Lbd @ np.kron(np.ones((3,1)),np.eye(8))
        self._C1Rtmp = C1Rbd @ np.kron(np.ones((3,1)),np.eye(8))
        
        # rotation of k vector
        self.C2 = rotx @ self._C2tmp @ np.transpose(rotx) 
        self.C1L = rotx @ C1Lbd @ np.kron(np.ones((3,1)),np.eye(8))
        self.C1R = rotx @ C1Rbd @ np.kron(np.ones((3,1)),np.eye(8))
        self.C0 = C0tmp
        
        self.is_rotated = True
        # update coefficient matrice
        self._get_hij()


    def _get_hij(self):
        """
        Store coefficient matrices
        """
        # bulk coefficients matrices
        self.Hxx = self.C2[0:8,0:8]
        self.Hxy = self.C2[0:8,8:16]
        self.Hxz = self.C2[0:8,16:24]
        
        self.Hyx = self.C2[8:16,0:8]
        self.Hyy = self.C2[8:16,8:16]
        self.Hyz = self.C2[8:16,16:24]
        
        self.Hzx = self.C2[16:24,0:8]
        self.Hzy = self.C2[16:24,8:16]
        self.Hzz = self.C2[16:24,16:24]
        
        self.HxL = self.C1L[0:8,0:8]
        self.HyL = self.C1L[8:16,0:8]
        self.HzL = self.C1L[16:24,0:8]
        
        self.HxR = self.C1R[0:8,0:8]
        self.HyR = self.C1R[8:16,0:8]
        self.HzR = self.C1R[16:24,0:8]
        
        self.H0 = self.C0.copy()

    def get(self, kx, ky, kz):
        
        """
        Returns bulk Hamiltonian
        kx,ky,kz : float, wave vectors [Angstrom]
        """
        # wave vector in A^{-1} are converted in adimensional units
        kx *= _constants.length_scale
        ky *= _constants.length_scale
        kz *= _constants.length_scale
        
        H = np.zeros((8,8), dtype=complex)
        
        H = kx * self.Hxx * kx + \
                ky * self.Hyy * ky + \
                kz * self.Hzz * kz + \
                kx * self.Hxy * ky + \
                kx * self.Hxz * kz + \
                ky * self.Hyx * kx + \
                ky * self.Hyz * kz + \
                kz * self.Hzx * kx + \
                kz * self.Hzy * ky + \
                     self.HxL * kx + \
                     self.HyL * ky + \
                     self.HzL * kz + \
                kx * self.HxR      + \
                ky * self.HyR      + \
                kz * self.HzR      + \
                     self.H0        
        return H

    def solve(self, kx, ky, kz):
        if self.is_quantized:
            # the hamiltonian was already quantized
            # so we need to reset the coefficient matrices
            # to non quantized before considering a new kz value
            self._get_hij()
            self.is_quantized=False

        H = self.get(kx, ky, kz)
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # sort eigenvalues and eigenvectors
        mask = np.argsort(eigvals)[::-1]
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:,mask]
        #eigvals *= _constants.energy_scale # eV
        return eigvals, eigvecs

    def quantize1D(self, kz):
        """
        Quantize hamiltoninan
        
        Parameters
        ----------
        kz : float, parallel wave-vector [Angstrom]
        """
        if self.is_quantized:
            # the hamiltonian was already quantized
            # so we need to reset the coefficient matrices
            # to non quantized before considering a new kz value
            self._get_hij()
            
        self.kz = kz
        
        # convert to adimensional units (kz enters in A^{-1})
        kz_adim = kz * _constants.length_scale
            
        # qw hamiltonians
        self.HxL = ( self.HxL + kz_adim * self.Hzx ) * ( - 1.j )
        self.HyL = ( self.HyL + kz_adim * self.Hzy ) * ( - 1.j )
        
        self.HxR = ( self.HxR + kz_adim * self.Hxz ) * ( - 1.j )
        self.HyR = ( self.HyR + kz_adim * self.Hyz ) * ( - 1.j )
        
        
        self.H0 = self.H0 \
        + kz_adim * self.Hzz * kz_adim \
        + 1.0 * kz_adim * self.HzR \
        + self.HzL * 1.0 * kz_adim

        self.is_quantized = True


############## BULK COEFFICIENT MATRICES ################

# second order coefficients matrices
def Hxx(p):
    """
    Returns coefficient matrix 
    
    Parameters
    ----------
    p : dict, material parameters
    
    Returns
    ----------
    Hij : complex ndarray, coefficient matrices
    
    """
    Hxx = np.zeros((8,8), dtype=complex)

    Hxx[0,0] = p['Ac']
    Hxx[1,1] = p['Ac']
    Hxx[2,2] = 0.5 + p['L']
    Hxx[3,3] = 0.5 + p['M']
    Hxx[4,4] = 0.5 + p['M']
    Hxx[5,5] = 0.5 + p['L']
    Hxx[6,6] = 0.5 + p['M']
    Hxx[7,7] = 0.5 + p['M']
    return Hxx
    
def Hxy(p):
    Hxy = np.zeros((8,8), dtype=complex)
    Hxy[2,3] = p['Np']
    Hxy[3,2] = p['Nm']
    Hxy[5,6] = p['Np']
    Hxy[6,5] = p['Nm']
    
    Hxy[0,4] = p['B']
    Hxy[1,7] = p['B']
    return Hxy

def Hxz(p):
    Hxz = np.zeros((8,8), dtype=complex) 
    Hxz[2,4] = p['Np']
    Hxz[4,2] = p['Nm']
    Hxz[5,7] = p['Np']
    Hxz[7,5] = p['Nm']
    
    Hxz[3,0] = p['B']
    Hxz[6,1] = p['B']
    return Hxz

def Hyx(p):
    Hyx = np.zeros((8,8), dtype=complex)
    Hyx[2,3] = p['Nm']
    Hyx[3,2] = p['Np']
    Hyx[5,6] = p['Nm']
    Hyx[6,5] = p['Np']
    
    Hyx[4,0] = p['B']
    Hyx[7,1] = p['B']
    return Hyx

def Hyy(p):
    Hyy = np.zeros((8,8), dtype=complex)

    Hyy[0,0] = p['Ac']
    Hyy[1,1] = p['Ac']
    Hyy[2,2] = 0.5 + p['M']
    Hyy[3,3] = 0.5 + p['L']
    Hyy[4,4] = 0.5 + p['M']
    Hyy[5,5] = 0.5 + p['M']
    Hyy[6,6] = 0.5 + p['L']
    Hyy[7,7] = 0.5 + p['M']
    return Hyy
    
def Hyz(p):
    Hyz = np.zeros((8,8), dtype=complex) 
    
    Hyz[3,4] = p['Np']
    Hyz[4,3] = p['Nm']
    Hyz[6,7] = p['Np']
    Hyz[7,6] = p['Nm']
    
    Hyz[0,2] = p['B']
    Hyz[1,5] = p['B']
    return Hyz
    
def Hzx(p):
    Hzx = np.zeros((8,8), dtype=complex)
    Hzx[2,4] = p['Nm']
    Hzx[4,2] = p['Np']
    Hzx[5,7] = p['Nm']
    Hzx[7,5] = p['Np']
    
    Hzx[0,3] = p['B']
    Hzx[1,6] = p['B']
    return Hzx
    
def Hzy(p):
    Hzy = np.zeros((8,8), dtype=complex)
    
    Hzy[3,4] = p['Nm']
    Hzy[4,3] = p['Np']
    Hzy[6,7] = p['Nm']
    Hzy[7,6] = p['Np']
    
    Hzy[2,0] = p['B'] 
    Hzy[5,1] = p['B']
    return Hzy

def Hzz(p):
    Hzz = np.zeros((8,8), dtype=complex)

    Hzz[0,0] = p['Ac']
    Hzz[1,1] = p['Ac']
    Hzz[2,2] = 0.5 + p['M']
    Hzz[3,3] = 0.5 + p['M']
    Hzz[4,4] = 0.5 + p['L']
    Hzz[5,5] = 0.5 + p['M']
    Hzz[6,6] = 0.5 + p['M']
    Hzz[7,7] = 0.5 + p['L']
    return Hzz
    


# build first order coefficient matrices    
def HxL(p):
    HxL = np.zeros((8,8), dtype=complex)
    HxL[0,2] = 1.j * p['P']
    HxL[1,5] = 1.j * p['P']
    return HxL

def HyL(p):
    HyL = np.zeros((8,8), dtype=complex)
    HyL[0,3] = 1.j * p['P']
    HyL[1,6] = 1.j * p['P']
    return HyL
    
def HzL(p):
    HzL = np.zeros((8,8), dtype=complex)
    HzL[0,4] = 1.j * p['P']
    HzL[1,7] = 1.j * p['P']
    return HzL
    
def HxR(p):
    HxR = np.zeros((8,8), dtype=complex)
    HxR[2,0] = - 1.j * p['P']
    HxR[5,1] = - 1.j * p['P']
    return HxR

def HyR(p):
    HyR = np.zeros((8,8), dtype=complex)
    HyR[3,0] = - 1.j * p['P']
    HyR[6,1] = - 1.j * p['P']
    return HyR
    
def HzR(p):
    HzR = np.zeros((8,8), dtype=complex)
    HzR[4,0] = - 1.j * p['P']
    HzR[7,1] = - 1.j * p['P']
    return HzR

# zeroth order coefficient matrix

# spin-orbit term
def Hso(p):
    Hso = np.zeros((8,8), dtype=complex)

    Hso[2,3] = - 1.j
    Hso[3,2] = 1.j
    Hso[2,7] = 1.
    Hso[7,2] = 1.
    Hso[3,7] = -1.j
    Hso[7,3] = 1.j
    Hso[4,5] = -1.
    Hso[5,4] = -1.
    Hso[4,6] = 1.j
    Hso[6,4] = -1.j
    Hso[5,6] = 1.j
    Hso[6,5] = -1.j
    
    Hso *= p['delta'] / 3.0

    return Hso

# band-edge term
def Hd(p, ec, ev):
    """
    Returns diagonal kp matrix term
    
    Parameters
    ----------
    p : dict, material parameters
    ec : float, conduction band edge
    ev : float, valence band edge
    
    Returns
    ----------
    Hd : complex ndarray
    
    """
    Hd = np.zeros((8,8), dtype=complex)
    
    # compute mean valence band edge
    evm = ev - p['delta']/3.
    
    Hd = np.diag([ec , ec , evm , evm , evm , evm , evm , evm]).astype(complex)
    
    return Hd




########## ROTATION MATRICES ###########

"""Functions for rotations and change of basis of the 
kp bulk hamiltonian
"""

def R(theta, phi):
    # standard rotation matrix in R^3
    R = np.zeros((3,3), dtype=complex)
    R[0,0] = np.cos(phi) * np.cos(theta)
    R[0,1] = np.sin(phi) * np.cos(theta)
    R[0,2] = - np.sin(theta)
    R[1,0] = - np.sin(phi)
    R[1,1] = np.cos(phi)
    #R[1,2] = 0.0
    R[2,0] = np.cos(phi) * np.sin(theta)
    R[2,1] = np.sin(phi) * np.sin(theta)
    R[2,2] = np.cos(theta)
    return R

def U(theta, phi):
    # spinor orbital rotation matrix
    U = np.zeros((8,8), dtype=complex)
    U[0,0] = 1.0
    U[1,1] = 1.0
    U[2:8,2:8] = np.kron( np.eye(2) , R( theta , phi ) )
    return U


def Abar(theta, phi):
    # standard spin rotation matrix
    Abar = np.zeros((2,2), dtype=complex)
    Abar[0,0] = + np.exp(- 1j * phi * 0.5) * np.cos( theta * 0.5)
    Abar[0,1] = + np.exp(+ 1j * phi * 0.5) * np.sin( theta * 0.5)
    Abar[1,0] = - np.exp(- 1j * phi * 0.5) * np.sin( theta * 0.5)
    Abar[1,1] = + np.exp(+ 1j * phi * 0.5) * np.cos( theta * 0.5)
    return Abar

def A(theta, phi):
    # spinor spin rotation matrix
    Ab = Abar(theta, phi)
    A = np.zeros((8,8), dtype=complex)
    A[0:2,0:2] = Ab
    A[2:8,2:8] = np.kron(Ab,np.eye(3))
    return A

def Q():
    # Matrix of (complex) Clebsh-Gordan coefficients.
    # In this basis the spin orbit interaction matrix is diagonal 
    Q = np.zeros((8,8), dtype=complex)
    Q[0,0] = 1.0
    Q[1,1] = 1j
    Q[2,2] = 1. / np.sqrt(2.)
    Q[2,3] = 1j / np.sqrt(2.)
    Q[3,5] = 1j / np.sqrt(2.)
    Q[3,6] = 1. / np.sqrt(2.)
    Q[4,4] = - 2. * 1j / np.sqrt(6.)
    Q[4,5] = 1j / np.sqrt(6.)
    Q[4,6] = - 1. / np.sqrt(6.)
    Q[5,2] = 1. / np.sqrt(6.)
    Q[5,3] = - 1j / np.sqrt(6.)
    Q[5,7] = 2. / np.sqrt(6.)
    Q[6,4] = 1. / np.sqrt(3.)
    Q[6,5] = 1. / np.sqrt(3.)
    Q[6,6] = 1j / np.sqrt(3.)
    Q[7,2] = - 1j / np.sqrt(3.)
    Q[7,3] = - 1. / np.sqrt(3.)
    Q[7,7] = 1j / np.sqrt(3.)
    return Q

def P(theta,phi):
    # full spinor rotation matrix (orbital + spin + ang mom)
    P = Q() @ A( theta , phi ) @ U( theta , phi )
    #P = A( theta , phi ) @ U( theta , phi )

    return P


######################################################################
######################################################################

# Angular momentum matrices

# z-component of the Bloch angular momentum
Jz = np.diag(v = np.array([0.5,-0.5, 1.5, -1.5, 0.5, -0.5, 0.5, -0.5]))
