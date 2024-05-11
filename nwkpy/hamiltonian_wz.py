import numpy as np
from scipy.linalg import block_diag
from nwkpy import _constants, _database
import copy
class HamiltonianWZ():
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
        material='GaAs',
        valence_band_edge=0.0,
        principal_axis_direction='001',
        rescaling=0.0,
        temperature=0.0,
        rembands=True,
        spherical_approx=False,
        decouple_split_off=None,
        decouple_conduction = None,
        user_defined_params=None
    ):

        # definition of the parameter set of the hamiltonian
        self.material = material

        if user_defined_params is not None:
            p = copy.copy(user_defined_params[material])
        else:
            p = copy.copy(_database.params[material])
        
        #self.p = p
    
        # the energy gap depend upon temperature...
        p['Eg'] = p['Eg'] - p['alpha'] * temperature**2 / (temperature + p['beta'])

        # the lattice constant depends upon temperature
        p['alc'] = p['alc'] + p['alcTpar'] * (temperature - 300.0)
        
        # modify Ep to eliminate spurious solutions
        if rescaling is not None:
            if type(rescaling)==float or type(rescaling)==int:
                # Ep is lowered
                p['Ep1'] = p['Ep1'] - rescaling * p['Ep1']
                p['Ep2'] = p['Ep2'] - rescaling * p['Ep2'] 
            elif rescaling=='S=0':
                # Ep is such that S = 0
                p['Ep1'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) / p['me_par']
                p['Ep2'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) / p['me_perp']
            elif rescaling=='S=1':
                # Ep is such that S = 0 (completely neglect remote bands)
                p['Ep1'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) *(1./ p['me_par'] - 1.0)
                p['Ep2'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) *(1./ p['me_perp'] - 1.0)
            elif rescaling=='P=0':
                p['Ep1'] = 0.0
                p['Ep2'] = 0.0
            else:
                raise ValueError('Invalid rescaling parameter')
    
            # Ep is such that Ac = 1
            #p['Ep'] = (p['Eg']*(p['Eg'] + p['delta'])) / (p['Eg'] + 2./3. * p['delta']) *(1./ p['me'] - 2.0)
    
        # energy in atomic units
        p['Eg'] /= _constants.energy_scale
        p['Ep1'] /= _constants.energy_scale
        p['Ep2'] /= _constants.energy_scale
        p['delta'] /= _constants.energy_scale
        p['delta1'] /= _constants.energy_scale
        p['delta2'] /= _constants.energy_scale
        p['delta3'] /= _constants.energy_scale
    
        # compute P0
        p['P1'] = np.sqrt(p['Ep1'] * 0.5)
        p['P2'] = np.sqrt(p['Ep2'] * 0.5)
        
        # compute conduction band effective mass
        if rembands:
            p['Ac1'] = 0.5 / p['me_par'] - p['Ep1'] * ( p['Eg'] + 2.0 * p['delta2']) /( ( p['Eg'] + p['delta1'] + p['delta2']) * ( p['Eg'] + 2.0 * p['delta2']) - 2.0*p['delta3']**2 ) 
            p['Ac2'] = 0.5 / p['me_perp'] - p['Ep2'] * ((p['Eg'] + p['delta1'] + p['delta2']) * (p['Eg'] + p['delta2']) - p['delta3']**2)/p['Eg']/( (p['Eg'] + p['delta1'] + p['delta2'])*(p['Eg'] + 2.0 * p['delta2']) - p['delta3']**2 )
        else:
            p['Ac1'] = 0.5
            p['Ac2'] = 0.5

        # decouple condunction band from the rest
        if decouple_conduction is not None:
            p['Eg'] = decouple_conduction / _constants.energy_scale 

        # modify RSP params for 8 band kp
        p['A1'] = p['A1'] + p['Ep2'] / p['Eg']
        #p['A2'] = p['A2']
        p['A3'] = p['A3'] - p['Ep2'] / p['Eg']
        p['A4'] = p['A4'] + 0.5 * p['Ep1'] / p['Eg']
        p['A5'] = p['A5'] + 0.5 * p['Ep1'] / p['Eg']
        p['A6'] = p['A6'] + np.sqrt(2.)*0.5 * np.sqrt(p['Ep1'] * p['Ep2']) / p['Eg']
    
        # compute DKK params
        # store luttinger params
        if spherical_approx is True:
            pass

        p['L1'] = p['A5'] + p['A2'] + p['A4'] - 1.
        p['L2'] = p['A1'] - 1.
        p['M1'] = -p['A5'] +p['A2'] + p['A4'] -1.
        p['M2'] = p['A1'] + p['A3'] -1.
        p['M3'] = p['A2'] - 1.
        p['Np1'] = 3.*p['A5'] - p['A2'] - p['A4'] + 1.
        p['Nm1'] = -p['A5'] + p['A2'] + p['A4'] -1.
        p['Np2'] = np.sqrt(2)*p['A6'] - p['A1'] - p['A3'] + 1.
        p['Nm2'] = p['A1'] + p['A3'] -1.

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
        kx,ky,kz : float, wave vectors [adimensional units]
        """
        # wave vector in nm are converted in adimensional units
        #kx *= _constants.length_scale
        #ky *= _constants.length_scale
        #kz *= _constants.length_scale
        
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
        kz : float, parallel wave-vector [adimensional units]
        """
        if self.is_quantized:
            # the hamiltonian was already quantized
            # so we need to reset the coefficient matrices
            # to non quantized before considering a new kz value
            self._get_hij()
            
        self.kz = kz
            
        # qw hamiltonians
        self.HxL = ( self.HxL + self.kz * self.Hzx ) * ( - 1.j )
        self.HyL = ( self.HyL + self.kz * self.Hzy ) * ( - 1.j )
        
        self.HxR = ( self.HxR + self.kz * self.Hxz ) * ( - 1.j )
        self.HyR = ( self.HyR + self.kz * self.Hyz ) * ( - 1.j )
        
        
        self.H0 = self.H0 \
        + kz * self.Hzz * kz \
        + 1.0 * kz * self.HzR \
        + self.HzL * 1.0 * kz

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

    Hxx[0,0] = p['Ac2']
    Hxx[1,1] = p['Ac2']
    Hxx[2,2] = 0.5 + p['L1']
    Hxx[3,3] = 0.5 + p['M1']
    Hxx[4,4] = 0.5 + p['M3']
    Hxx[5,5] = 0.5 + p['L1']
    Hxx[6,6] = 0.5 + p['M1']
    Hxx[7,7] = 0.5 + p['M3']
    return Hxx
    
def Hxy(p):
    Hxy = np.zeros((8,8), dtype=complex)
    Hxy[2,3] = p['Np1']
    Hxy[3,2] = p['Nm1']
    Hxy[5,6] = p['Np1']
    Hxy[6,5] = p['Nm1']
    
    Hxy[0,4] = p['B3']
    Hxy[1,7] = p['B3']
    return Hxy

def Hxz(p):
    Hxz = np.zeros((8,8), dtype=complex) 
    Hxz[2,4] = p['Np2']
    Hxz[4,2] = p['Nm2']
    Hxz[5,7] = p['Np2']
    Hxz[7,5] = p['Nm2']
    
    Hxz[3,0] = p['B2']
    Hxz[6,1] = p['B2']
    return Hxz

def Hyx(p):
    Hyx = np.zeros((8,8), dtype=complex)
    Hyx[2,3] = p['Nm1']
    Hyx[3,2] = p['Np1']
    Hyx[5,6] = p['Nm1']
    Hyx[6,5] = p['Np1']
    
    Hyx[4,0] = p['B3']
    Hyx[7,1] = p['B3']
    return Hyx

def Hyy(p):
    Hyy = np.zeros((8,8), dtype=complex)

    Hyy[0,0] = p['Ac2']
    Hyy[1,1] = p['Ac2']
    Hyy[2,2] = 0.5 + p['M1']
    Hyy[3,3] = 0.5 + p['L1']
    Hyy[4,4] = 0.5 + p['M3']
    Hyy[5,5] = 0.5 + p['M1']
    Hyy[6,6] = 0.5 + p['L1']
    Hyy[7,7] = 0.5 + p['M3']
    return Hyy
    
def Hyz(p):
    Hyz = np.zeros((8,8), dtype=complex) 
    
    Hyz[3,4] = p['Np2']
    Hyz[4,3] = p['Nm2']
    Hyz[6,7] = p['Np2']
    Hyz[7,6] = p['Nm2']
    
    Hyz[0,2] = p['B1']
    Hyz[1,5] = p['B1']
    return Hyz
    
def Hzx(p):
    Hzx = np.zeros((8,8), dtype=complex)
    Hzx[2,4] = p['Nm2']
    Hzx[4,2] = p['Np2']
    Hzx[5,7] = p['Nm2']
    Hzx[7,5] = p['Np2']
    
    Hzx[0,3] = p['B2']
    Hzx[1,6] = p['B2']
    return Hzx
    
def Hzy(p):
    Hzy = np.zeros((8,8), dtype=complex)
    
    Hzy[3,4] = p['Nm2']
    Hzy[4,3] = p['Np2']
    Hzy[6,7] = p['Nm2']
    Hzy[7,6] = p['Np2']
    
    Hzy[2,0] = p['B1'] 
    Hzy[5,1] = p['B1']
    return Hzy

def Hzz(p):
    Hzz = np.zeros((8,8), dtype=complex)

    Hzz[0,0] = p['Ac1']
    Hzz[1,1] = p['Ac1']
    Hzz[2,2] = 0.5 + p['M2']
    Hzz[3,3] = 0.5 + p['M2']
    Hzz[4,4] = 0.5 + p['L2']
    Hzz[5,5] = 0.5 + p['M2']
    Hzz[6,6] = 0.5 + p['M2']
    Hzz[7,7] = 0.5 + p['L2']
    return Hzz
    


# build first order coefficient matrices    
def HxL(p):
    HxL = np.zeros((8,8), dtype=complex)
    HxL[0,2] = 1.j * p['P2']
    HxL[1,5] = 1.j * p['P2']
    return HxL

def HyL(p):
    HyL = np.zeros((8,8), dtype=complex)
    HyL[0,3] = 1.j * p['P2']
    HyL[1,6] = 1.j * p['P2']
    return HyL
    
def HzL(p):
    HzL = np.zeros((8,8), dtype=complex)
    HzL[0,4] = 1.j * p['P1']
    HzL[1,7] = 1.j * p['P1']
    return HzL
    
def HxR(p):
    HxR = np.zeros((8,8), dtype=complex)
    HxR[2,0] = - 1.j * p['P2']
    HxR[5,1] = - 1.j * p['P2']
    return HxR

def HyR(p):
    HyR = np.zeros((8,8), dtype=complex)
    HyR[3,0] = - 1.j * p['P2']
    HyR[6,1] = - 1.j * p['P2']
    return HyR
    
def HzR(p):
    HzR = np.zeros((8,8), dtype=complex)
    HzR[4,0] = - 1.j * p['P1']
    HzR[7,1] = - 1.j * p['P1']
    return HzR

# zeroth order coefficient matrix

# spin-orbit term
def Hso(p):
    Hso = np.zeros((8,8), dtype=complex)

    Hso[2,3] = - 1.j * p['delta2']
    Hso[3,2] = 1.j * p['delta2']
    Hso[2,7] = 1. * p['delta3']
    Hso[7,2] = 1. * p['delta3']
    Hso[3,7] = -1.j * p['delta3']
    Hso[7,3] = 1.j * p['delta3']
    Hso[4,5] = -1. * p['delta3']
    Hso[5,4] = -1. * p['delta3']
    Hso[4,6] = 1.j * p['delta3']
    Hso[6,4] = -1.j * p['delta3']
    Hso[5,6] = 1.j * p['delta2']
    Hso[6,5] = -1.j * p['delta2']

    Hso[2,2] = p['delta1']
    Hso[3,3] = p['delta1']
    Hso[5,5] = p['delta1']
    Hso[6,6] = p['delta1']
    
    #Hso *= p['delta'] / 3.0

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

#def Q():
#    # Matrix of (complex) Clebsh-Gordan coefficients.
#    # In this basis the spin orbit interaction matrix is diagonal 
#    Q = np.zeros((8,8), dtype=complex)
#    Q[0,0] = 1.0
#    Q[1,1] = 1j
#    Q[2,2] = 1. / np.sqrt(2.)
#    Q[2,3] = 1j / np.sqrt(2.)
#    Q[3,5] = 1j / np.sqrt(2.)
#    Q[3,6] = 1. / np.sqrt(2.)
#    Q[4,4] = - 2. * 1j / np.sqrt(6.)
#    Q[4,5] = 1j / np.sqrt(6.)
#    Q[4,6] = - 1. / np.sqrt(6.)
#    Q[5,2] = 1. / np.sqrt(6.)
#    Q[5,3] = - 1j / np.sqrt(6.)
#    Q[5,7] = 2. / np.sqrt(6.)
#    Q[6,4] = 1. / np.sqrt(3.)
#    Q[6,5] = 1. / np.sqrt(3.)
#    Q[6,6] = 1j / np.sqrt(3.)
#    Q[7,2] = - 1j / np.sqrt(3.)
#    Q[7,3] = - 1. / np.sqrt(3.)
#    Q[7,7] = 1j / np.sqrt(3.)
#    return Q

def Q():
    Q = np.eye(8, dtype=complex)
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




if __name__=='__main__':
    H = Hamiltonian(
        material='GaAs',
        valence_band_edge=0.0,
        principal_axis_direction='001',
        rescaling='S=0',
        temp=0.0,
        rembands=True,
        user_defined_params=None        
    )
    print(H.get(0,0,0))
