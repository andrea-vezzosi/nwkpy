import numpy as np
from scipy.sparse import coo_matrix, dok_matrix, csc_matrix,  tril, triu
from scipy.fft import fft2, ifft2
from scipy.interpolate import LinearNDInterpolator
from nwkpy import _constants
from nwkpy import _common
from nwkpy import hamiltonian
from nwkpy.fem import get_spinor_dist
from nwkpy.fem import delete_from_csr, sort_eig
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

import copy
import matplotlib.pyplot as plt
from matplotlib import figure
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.tri import UniformTriRefiner
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# This module contains physical objects and their related functionalities

class FreeChargeDensity:
    def __init__(self, fs, logger=None):
        
        # finite element space
        self.fs = fs
        
        self.ch_dens = None
        
        # the negative charge density on the mesh nodes
        self.n = np.zeros(self.fs.mesh.ng_nodes)

        # the positive charge density on the mesh nodes
        self.p = np.zeros(self.fs.mesh.ng_nodes)

        # fermi occupation for the last added charge
        self.f_occ = None
        
        # linear interpolator, used only if  self._projection_method=True
        self.linear_interpolator = None
        
        # charge interpolation coefficients for electron and hole components (all regions)
        self.charge_matrix_el = csc_matrix((fs.dlnc.sum(),fs.dlnc.sum()), dtype=float)
        self.charge_matrix_h = csc_matrix((fs.dlnc.sum(),fs.dlnc.sum()), dtype=float)

        # charge interpolation coefficients for electron and hole components (specific regions)
        self.charge_matrix_el_pure = csc_matrix((fs.dlnc.sum(),fs.dlnc.sum()), dtype=float)
        self.charge_matrix_h_pure = csc_matrix((fs.dlnc.sum(),fs.dlnc.sum()), dtype=float)

        self.background_charge_list=[]

        self.rectangular_mesh=None

        if logger is not None:
            self.logger = logger
        else:
            self.logger = _common.Logger(rank=0,logfile_path='./')

    def __add__(self, rho_add):
        # this method is to add two different charge objects
        # has to be modified for standard envelope approach
        new_rho = self

        new_charge_matrix_el = self.charge_matrix_el + rho_add.charge_matrix_el
        new_rho.charge_matrix_el = new_charge_matrix_el

        new_charge_matrix_h = self.charge_matrix_h + rho_add.charge_matrix_h
        new_rho.charge_matrix_h = new_charge_matrix_h
        
        return new_rho

    def delete_cpp_objects(self):
        self.fs.mesh.triangulation._cpp_triangulation = None
        self.fs.mesh.trifinder = None
        self.fs.mesh.triangulation._trifinder = None

    def get_cpp_objects(self):
        self.fs.mesh.triangulation._cpp_triangulation = self.fs.mesh.triangulation.get_cpp_triangulation()
        self.fs.mesh.trifinder = self.fs.mesh.triangulation.get_trifinder()
        
    def fermi_occ(self, e, mu, temp, particle):
        """Evaluate occupation factor according to fermi statistics times the charge sign
        ---------
        Parameters:

        e : ndarrray, float
            Subband energies in eV
        mu : float
            Chemical potential in eV
        temp : float
            Temperature in K
        e_separate : float
            Separation energy in eV
        
        particle : character
            particle kind, 'electron' or 'hole

        ---------
        Return:
        if particle == 'electron
            -f : ndarray
        elif particle == 'hole
            1-f : ndarray
        
        ---------
        Notes:
        The sign of the free charge is included in the output vector.
        Minus sign for electrons, positive sign for holes.
        """
        
        f = 1.0 / (1.0 + np.exp( (e - mu) / (8.617333262145e-5 * temp) ) )
        if particle=='electron':
            return - f
        elif particle=='hole':
            return 1.0 - f
        else:
            raise ValueError("invalid particle kind")     
    
    def add_charge(
            self, 
            psi, 
            e, 
            dk, 
            mu, 
            temp, 
            modified_EFA=False,
            particle='electron', 
            norm_sum_region=None, 
            thr_el=0.5, 
            thr_h=0.5
            ):

        '''Add new free charge density within the standard EFA approach

        -----------
        Parameters

        psi : ndarray complex,
            The wave function vector of dimension (nk, ng_nodes, ndof, ncom, neig) where
            ''nk'' is the number of k points, ''ng_nodes'' is the number of nodes in the mesh,
            ''ndof'' is the number of degrees of freedom, ncom is the number of components of the 
            wave function and ''neig'' is the number of subbands included.
        
        e : ndarray
            eigenvalues in eV. Dimension is (nk, neig)
        
        dk : float
            Spacing in k-space.
        
        mu : float
            Chemical potential in eV.
        
        temp : float
            Temperature in K.
        
        e_separate : float
            Separation energy.
        
        particle : character
            Particle kind, either 'electron' or 'hole'.
        
        -----------
        Notes:

        Updates the current self.charge_matrix (used for interpolation)
        and also the 1D array self.ch_dens.

        This class contains the carrier density. We want to express the carrier density in units 
        of cm^{-3} in order to be consistent with the implementation of the Poisson equation. 
        The wave functions used to calculate the charge are obtained solving the Schrodinger equation 
        in adimensional units, so we have |psi|^2=|psi_rescaled|^2 / l0^2. The l0^2 comes from the fact 
        that the normalization happens in two dimensions (see Ram mohan book). Also, we are using rescaled kz values
        for the integration. We have kz = kz_rescaled / l0. So we have in integral where everything is adimensional, times
        a factor 1/l0^3. To express the charge in cm^{-3} we simply express l0 = l0 = 0.521 * 10^{-8} cm. And infact 
        we see in the code that a factor of 10^{24} / (0.521)^3 is present. Also, there is a factor of 1/(2 pi), and also 
        a factor 2, since the integration should be performed for both positive and negative kz values, but the band structure is
        calculated only for positive values of kz.

        Alternatively one can express the both the wave functions and the kz in dimensional units outside...
        '''
        # check consistency of input parameters
        if modified_EFA is True:
            if norm_sum_region is None:
                raise ValueError("norm_sum_region equal to None is invalid for modified EFA")

        # fermi occupation for electrons and holes
        occupation_factor_el = self.fermi_occ(e, mu, temp, particle='electron')
        occupation_factor_h = self.fermi_occ(e, mu, temp, particle='hole')

        self.f_occ_el = occupation_factor_el
        self.f_occ_h = occupation_factor_h
        
        # dk enters in A^{-1}, so convert first to adimensional units, multiply by a0
        dk = dk * _constants.length_scale
        
        """Lapushkin method (citation):
        Take the integral of the wave function modulus square in different regions.
        According to the region considered, if this integral is more than a predefined threshold,
        let's say 0.9, the state at that k-value and at that m-subband is assigned to the corresponding
        kind of particle. 
        The vector mask_el and mask_h selects these states.

        All the states that are nither pure electrons or holes are occupied in a different way;
        the region of the interpolation point defines the kind of charge: for example
        InAs --> electrons, GaSb--> holes.

        It is also possible that we have a single region... should be corrected

        What do I chose to compute the relative change with another charge? 
        """
        if modified_EFA is True:
            mask_el = norm_sum_region[:,0,:]>=thr_el
            mask_h = norm_sum_region[:,1,:]>=thr_h
        else:
            if particle=="electron":
                mask_el = np.full(occupation_factor_el.shape, 1)
                mask_h = np.full(occupation_factor_h.shape, 0)
            elif particle=="hole":
                mask_el = np.full(occupation_factor_el.shape, 0)
                mask_h = np.full(occupation_factor_h.shape, 1)
            else:
                raise ValueError("Invalid particle kind")

        mask_elh = np.logical_not(mask_el | mask_h).astype(int)
        
        # add to charge matrices of hybridized states
        # the charge of pure electrons and holes states is set to zero
        f_occ_el = occupation_factor_el.copy()
        f_occ_h = occupation_factor_h.copy()
        f_occ_el[np.logical_not(mask_elh)] = 0.0
        f_occ_h[np.logical_not(mask_elh)] = 0.0
        charge_matrix_el_add, charge_matrix_h_add = self._get_charge_matrix( psi , f_occ_el, f_occ_h )
        self.charge_matrix_el += charge_matrix_el_add * dk * 0.5 / np.pi * 2.0 * (1e24 / _constants.length_scale**3)
        self.charge_matrix_h += charge_matrix_h_add * dk * 0.5 / np.pi * 2.0 * (1e24 / _constants.length_scale**3)

        # add to charge matrices of pure electrons and holes
        # the charge of hybridized states is set to zero
        f_occ_el = occupation_factor_el.copy()
        f_occ_h = occupation_factor_h.copy()
        f_occ_el[np.logical_not(mask_el)] = 0.0
        f_occ_h[np.logical_not(mask_h)] = 0.0
        charge_matrix_el_add, charge_matrix_h_add = self._get_charge_matrix( psi , f_occ_el, f_occ_h )
        self.charge_matrix_el_pure += charge_matrix_el_add * dk * 0.5 / np.pi * 2.0 * (1e24 / _constants.length_scale**3)
        self.charge_matrix_h_pure += charge_matrix_h_add * dk * 0.5 / np.pi * 2.0 * (1e24 / _constants.length_scale**3)

        # this is non-sense now...
        self.ch_dens = np.real(self.charge_matrix_el.diagonal())[self.fs.dlnc_cumul[:-1]]
        self.n = np.real(self.charge_matrix_el.diagonal())[self.fs.dlnc_cumul[:-1]]
        
        # to fix
        self.f_occ = occupation_factor_el

    def _get_charge_matrix(self, psi, occupation_factor_el, occupation_factor_h):
        # computes finite element global matrices
        
        # integers
        nn = self.fs.dlnc.sum() #* psi.shape[2] # total dof x ncomp
        #nq = np.array(nn, dtype='int64')**2
        nq = np.sum(self.fs.tdof_per_elem**2)
        row = np.empty(nq, dtype=int)
        col = np.empty(nq, dtype=int)
        data_el = np.empty(nq, dtype=complex)
        data_h = np.empty(nq, dtype=complex)
        j=0
        for fel in self.fs.felems:
            kloce = _common.get_loce(kconec=fel.nods, dlnc_cumul=self.fs.dlnc_cumul, ncom=1)
            nd = kloce.shape[0]
            ndq = nd**2
            # get coo array
            row[j:j+ndq] = np.repeat(kloce, nd)
            col[j:j+ndq] = np.tile(kloce,nd)
            
            # compute matrix element
            nodeloc = kloce
            psi_nod = psi[:,nodeloc,:,:]
            rho_mat_el = get_charge_element_matrix(psi_nod, occupation_factor_el)
            rho_mat_h = get_charge_element_matrix(psi_nod, occupation_factor_h)
            data_el[j:j+ndq] = rho_mat_el.flatten()
            data_h[j:j+ndq] = rho_mat_h.flatten()
            j+=ndq
            
        c_el = coo_matrix((data_el, (row, col)), dtype=complex, shape=(nn,nn))
        c_h = coo_matrix((data_h, (row, col)), dtype=complex, shape=(nn,nn))

        dok_el = dok_matrix((c_el.shape),dtype=c_el.dtype)
        dok_el._update(zip(zip(c_el.row,c_el.col),c_el.data))
        new_c_el = dok_el.tocsc() 

        dok_h = dok_matrix((c_h.shape),dtype=c_h.dtype)
        dok_h._update(zip(zip(c_h.row,c_h.col),c_h.data))
        new_c_h = dok_h.tocsc()

        return new_c_el, new_c_h

    def interp(self, coords, total=True, values=None):
        x = coords[:,0]
        y = coords[:,1]
        charge=np.zeros(coords.shape[0])
        n = np.zeros(coords.shape[0])
        p = np.zeros(coords.shape[0])

        iels = self.fs.mesh.trifinder(x,y)
        for i in np.arange(len(iels),dtype=int)[iels>=0]:
            iel = iels[i]
            fel = self.fs.felems[iel]
            particle = self.fs.mesh.particle_per_elem[iel]
            nodeloc = _common.get_loce(kconec=fel.nods, dlnc_cumul=self.fs.dlnc_cumul, ncom=1)
            indexer = np.meshgrid(nodeloc,nodeloc)
                
            coefficient_matrix_el_pure = self.charge_matrix_el_pure[indexer[1],indexer[0]].toarray()
            coefficient_matrix_h_pure = self.charge_matrix_h_pure[indexer[1],indexer[0]].toarray()

            # review this
            if particle == 'electron':
                coefficient_matrix = self.charge_matrix_el[indexer[1],indexer[0]].toarray()
            elif particle == 'hole':
                coefficient_matrix = self.charge_matrix_h[indexer[1],indexer[0]].toarray()
            else:
                raise ValueError('invalid particle kind')
            csi, eta = fel.get_refcoord(x[i],y[i])
            N = fel.shape.fun(csi,eta)
            NN = np.outer(fel.T.T @ N,fel.T.T @ N)

            if particle == 'electron':
                coefficient_matrix = self.charge_matrix_el[indexer[1],indexer[0]].toarray()
                n[i] += np.real(np.sum(coefficient_matrix * NN))
            elif particle == 'hole':
                coefficient_matrix = self.charge_matrix_h[indexer[1],indexer[0]].toarray()
                p[i] += np.real(np.sum(coefficient_matrix * NN))
            else:
                raise ValueError('invalid particle kind')
            n[i] += np.real(np.sum(coefficient_matrix_el_pure * NN))
            p[i] += np.real(np.sum(coefficient_matrix_h_pure * NN))
            
            # add background charges if present
            if self.background_charge_list:
                for bgc in self.background_charge_list:
                    charge[i] += bgc[fel.nreg]
        if total:
            return n + p
        else:
            return n, p 

    def get_linchdens(self):
        # integrate charge density on total 2D domain
        linchdens = 0.0
        if self._projection_method:
            linchdens = self.ch_dens.sum() * self.rectangular_mesh.dxs * self.rectangular_mesh.dys
        else:
            for fel in self.fs.felems:
                gauss_coords = fel.gauss_coords
                f = self.interp(gauss_coords)
                linchdens += fel.int_f( f )
        # integrals has to be done in non-rescaled coordinate in [cm^2]
        # the constant factor is a0^2 that is the square of the bohr radius expressed in cm^2 (1cm = 10^-8 A)
        linchdens *= _constants.length_scale**2 * 1e-16
        return linchdens
    
    def get_total_charge(self):
        # integrate charge density on total 2D domain
        ntot = 0.0
        ptot = 0.0
        for fel in self.fs.felems:
            gauss_coords = fel.gauss_coords
            n, p = self.interp(gauss_coords, total=False)
            ntot += fel.int_f( n )
            ptot += fel.int_f( p )
        # integrals has to be done in non-rescaled coordinate in [cm^2]
        # the constant factor is a0^2 that is the square of the bohr radius expressed in cm^2 (1cm = 10^-8 A)
        ntot *= _constants.length_scale**2 * 1e-16
        ptot *= _constants.length_scale**2 * 1e-16
        return ntot, ptot       

    def integrate(self, density):
        # integrate charge density on total 2D domain
        integral = np.sum(density * self.rectangular_mesh.dxs * self.rectangular_mesh.dys) * _constants.length_scale**2 * 1e-16
        return integral



################################################################################################################
################################################################################################################

class DopingChargeDensity:
    def __init__(self, doping_concentration_value=None, region_fun=None):

        self.doping_concentration_value = doping_concentration_value

        self.region_fun = region_fun
        """
        An example of function that returns put doping inside an hexagon
        Be careful with rescaled unit !
        def hexagon(coords):
            width=10
            s=width/2. / np.cos(np.pi/6.)
            mask = np.full(coords.shape[0],False)
            x, y = map(abs,coords.T)
            for i in range(coords.shape[0]):
                mask[i] = y[i] < 3**0.5 * min(s - x[i], s / 2)
            return mask
        """

    def interp(self, coords):
        doping_concentration = np.zeros(coords.shape[0])
        mask = self.region_fun(coords)
        doping_concentration[mask] = self.doping_concentration_value
        return doping_concentration

################################################################################################################
################################################################################################################

class WaveFunction:
    def __init__(self, fs, psi):
        
        # fs : finite element space object
        # psi : ndarray, solution vector of dimension (nk, ntot, ncom, neig)
        
        self.fs = fs
        self.psi = psi

        self.nk = psi.shape[0]
        self.ncom = psi.shape[2]
        self.neig = psi.shape[3]

        self.symop = None

        # angular momentum matrix
        self.Fz = None

    def interp(self, coords, psi=None, symop=None):
        # if a symmetry operation has been performed on the wave function

        if psi is None:
            psi = self.psi

        if symop is not None:
            # the symmetry operation is defined on a "oblique" cartesian system (a=b, gamma=120 deg)
            coords_uv = U @ coords.T
            # rotation in the uv basis 
            coords_uv_rot = np.linalg.inv(symop.mat) @ coords_uv
            #coords_uv_rot = symop.mat @ coords_uv
            # back to standard cartesian basis + transpose
            coords = (Uinv @ coords_uv_rot).T
        x = coords[:,0]
        y = coords[:,1]
        npts = coords.shape[0]
        s = np.zeros((self.nk, npts, self.ncom, self.neig), dtype=complex)

        iels = self.fs.mesh.trifinder(x, y)
        for i in np.arange(len(iels),dtype=int)[iels>=0]:
            
            # finite element object
            iel = iels[i]
            fel = self.fs.felems[iel]
            
            # get nodal sol on this element
            nodeloc = _common.get_loce(kconec=fel.nods, dlnc_cumul=self.fs.dlnc_cumul, ncom=1)
            #snod = self.psi[:, nodeloc, :, :]
            snod = psi[:, nodeloc, :, :]
            snod = np.einsum('jikl',snod, optimize="greedy")
            
            s[:, i, :, :] = fel.interp_sol( x[i] , y[i] , snod).T
        return s
    
    #def get_orbital_angular_momentum(self, component="z"):
    #    # get the expectation value for (kz, ncom, neig)
    #    expval = np.zeros((self.nk, self.ncom, self.neig))
    #    # loop over the elements and compute elemental expectation values
    #    for fel in self.fs.felems:
    #        # get the loce vector
    #        nodeloc = _common.get_loce(fel.nods, self.fs.dlnc_cumul, ncom=1)
#
    #        # get the position of the gauss points on this element
    #        xg = fel.gauss_coords[:,0]
    #        yg = fel.gauss_coords[:,1]
#
    #        # integral matrices matrices
    #        # \int psi x dpsi/dy
    #        psi_x_psiy_mat = fel.int_psi_f_psiy(xg)
    #        # \int psi y dpsi/dx
    #        psi_y_psix_mat = fel.int_psi_f_psix(yg)
#
    #        # get solution on the nodes
    #        psinod = self.psi[:,nodeloc,:,:]
#
    #        integrand = -1.j * (psi_x_psiy_mat - psi_y_psix_mat) # hbar=1 (in a.u.)
    #        expval_iel =  np.einsum('ijkl, jo, iokl ->ikl', np.conjugate(psinod), integrand, psinod).real
    #        expval += expval_iel
    #    return expval
    
    def get_dominant_total_angular_momentum(self, mj_values, neig=100):

        # spin angular momentum
        comps=np.s_[2:8]
        Qmat = np.diag([1.,-1.j, -1,-1.j,1.j,1,-1,-1.j])
    
        Jz = np.diag(v = np.array([0.5, -0.5, 1.5, -1.5, 0.5, -0.5, 0.5, -0.5]))
        Jz = np.conjugate(Qmat) @ Jz @ Qmat.T
        Jz = Jz[comps,comps]

        # add a small magnetic field along the z-direction to break the spin degeneracy
        theta = np.arccos(1./np.sqrt(3.))
        phi = np.pi/4.
        mat = hamiltonian.P(theta, phi)
        sigma_z = np.diag(v=np.array([1, -1, 1, 1, 1, -1, -1, -1])) * 0.5
        sigma_z_j = mat.conj() @ sigma_z @ mat.T
        sigma_z_j = sigma_z_j[comps,comps]
        
        # want it to be a very small perturbation
        #sigma_z_j *= 1e-25
    
        ###### ASSEMBLY ########
        nn = self.fs.dlnc.sum() * self.ncom
        #nq = np.array(nn, dtype='int64')**2
        nq = np.sum((self.fs.tdof_per_elem * self.ncom)**2)
        row = np.empty(nq, dtype=int)
        col = np.empty(nq, dtype=int)
        data = np.empty(nq, dtype=complex)
        rowB = np.empty(nq, dtype=int)
        colB = np.empty(nq, dtype=int)
        dataB = np.empty(nq, dtype=complex)
        j=0
        k=0
        for fel in self.fs.felems:
            # get coo arrays
            kloce = _common.get_loce(kconec=self.fs.conec[fel.iel], dlnc_cumul=self.fs.dlnc_cumul, ncom=self.ncom)
            nd = kloce.shape[0]
            
            # compute Lz + Jz element matrix 
            aelm = np.kron(fel.Lz , np.eye(self.ncom)) + np.kron(fel.I, Jz  ) #+ np.kron(fel.I, sigma_z_j) 
            belm = np.kron(fel.I, np.eye(self.ncom))

            # eliminate zero entries in element matrices for Lz + Jz
            mask_data_aelm = np.nonzero(aelm)
            data_aelm = aelm[mask_data_aelm]
            data[j:j+len(data_aelm)] = data_aelm
            row[j:j+len(data_aelm)] = kloce[mask_data_aelm[0]]
            col[j:j+len(data_aelm)] = kloce[mask_data_aelm[1]]

            # eliminate zero entries in element matrices for B
            mask_data_belm = np.nonzero(belm)
            data_belm = belm[mask_data_belm]
            dataB[j:j+len(data_belm)] = data_belm
            rowB[j:j+len(data_belm)] = kloce[mask_data_belm[0]]
            colB[j:j+len(data_belm)] = kloce[mask_data_belm[1]]

            j += len(data_aelm)
            k += len(data_belm)
        
        data = data[:j]
        row = row[:j]
        col = col[:j]

        dataB = dataB[:k]
        rowB = rowB[:k]
        colB = colB[:k]

        # build global sparse matrices
        self.Fz = coo_matrix((data, (row, col)), shape=(nn, nn))
        self.Fz.eliminate_zeros()
        self.Fz = triu( self.Fz.tocsr() )
        self.Fz += tril(self.Fz.getH(), k=-1)

        self.B = coo_matrix((dataB, (rowB, colB)), shape=(nn, nn))
        self.B.eliminate_zeros()
        self.B = triu( self.B.tocsr() )
        self.B += tril(self.B.getH(), k=-1)

        ###### BOUNDARY CONDITIONS ########

        bn = self.fs.mesh.bn
        to_delete = []
        for i in range(bn.shape[0]):
            to_delete.append(_common.get_loce(np.array([bn[i]]), self.fs.dlnc_cumul, ncom=self.ncom)[:self.ncom])
        to_delete = np.hstack(to_delete)

        #self.to_delete = to_delete
        Fz = delete_from_csr(self.Fz, row_indices=to_delete, col_indices=to_delete )
        B = delete_from_csr(self.B, row_indices=to_delete, col_indices=to_delete )

        # we need to compute many eigenvalues of this matrix. 
        # It's better to use a kind of eigenvalue slicing technique. We already know where the
        # eigenvalue density is going to be higher: 1/2, 3/2 ...

        sigma_values = mj_values

        eigvals = np.zeros((len(sigma_values), neig))
        eigvecs = np.zeros((len(sigma_values), Fz.shape[0], neig), dtype=complex)

        for i in range(len(sigma_values)):
            # diagonalize total angular momentum z-component
            ###### DIAGONALIZATION ####### 
            eigvals_mj, eigvecs_mj = eigsh(
                A = Fz,
                M = B,
                k=neig,
                which='LM',
                v0=None,
                sigma = sigma_values[i],
                tol=1e-16,
                OPinv=None
            )

            # the eigenvectors corresponding to degenerate subspace may not be orthogonal...
            # I now post-normalize them using the AM operator
            Fzs = eigvecs_mj.conj().T @ Fz @ eigvecs_mj
            Bs = eigvecs_mj.conj().T @ B @ eigvecs_mj
            _, Vjk = eigh(Fzs, Bs)
            eigvecs[i,:,:] = eigvecs_mj @ Vjk
            eigvals[i,:] = np.diagonal(eigvecs_mj.conj().T @ Fz @ eigvecs_mj).real

        # want a single vector
        eigvals = np.hstack(eigvals)
        eigvecs = np.hstack(eigvecs)

        ###### POSTPROCESSING #######        
        ntot =  self.fs.dlnc.sum() * self.ncom     
        a = np.full( ntot , 1 , dtype=bool )
        to_zero = to_delete
        a[to_zero] = False
        post_processed_eigvecs = np.zeros(( ntot , eigvecs.shape[1] ), dtype=complex)
        post_processed_eigvecs[a,:] = eigvecs
        eigvecs = post_processed_eigvecs

        eigvals, eigvecs = sort_eig(eigvals, eigvecs, 'electron')

        # now I need to evaluate the projections of the original eigenstates onto
        # the total angular momentum eigenstates I have just computed

        eigvecs, spinor_dist, norm_sum, norm_sum_region = get_spinor_dist(self.fs, eigvecs, self.ncom)
        
        eigvecs = eigvecs / np.sqrt(norm_sum)

        eigvecs_psi = self.psi.reshape(self.nk, self.fs.dlnc.sum()*self.ncom,-1)

        coeff = np.zeros((self.nk,eigvecs.shape[1], eigvecs_psi.shape[2]), dtype=complex )

        #mj_dominant = np.zeros((self.nk, self.psi.shape[-1]))
        mj_proj = np.zeros((self.nk, mj_values.shape[0], self.psi.shape[-1]))

        for i in range(self.nk):
            coeff[i,:,:] = eigvecs.conj().T @ self.B @ eigvecs_psi[i,:,:]
            # normalize
            norm = np.sum(np.abs(coeff[i,:,:])**2,axis=0)
            coeff[i,:,:] = coeff[i,:,:] / np.sqrt(norm)
            # for each subband
            for j in range(mj_proj.shape[2]):
                a = np.abs(coeff[i,:,:][:,j])**2
                aa = np.zeros(mj_values.shape[0])
                r=0.5
                for k in range(len(mj_values)):
                    mj = mj_values[k]
                    mask = np.where(np.abs(eigvals - mj) <= r)[0]
                    mj_proj[i,k,j] = a[mask].sum()

        mj_proj = mj_proj[:,::-1,:][:,len(aa)//2:,:] + mj_proj[:,len(aa)//2:,:]

        return eigvals, eigvecs, coeff, mj_proj

    def symmetrize_to_angular_momentum(self, energy_spectrum=None):

        # I want to symmetrize according to the operator H + Fz
        
        # coefficient matrices
        dim = self.psi.shape[-1]
        V = np.zeros((self.nk, dim, dim), dtype=complex)
        expval_Fz = np.zeros((self.nk, dim))

        # here we compute the quantity: alphaH * H|psi>. Remember that H|psi> = E|psi>.
        alphaH_Hpsi_kz = np.zeros((self.nk,self.ncom))
        if energy_spectrum is not None:
            E0 = np.max(energy_spectrum,axis=1)
            alphaH_Hpsi_kz = (energy_spectrum - E0[:,np.newaxis])/(np.min(energy_spectrum,axis=1)-E0)[:,np.newaxis]

        eigvecs = self.psi.reshape(self.nk, self.fs.dlnc.sum()*self.ncom,-1)
        for nk in range(self.nk):

            # angular momentum projection
            O = eigvecs[nk,:,:].conj().T @ self.Fz @ eigvecs[nk,:,:]

            # add diagonal hamiltonian
            O += np.diag(v=alphaH_Hpsi_kz[nk,:])
            
            # coefficient matrices
            _, Vnk = np.linalg.eig(O)
            #mask = np.argsort(np.abs(Vnk)**2, axis=0)[-2::,:].T
            #Vnk = Vnk[:,np.argsort(mask[:,0])]
            V[nk,:,:] = Vnk

        # a new symmetrized wave function is obtained
        psi_Fz_symm = np.einsum("ilm,ijkl->ijkm", V, self.psi)
        eigvecs_Fz_symm = psi_Fz_symm.reshape(self.nk, self.fs.dlnc.sum()*self.ncom,-1)


        # evaluate the expectation value of the total angular momentum for each kz
        for nk in range(self.nk):
            expval_Fz[nk,:] = np.diagonal(eigvecs_Fz_symm[nk,:,:].conj().T @ self.Fz @ eigvecs_Fz_symm[nk,:,:]).real

        # two things are missing...
        # 1) evaluate the corrected energies by computing the expectation value of H in this new basis
        # 2) assign to each new eigenstate teh corresponding angular momentum. (The order in energy should correspond
        # to the order in |Fz|)
        
        return expval_Fz, V

    def get_spin_angular_momentum(self, spinor_dist=None, component="z"):
        expval = np.zeros((self.nk, self.ncom, self.neig))
        if component=="z" and spinor_dist is not None:
            # we can simply multiply the angular momentum matrix Jz 
            # by the |psi|**2 for each component
            expval = np.einsum('ijk,j->ijk', spinor_dist, np.diagonal(hamiltonian.Jz))
        else:
            raise ValueError("other components are not allowed up to now...")
        return expval
    
    def apply_symmetry_operator(self, symop=None):
        # this method should change the envelope function spinor vector first.
        # the next step is to evaluate the (x',y') transformed coordinates, according to this symmetry operator.
        # I just need the matrix R^-1 that will multiply thr "coords" vector in the interpolation method.

        self.symop = symop

        self.psi = np.einsum('kl,ijlm->ijkm',symop.basis_op, self.psi)

        self.rotmat = symop.coord_op

    def apply_symmetry_operator(self, symop=None):
        # this method should change the envelope function spinor vector first.
        # the next step is to evaluate the (x',y') transformed coordinates, according to this symmetry operator.
        # Here we assume the mesh is invariant under the symmetry operation
        permutation = self.fs.mesh.symops[symop.name]
        permutation = _common.get_loce(permutation, self.fs.dlnc_cumul, ncom=1)
        operator = symop.op
        psip = np.einsum('kl,ijlm->ijkm',operator, self.psi)
        psip = psip[:,permutation,:,:]
        return psip

    def project_irrep(self, symg, irrep_label, partner_fun_label, coords):
        # this method will project out (and interpolate on coords)
        # the part of the wave function belonging to a certain irrep of the symmetry group
        
        # dimension of the representation
        l = symg.irrep[irrep_label]['l']

        mu = partner_fun_label

        # number of operations in the group
        h = symg.h
        
        s=0 # the final interpolated solution 

        # loop over the symmetry operations of the group
        for symop_name in symg.symops.keys():
            symop = symg.symops[symop_name]
            self.symop = symop
            
            # symmetry operation on the wave function spinor
            # we have to multiply by the character of the representation, that is the sum of the 
            # diagonal elements of the j-th representation (here I chose G7)
            #psi = np.einsum('kl,ijlm->ijkm',symop.op, self.psi) * np.trace(np.conjugate(symg.G6[symop_name]))
            psi = np.einsum('kl,ijlm->ijkm',symop.op, self.psi) * np.conjugate(symg.irrep[irrep_label][symop_name][mu,mu])#* np.trace(symg.G5[symop_name])

            # now I have to perform the interpolation on a set of coordinates
            s += self.interp(coords, psi, symop)

        s = s * (l/h)
        return s
    
    def project_irrep_integrate(self,symg):
        # need to know the dimensions of the weights matrix
        
        # the result is stored in a dictionary: irrep_name --> projection coefficients
        projections = {irrep_name:0 for irrep_name in symg.irrep_names}
        integral = np.zeros((symg.ltot, self.nk, self.neig))
        # loop over the elements and compute elemental expectation values
        # get the loce vector
        #nodeloc = _common.get_loce(fel.nods, self.fs.dlnc_cumul, ncom=1)
        wt = self.fs.felems[0].shape.gpar.wt

        # get the position of the gauss points on this element
        coords = self.fs.gauss_coords_global
        
        i=0
        for irrep_label in symg.irrep.keys():
            for partner_fun_label in range(symg.irrep[irrep_label]['l']):

                irrep_name = symg.irrep_names[i]
    
                # projection on the irrep
                psiq_j = np.abs(self.project_irrep(symg, irrep_label, partner_fun_label, coords))**2
                psiq_j = psiq_j.reshape(psiq_j.shape[0], psiq_j.shape[1]//wt.shape[0], wt.shape[0], psiq_j.shape[2] ,psiq_j.shape[3])
                psiq_j = np.einsum('ijklm,j->ijklm', psiq_j, self.fs.detJ_per_elem)
                integral = np.einsum('ijklm,k->im', psiq_j, wt)
                projections[irrep_name] = integral
                i+=1
        
        return projections
    

    def apply_class_operator(self, class_operator, coords):

        s=0 # the final interpolated solution 

        # Questa funzione e troppo lenta...

        # loop over the symmetry operations of the group
        for symop_name in class_operator.symops.keys():
            symop = class_operator.symops[symop_name]
            self.symop = symop
            
            # symmetry operation on the wave function spinor
            psi = np.einsum('kl,ijlm->ijkm',symop.op, self.psi, optimize="greedy")
             # now I have to perform the interpolation on a set of coordinates
            s += self.interp(coords, psi, symop)
        return s
    
    def apply_class_operator(self, class_operator):

        # loop over the symmetry operations of the group
        psip=0.0
        for symop_name in class_operator.symops.keys():
            symop = class_operator.symops[symop_name]
            # symmetry operation on the wave function spinor
            psip += self.apply_symmetry_operator(symop)
        return psip

    def get_class_operator(self, class_operators, kz0 = True):
        
        # if more than one class operator is given, the sum the the matrix presentatins of the operators
        # will be performed

        # by default the PTCO at kz=0 is not applied since the 
        # symmetry group can in principle be different
        slice = np.s_[1:]

        if kz0:
            # apply PTCO also at kz=0
            # I define this slice to modify the wave function at kz=0 too using PTCO
            slice = np.s_[:]

        # the dimension is given by the dimension of the solution subspace chosen
        dim = self.psi.shape[-1]

        Otot = np.zeros((self.nk, dim, dim), dtype=complex)
        
        # I want the integral to be performed once...

        # gauss weights
        wt = self.fs.felems[0].shape.gpar.wt

        # get the position of the gauss points on the entire mesh
        coords = self.fs.gauss_coords_global

        # now I need to apply the class operator to each eigenstate of the hamiltonian,
        # including Kramers doublets and wave vector dependance. I want the dimensions (nk,gausspts,ncom,neig)

        # I have to apply different classes operators, need a small loop...
        # for each class operator there is a loop on the symmetry operations included
        Cpsi_j = 0.0
        for class_operator in class_operators:
            Cpsi_j += self.apply_class_operator(class_operator, coords)
        # the "full" class operator has acted on the wave function spinor, and result has been interpolated on
        # all the gauss points of the mesh, for each wave vector, component and subband index

        # now need to compute the matrix elements. To do that I interpolate the wave function, take the complex conjugate,
        # multiply and then integrate this quantity. I have to take the cross product for the subband index.

        # interpolation and complex conjugate
        psi_i = np.conjugate( self.interp(coords) )

        # multiply
        # let me use a loop on kz
        #for nk in range(self.psi.shape[0]):
        for nk in range(self.nk):
            O = np.einsum("jkl,jkm->jklm", psi_i[nk] , Cpsi_j[nk], optimize="greedy") 
            #O = np.einsum("ijkl,ijkm->ijklm", psi_i , Cpsi_j, optimize="greedy")
        
            # reshape before the integral. I want (nk,nelem,ngauss,ncom, neig,neig)
            # so I can write the integral as a simple sum on the gauss points. 
            O = O.reshape(O.shape[0]//wt.shape[0], wt.shape[0], O.shape[1] , O.shape[2], O.shape[3])
            #O = O.reshape(O.shape[0], O.shape[1]//wt.shape[0], wt.shape[0], O.shape[2] , O.shape[3], O.shape[4])

            # multiply each elemental integral by its jacobian
            O = np.einsum('jklmn,j->jklmn', O, self.fs.detJ_per_elem)
            #O = np.einsum('ijklmn,j->ijklmn', O, self.fs.detJ_per_elem)
        
            # multiply gauss weights and sum on all indices except the first (nk), and the last two (neig,neig)
            #O = np.einsum('ijklmn,k->imn', O, wt)
            O = np.einsum('jklmn,k->mn', O, wt)

            Otot[nk,:,:] = O
        return Otot
    
    #def get_class_operator(self, class_operators, kz0 = True):
    #    
    #    # if more than one class operator is given, the sum the the matrix presentatins of the operators
    #    # will be performed
#
    #    # by default the PTCO at kz=0 is not applied since the 
    #    # symmetry group can in principle be different
    #    slice = np.s_[1:]
#
    #    if kz0:
    #        # apply PTCO also at kz=0
    #        # I define this slice to modify the wave function at kz=0 too using PTCO
    #        slice = np.s_[:]
#
    #    # the dimension is given by the dimension of the solution subspace chosen
    #    dim = self.psi.shape[-1]
#
    #    Otot = np.zeros((self.nk, dim, dim), dtype=complex)
#
    #    # now I need to apply the class operator to each eigenstate of the hamiltonian,
    #    # including Kramers doublets and wave vector dependance. I want the dimensions (nk,gausspts,ncom,neig)
#
    #    # I have to apply different classes operators, need a small loop...
    #    # for each class operator there is a loop on the symmetry operations included
    #    Cpsi_j = 0.0
    #    for class_operator in class_operators:
    #        Cpsi_j += self.apply_class_operator(class_operator)
    #    # the "full" class operator has acted on the wave function spinor
#
    #    # now need to compute the matrix elements. I have to take the cross product for the subband index.
    #    U = self.psi.reshape(self.nk, self.fs.dlnc.sum()*self.ncom,-1)
    #    Utilde = Cpsi_j.reshape(self.nk, self.fs.dlnc.sum()*self.ncom,-1)
#
    #    # let me use a loop on kz
    #    for nk in range(self.nk):
#
    #        Otot[nk,:,:] = U[nk].conj().T @ B @ Utilde[nk]
    #    return Otot
        
# the coefficient matrix for each value of k has been evaluated. Now we can use it 
# to write down the new eigenstates of the hamiltonian with psi_k = sum_j Vjk * psi_j

# a new symmetrized wave function is obtained
#self.psi[slice] = np.einsum("ilm,ijkl->ijkm", V, self.psi)[slice]

#    def get_energy_expval(self, H):
#        energy = np.zeros((self.nk, self.neig))
#        for nk in range(self.nk):
#            U = self.psi.reshape(self.nk, self.fs.dlnc.sum()*self.ncom,-1)


################################################################################################################
################################################################################################################

class ElectrostaticPotential:
    def __init__(self, fs, V=None, electric_field=(0.0,np.pi*0.5)):
        
        # fs : finite element space object
        # epot : ndarray, electrostatic energy vector 
        # Ey : (modulus of electric field V/micrometers, angle wiht x-axis clockwise)
        
        self.fs = fs

        # potential initialize to zero everywhere
        self.V = np.zeros(self.fs.mesh.vertices.shape[0])
        
        # add external electric field potential energy
        E = electric_field[0] # modulus of the electric field vector
        theta = electric_field[1] # theta angle wiht x-axis clockwise

        # compute the potential energy due to the electric field along y
        # V = -e E \cdot r
        # multiply by a0 to convert the electric field in adimensional units 
        # 1e-4 is because a0 is in 1e-10m, while Ey is in V/1e-6m
        # the position vector is already in adimensional units.
        # consequently the elec potential energy is expressed in eV
        # when it enters in the schrod equation will be converted to adimensional 
        # deviding by Ha.
        Ux = - E * np.cos(theta) * self.fs.mesh.vertices[:,0] * _constants.length_scale * 1e-4
        Uy = - E * np.sin(theta) * self.fs.mesh.vertices[:,1] * _constants.length_scale * 1e-4
        self.V += Ux + Uy

        # add an electrostatic potetnial energy in eV defined on the mesh
        if V is not None:
            self.V += V

    # interpolate potential on coords
    def interp(self, coords):
        x = coords[:,0]
        y = coords[:,1]
        npts = coords.shape[0]
        s = np.zeros(npts, dtype=float)

        iels = self.fs.mesh.trifinder(x, y)
        for i in np.arange(len(iels),dtype=int)[iels>=0]:
            
            # finite element object
            iel = iels[i]
            fel = self.fs.felems[iel]
            
            # get nodal sol on this element
            nodeloc = _common.get_loce(kconec=fel.nods, dlnc_cumul=self.fs.dlnc_cumul, ncom=1)
            snod = self.V[nodeloc]
            #snod = np.einsum('jikl',snod, optimize="greedy")
            
            s[i] = fel.interp_sol( x[i] , y[i] , snod).T
        return s

    def mean_value(self):
        mean=0.0
        for fel in self.fs.felems:
            f = fel.interp_sol_gauss(self.V[fel.nods])
            mean +=fel.int_f(f)

        return mean
    
    # plot electrostatic potential
    def plot(
            self,  
            xlim, 
            ylim,
            figsize=(5,5), 
            subdiv=1, 
            cmapin='rainbow', 
            levels=21, 
            fontsize=20, 
            polygons=None
        ):
    
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0) 
        ax = fig.add_subplot(gs[0,0])

        xlim_l = xlim[0]
        xlim_r = xlim[1]
        ylim_l = ylim[0]
        ylim_r = ylim[1]

    
        triangles = self.fs.mesh.triangles[:,:3]
        x = self.fs.mesh.vertices[:,0] #*staticdata.br / 10 # nm
        y = self.fs.mesh.vertices[:,1] #*staticdata.br / 10 # nm
        trigrid = tri.Triangulation(x, y, triangles )
        ref_trigrid = UniformTriRefiner(trigrid)
        trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
        vert = np.vstack([trigrid.x, trigrid.y]).T
        triangles = trigrid.triangles
        

        V = np.zeros(len(vert))
        V = self.interp(vert)
    
        ax.tick_params('both', labelsize=15) 
        
        ch2D = ax.tricontourf(
            vert[:,0] * _constants.length_scale / 10, # plot in nm
            vert[:,1] * _constants.length_scale / 10,# plot in nm
            triangles,
            V * 1e3, # plot in meV
            levels=levels,
            cmap=cmapin
        )
    
        ax.tick_params('both', labelsize=15)
        
        cbaxes = inset_axes(ax, width="5%", height="90%",loc='center left', bbox_to_anchor=(0-0.1, 0, 1, 1), bbox_transform=ax.transAxes) 
        cbar = plt.colorbar(ch2D,
                     cax=cbaxes,
                     orientation='vertical')
        #cbar.ax.set_yticks(ticks=ticks_cb,labels=ticks_cb)
        cbar.ax.tick_params(labelsize=fontsize+3)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.set_ylabel(r'-e$\phi$ $[meV]$', fontsize=fontsize+2)
        cbar.ax.yaxis.set_ticks_position('left')

        if polygons is not None:
            for poly in polygons:
                newpoly = copy.copy(poly)
                ax.add_patch(newpoly)
    
        #ax.set_title(r'$n_{h}$ $[10^{16} \mathrm{cm}^{-3}]$', fontsize=fontsize)
    
        #ax.set_xlabel('position [nm]', size = fontsize, color='black')
        #ax.set_ylabel('position [nm]', size = fontsize, color='black')   

        ax.set_ylim(ylim_l,ylim_r)
        ax.set_xlim(xlim_l,xlim_r)
        ax.set_aspect('equal', adjustable='box')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.spines["bottom"].set_linewidth(spines_lw)
        #ax.spines["left"].set_linewidth(spines_lw) 
        return fig

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

# Helper functions

def get_charge_element_matrix(psi_nod, occupation_factor):
    # indice dei nodi
    # psi_nod : (nk,ndof_tot_iel,ncom,neig)
    psiq = np.einsum('ijlm,ihlm->ijhm',np.conjugate(psi_nod), psi_nod )
    # psiq : (nk,ndof_tot_iel,ndof_tot_iel,neig)
    psiq_occ = np.einsum('ijlm,im->jl',psiq, occupation_factor )
    return psiq_occ.squeeze()   

# COMPUTE LOCE: LOCALIZATION TABLE

def get_M(vec_k, ax, ay, prime=False):
        
    # dimension of the k-space
    Ny = vec_k.shape[2]
    Nx = vec_k.shape[1]
    
    n=1
    
    # if true, evaluate n' = a - b. By default n = a + b is evaluated
    if prime:
        n=-1
    
    if ax%2==0:
        # then ax is even
        
        # define a b-vector of q-indices (even integers)
        bx = np.arange(-Nx + 2, Nx-1 , 2 , dtype=int)
        nx = (ax + n * bx)//2 + Nx//2 
        
    else:
        # then ax is odd
        
        # define a b-vector of q-indices (odd integers)
        bx = np.arange(-Nx + 1, Nx , 2 , dtype=int)
        nx = (ax + n * bx)//2 + Nx//2         
        
    
    if ay%2==0:
        # then ay is even
        
        # define a b-vector of q-indices (even integers)
        by = np.arange(-Ny + 2, Ny-1 , 2 , dtype=int)
        ny = (ay + n * by)//2 + Ny//2
    else:
        # then ay is odd
        
        # define a b-vector of q-indices (odd integers)
        by = np.arange(-Ny + 1, Ny , 2 , dtype=int)
        ny = (ay + n * by)//2 + Ny//2 
        
    # create padding tuples
    xaxis_pad_width = ((nx<0).sum(),(nx>Nx-1).sum())[::n]
    yaxis_pad_width = ((ny<0).sum(),(ny>Ny-1).sum())[::n]
    
    # create indexing arrays
    nx_idx = nx[ (nx < Nx) & (nx >= 0) ][:,np.newaxis]
    ny_idy = ny[ (ny < Ny) & (ny >= 0) ]
    
    # create a new array in q-coordinate for a given Q
    vec_q = vec_k[:,nx_idx,ny_idy,:,:,:] # controllare questa riga quando nx è diverso da ny
    
    
    # pad with zeros if some q-vector are off-grid
    vec_q = np.pad(vec_q, [(0,0), xaxis_pad_width, yaxis_pad_width, (0,0), (0,0), (0,0)],'constant' )
    
    # compute the matrix index in the full matrix (2N-1, 2N-1)
    idx = bx + (Nx-1)
    idy = by + (Ny-1)
        
    return vec_q, idx[:,np.newaxis], idy


def spiral(X, Y):
    points = []
    x = y = 0
    dx = 0
    dy = -1
    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            points.append((x*2,y*2))
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return points


phiuv = 2.*np.pi/3.
U = np.array([
    [1.0, -(1/np.tan(phiuv))],
    [0.0, 1/np.sin(phiuv)]
])

Uinv = np.array([
    [1.0, np.cos(phiuv)],
    [0.0, np.sin(phiuv)]
])
################################################################################################################
################################################################################################################





