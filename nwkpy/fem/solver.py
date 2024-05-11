import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, coo_matrix, tril, triu, hstack, vstack

from nwkpy import _common
from nwkpy import _constants


def get_spinor_dist(fs, eigvecs, ncom):
    
    ###### COMPUTE SPINOR DISTRIBUTION #######
    nreg = len(fs.mesh.reg2mat.keys())    
    norm = np.zeros((nreg, ncom,  eigvecs.shape[-1]))

    for fel in fs.felems:
        i = fel.nreg - 1 # the correct index since region numbers start from 1
        detJ = fel.detJ
        T = fel.T
        N = fel.shape.N
        wt = fel.shape.gpar.wt
        
        nodeloc = _common.get_loce(fel.nods, fs.dlnc_cumul, ncom=ncom)

        for nu in range(eigvecs.shape[-1]):
            psi_nodal = eigvecs[nodeloc,nu].reshape(( fel.dof_per_node.sum() , ncom)).T
            norm[i,:,nu] += np.abs(psi_nodal @ (T.T @ N))**2 @ wt.T * detJ
    
    norm_sum_region = np.sum(norm , axis=1)
    norm_sum = np.sum(norm_sum_region , axis=0)
    #assert np.allclose(norm_sum , 1.0)
    
    spinor_dist = np.sum(norm, axis=0)
    #eigvecs /= np.sqrt(norm_sum)
    
    # return occupation for len(nuvals) eigenvals
    # dimension(spinor_dist) = ncom x len(nuvals)
    
    return eigvecs, spinor_dist, norm_sum, norm_sum_region   

def sort_eig(eigvals, eigvecs):
    
    # sort according in ascending energy magnitude
    mask = np.argsort(eigvals)   
    eigvecs = eigvecs[:,mask]
    eigvals = eigvals[mask]
    
    return eigvals, eigvecs

#########################################################################

########################### SOLVERS #####################################

#################### FEM GENERALIZED EIGENVALUE PROBLEM #############

class GenEigenProblem:
    def __init__(self):
        
        # hamiltonian global matrix
        self.agl = None
        
        # overlap matrix
        self.bgl = None
        
        # components of the solution (setted when assembly)
        self.ncom = None
        
        # fem space (setted when assembly)
        self.fs = None
        
        # elec pot matrix
        self.epotgl = None
    
    def assembly(self, fs, v):
    
        ###### ASSEMBLY ########
        nn1 = fs.Vfs.dlnc.sum() * 2
        nn2 = fs.Wfs.dlnc.sum() * 6
        nn = nn1 + nn2
        nq = np.sum((fs.Vfs.tdof_per_elem * 2 + fs.Wfs.tdof_per_elem * 6)**2) 
        row_agl = np.empty(nq, dtype=int)
        col_agl = np.empty(nq, dtype=int)
        row_bgl = np.empty(nq, dtype=int)
        col_bgl = np.empty(nq, dtype=int)
        data_agl = np.empty(nq, dtype=complex)
        data_bgl = np.empty(nq, dtype=float)
        j=0
        k=0
        for fel in fs.product_felems:
            # get coo arrays
            kloce1 = _common.get_loce(kconec=fs.Vfs.conec[fel.iel], dlnc_cumul=fs.Vfs.dlnc_cumul, ncom=2)
            kloce2 = _common.get_loce(kconec=fs.Wfs.conec[fel.iel], dlnc_cumul=fs.Wfs.dlnc_cumul, ncom=6)
            # the full kloce is obtained by concatenatinge the two. Attention! The second k loce (hole components)
            # must be displced by the integer nn1 = total_dof(V) * ncom(V)
            kloce = np.concatenate([kloce1, kloce2 + nn1])
            nd = kloce.shape[0]
            
            # compute stiffness and overlap element matrices 
            aelm = v.get_aelm(fel)
            belm = v.get_belm(fel)

            # eliminate zero entries in element matrices
            mask_data_aelm = np.nonzero(aelm)
            mask_data_belm = np.nonzero(belm)
            data_aelm = aelm[mask_data_aelm]
            data_belm = belm[mask_data_belm]

            data_agl[j:j+len(data_aelm)] = data_aelm
            row_agl[j:j+len(data_aelm)] = kloce[mask_data_aelm[0]]
            col_agl[j:j+len(data_aelm)] = kloce[mask_data_aelm[1]]

            data_bgl[k:k+len(data_belm)] = data_belm
            row_bgl[k:k+len(data_belm)] = kloce[mask_data_belm[0]]
            col_bgl[k:k+len(data_belm)] = kloce[mask_data_belm[1]]

            j += len(data_aelm)
            k += len(data_belm)
        
        data_agl = data_agl[:j]
        row_agl = row_agl[:j]
        col_agl = col_agl[:j]

        data_bgl = data_bgl[:k]
        row_bgl = row_bgl[:k]
        col_bgl = col_bgl[:k]

        # build global sparse matrices
        self.agl = coo_matrix((data_agl, (row_agl, col_agl)), shape=(nn, nn))
        self.bgl = coo_matrix((data_bgl, (row_bgl, col_bgl)), shape=(nn, nn))
        self.agl.eliminate_zeros()
        self.bgl.eliminate_zeros()
        self.agl = triu( self.agl.tocsr() )
        self.bgl = triu( self.bgl.tocsr() )

        self.agl += tril(self.agl.getH(), k=-1)
        self.bgl += tril(self.bgl.transpose(), k=-1)

        # set fem space
        self.fs = fs
        self.ncom = v.ncom
        self.v = v
        
    def solve(self, k=6, which='LM', tol=1e-16, v0=None, sigma=None, eigenvalue_shift=None):

        if isinstance(sigma,float) or isinstance(sigma,int):
            sigma = np.array([sigma])
        
        # sigma is in eV, convert to adimensional since the assembled hamiltonian is in atomic units
        sigma = sigma / _constants.energy_scale
        

        if eigenvalue_shift is not None:
            self.agl += eigenvalue_shift['deltaH']
        
        ###### BOUNDARY CONDITIONS ########
        bn = self.fs.Vfs.mesh.bn
        to_delete1 = []
        for i in range(bn.shape[0]):
            to_delete1.append(_common.get_loce(np.array([bn[i]]), self.fs.Vfs.dlnc_cumul, ncom=2)[:2])
        to_delete1 = np.hstack(to_delete1)

        bn = self.fs.Wfs.mesh.bn
        to_delete2 = []
        for i in range(bn.shape[0]):
            to_delete2.append(_common.get_loce(np.array([bn[i]]), self.fs.Wfs.dlnc_cumul, ncom=6)[:6])
        to_delete2 = np.hstack(to_delete2)

        to_delete = np.concatenate([to_delete1, to_delete2 + self.fs.Vfs.dlnc.sum() * 2])
        self.to_delete = to_delete
        agl = delete_from_csr(self.agl, row_indices=to_delete, col_indices=to_delete )
        bgl = delete_from_csr(self.bgl, row_indices=to_delete, col_indices=to_delete )
        dim = agl.shape[0]
        self.dim=dim
        
        # initial arnoldi vector
        if v0 is None:
            v0 = np.ones(agl.shape[0])

        eigvals_tot = np.zeros((len(sigma), k))
        eigvecs_tot = np.zeros((len(sigma), self.agl.shape[0], k), dtype=complex) 

        for i in range(len(sigma)):     
        
            ###### DIAGONALIZATION ####### 
            eigvals, eigvecs = eigsh(
                A = agl,
                M = bgl,
                k=k,
                which=which,
                v0=v0,
                sigma = sigma[i],
                tol=tol,
                OPinv=None
            )
            
            ###### POSTPROCESSING #######
            ntot =  self.fs.Vfs.dlnc.sum() * 2 + self.fs.Wfs.dlnc.sum() * 6      
            a = np.full( ntot , 1 , dtype=bool )
            to_zero = to_delete
            a[to_zero] = False
            post_processed_eigvecs = np.zeros(( ntot , eigvecs.shape[1] ), dtype=complex)
            post_processed_eigvecs[a,:] = eigvecs
            eigvecs = post_processed_eigvecs
            
            ###### NORMALIZATION #######
            nn = 2*self.fs.Vfs.dlnc.sum()

            # the eigenvectors corresponding to degenerate subspace may not be orthogonal...
            # I now post-normalize them using the Hamiltonian
            Hs = eigvecs.conj().T @ self.agl @ eigvecs
            Bs = eigvecs.conj().T @ self.bgl @ eigvecs
            _, Vjk = eigh(Hs, Bs)
            eigvecs = eigvecs @ Vjk
            eigvals = np.diagonal(eigvecs.conj().T @ self.agl @ eigvecs).real
            # assert is all ok...
            
            # store these eigenvectors
            eigvecs_tot[i,:,:] = eigvecs
            eigvals_tot[i,:] = eigvals

        # want a single vector
        eigvals = np.hstack(eigvals_tot)
        eigvecs = np.hstack(eigvecs_tot)

        # sort eigenvalues and eigenvectors
        eigvals, eigvecs = sort_eig(eigvals, eigvecs)

        eigvecs_V = eigvecs[:nn,:]
        eigvecs_W = eigvecs[nn:,:]
        
        ###### SPINOR DISTRIBUTION ######
        # compute spinor distribution and normalize eigenvectors
        eigvecs_V, spinor_dist_V, norm_sum_V, norm_sum_region_V = get_spinor_dist(self.fs.Vfs, eigvecs_V, 2)
        eigvecs_W, spinor_dist_W, norm_sum_W, norm_sum_region_W = get_spinor_dist(self.fs.Wfs, eigvecs_W, 6)
        norm_sum = norm_sum_V + norm_sum_W
        norm_sum_region = norm_sum_region_V + norm_sum_region_W
        spinor_dist = np.concatenate((spinor_dist_V, spinor_dist_W),axis=0) / norm_sum
        # express in eV
        eigvals *= _constants.energy_scale
        eigvecs_V = eigvecs_V.reshape((self.fs.Vfs.dlnc.sum(),2,k*len(sigma))) / np.sqrt(norm_sum)
        eigvecs_W = eigvecs_W.reshape((self.fs.Wfs.dlnc.sum(),6,k*len(sigma))) / np.sqrt(norm_sum)
        return eigvals, eigvecs_V, eigvecs_W, spinor_dist, norm_sum_region
    
###############################################################################################################
################################################################################################################

######################## SYSTEM SOLVER ######################

class LinearSystem:
    def __init__(self):
        
        # hamiltonian global matrix
        self.Mgl = None
        
        # overlap matrix
        self.Fgl = None
        
        # components of the solution (setted when assembly)
        self.ncom = None
        
        # fem space (setted when assembly)
        self.fs = None
    
    def assembly(self, fs, v, dirichlet_borval=None, neumann_borval=None):
    
        ###### ASSEMBLY ########
        
        # store integers
    
        nn = fs.dlnc.sum() * v.ncom
        #nq = np.array(nn, dtype='int64')**2
        nq = np.sum((fs.tdof_per_elem * v.ncom)**2)
        ns = np.sum(fs.tdof_per_elem * v.ncom)
        row_mgl    = np.zeros(nq, dtype=int)
        col_mgl    = np.zeros(nq, dtype=int)
        data_mgl = np.zeros(nq, dtype=float)
        
        row_fgl = np.zeros(ns, dtype=int)
        col_fgl = np.zeros(ns, dtype=int)
        data_fgl = np.zeros(ns, dtype=float)

        col_cgl = np.zeros(ns, dtype=int)
        row_cgl = np.zeros(ns, dtype=int)
        data_cgl = np.zeros(ns, dtype=float)
        
        i=0 
        j=0
        k=0
        for fel in fs.felems:
    
            # get element matrices
            melm, felm, celm = v.get_elmat(fel)
            
                
            kloce = _common.get_loce(kconec=fel.nods, dlnc_cumul=fs.dlnc_cumul, ncom=1)
            nd = kloce.shape[0]
            ndq = nd**2 
            row_mgl[j:j+ndq] = np.repeat(kloce, nd)
            col_mgl[j:j+ndq] = np.tile(kloce,nd)
            data_mgl[j:j+ndq] = melm.flatten()
            #
            row_fgl[i:i+nd]=kloce
            data_fgl[i:i+nd] = felm.squeeze()
            #
            ## constrain vector
            col_cgl[i:i+nd]=kloce
            data_cgl[i:i+nd] = celm.squeeze()
#
            i+=nd
            j+=ndq

        Mgl = coo_matrix((data_mgl, (row_mgl, col_mgl)), shape=(nn,nn)).tocsr()
        Fgl = coo_matrix((data_fgl, (row_fgl, col_fgl)), shape=(nn,1)).tocsr()
        Cgl = coo_matrix((data_cgl, (row_cgl, col_cgl)), shape=(1,nn)).tocsr()

        # set fem space
        self.fs = fs
        self.ncom = v.ncom

        # impose boundary conditions
        if dirichlet_borval is not None:
            self.pure_neumann=False
            DirichletBC = SystemDirichletBoundaryConditions(fs, ncom=1, borval=dirichlet_borval)
            Mgl, Fgl = DirichletBC.impose_bc(Mgl, Fgl)
        else:
            self.pure_neumann=True
            '''Pure Neumann boundary conditions
            -----------
            This is a specific case. When only the normal derivative of the potential
            is specified on the border we have two issues.
            1) the potential is defined up to a constant.
            2) the compatibility requirement (Gauss theorem) have to be fulfilled

            Generally one uses lagrange multiplied to look for a zero-mean value potential.
            The lagrange multiplier also accounts the fulfillment of the gauss theorem.

            A row vector called c=(1 x nd) contains the matrix elements to include
            '''
            Mgl = vstack([hstack([Mgl, Cgl.T]),hstack([Cgl,0.])]).tocsr()
            Fgl = vstack([Fgl,0.]).tocsr()

        self.Mgl = Mgl
        self.Fgl = Fgl

    def solve(self):
        
        ###### DIAGONALIZATION #######
        x  = spsolve(A=self.Mgl, b=self.Fgl)
        
        return x    




################ BOUNDARY CONDITIONS ################
def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices.any():
        rows = list(row_indices)
    if col_indices.any():
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat


######################## POISSON BOUNDARY CONDITIONS ######################     
class BoundaryConditions:
    def __init__(self, fs, ncom):
        #self.mesh = mesh
        #self.ndof = 3
        self.ncom = ncom
        self.fs = fs

class SystemDirichletBoundaryConditions(BoundaryConditions):
    def __init__(self, fs, ncom, borval=None ):
        BoundaryConditions.__init__(self, fs, ncom)
        
        # dict with border function value
        self.borval = borval
        
        # global border node indices
        bn = self.fs.mesh.bn
        
        kcond = []
        for i in range(bn.shape[0]):
            kcond.append(_common.get_loce(np.array([bn[i]]), self.fs.dlnc_cumul, self.ncom)[:self.ncom])
        self.kcond = np.hstack(kcond)

        """
        example:
        borval={
            'ref' : 0.0,
            2 : 0.1
        }
        """
        if self.borval['ref'] is not None:
            reference = self.borval['ref']
            vcond = np.full(len(self.kcond), reference)
        else:
            vcond = np.full(len(self.kcond), np.nan)

        for key in self.borval.keys():
            potval = self.borval[key]
            if isinstance(key, int):
                edge_to_vertices = np.unique(self.fs.mesh.e_v[np.where(self.fs.mesh.e_l==key)[0]])
                idx = np.where(np.isin(bn,edge_to_vertices))[0]
                vcond[idx] = potval

                # there could be midside nodes
                idx = np.where(self.fs.mesh.v_l[bn]==key)[0]
                vcond[idx] = potval
        self.vcond = vcond
        mask = np.logical_not(np.isnan(vcond))
        self.vcond = vcond[mask]
        self.kcond = self.kcond[mask]
     
    
    #def _set_vcond(self):
    #    
    #    # default zero value on border
    #    vcond = np.zeros(len(self.kcond))
    #    borval = self.borval.copy()
    #    
    #    if borval is not None:
    #        l = None
    #        for key in borval.keys():
    #            potval = borval[key]
    #            if isinstance(potval,np.ndarray):
    #                l = key
    #        if l is not None:
    #            potval = borval[l]
    #            xpos = borval[l][:,0] / _constants.br
    #            potval = borval[l][:,1]
    #            spcint = CubicSpline(xpos, potval)
    #            borval.pop(l);
#
    #        # homogeneous dirichlet  
    #        Vg_fixed = np.vectorize(borval.get)(self.fs.mesh.v_l[self.fs.mesh.bn])
    #        vcond[:] = Vg_fixed
    #        
    #        # inhomogeneous dirichlet
    #        if l is not None:
    #            bottom_nodes_idx = np.where(self.fs.mesh.v_l[self.fs.mesh.bn]==l)[0]
    #            bottom_nodes = self.fs.mesh.bn[bottom_nodes_idx]
    #            # use spline cubic interpolation between points
    #            xint = np.sort(self.fs.mesh.vertices[bottom_nodes][:,0])
    #            potval_interpolated = spcint(xint)
    #            mask = np.argsort(self.fs.mesh.vertices[:,0][bottom_nodes])
    #            #print(xint*staticdata.br)
    #            #print(potval_interpolated)
    #            vcond[bottom_nodes_idx] = potval_interpolated[mask]
    #    assert np.isnan(vcond).sum()==0, 'Dirichlet boundary conditions missing'
    #    self.vcond = vcond
    
    
    def impose_bc(self, Mgl, Fgl):
        
        Mgl_dense = Mgl.toarray()
        Fgl_dense = Fgl.toarray()        
        for i in range(len(self.kcond)):
            Fgl_dense[:,0] = Fgl_dense[:,0] - Mgl_dense[:,self.kcond[i]]*self.vcond[i]
            Mgl_dense[self.kcond[i],:] = 0.0
            Mgl_dense[:,self.kcond[i]] = 0.0
            Mgl_dense[self.kcond[i],self.kcond[i]] = 1.0
            Fgl_dense[self.kcond[i],0] = self.vcond[i]
        
        Mgl = csr_matrix(Mgl_dense)
        Fgl = csr_matrix(Fgl_dense)
        del Mgl_dense, Fgl_dense
        return Mgl, Fgl   


class SystemNeumannBoundaryConditions(BoundaryConditions):
    def __init__(self, fs, ncom, borval=None):
        BoundaryConditions.__init__(self, fs, ncom)
        
        # dict with border normal derivative value
        # dict with border normal derivative function value
        self.borval = borval
        
        # global border node indices
        bn = self.fs.mesh.bn
        kcond = []
        for i in range(bn.shape[0]):
            kcond.append(_common.get_loce(np.array([bn[i]]), self.fs.dlnc_cumul, self.ncom)[:8])
        self.kcond = np.hstack(kcond)
        
        # set boundary value for each border node
        #self._set_vcond() 

    def impose_bc(self, Mgl, Fgl):
        return Mgl, Fgl
    

#########################################################################################

# common

def BlockMultiply(A, B, n):
    AB = np.block([
        [np.kron(A[:n,:n], B[:2,:2]), np.kron(A[:n,n:], B[:2,2:])],
        [np.kron(A[n:,:n], B[2:,:2]), np.kron(A[n:,n:] , B[2:,2:])]
    ])
    return AB
    
        
