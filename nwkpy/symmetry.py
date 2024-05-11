import numpy as np
from scipy.linalg import expm, eigh
from .operator_matrices import Jx, Jy, Jz, Sx, Sy, Sz, parity

########################################################################################
########################################################################################




# C3v symmetry group representation matrices
U = np.array([
    [1.j, -1.],
    [1., -1.j]
]) / np.sqrt(2.)

Uinv = np.array([
    [-1.j, 1.],
    [-1., 1.j]
]) / np.sqrt(2.)

eps = np.exp(1.j * np.pi * 2./3.)

irrep_C3v = {
    "E1/2" : {
        'l' : 2,
        'i' : Uinv @ np.array([
        [1,0],
        [0,1]
        ]) @ U,
    
        '3_001+' : Uinv @ np.array([
        [-eps, 0],
        [0, np.conjugate(-eps)]
        ]) @ U,
    
        '3_001-' : Uinv @ np.array([
        [np.conjugate(-eps), 0],
        [0, -eps]
        ]) @ U,
    
        'id' : Uinv @ np.array([
        [-1,0],
        [0,-1]
        ]) @ U,
    
        '3_001+d' : Uinv @ np.array([
        [eps, 0],
        [0, - np.conjugate(-eps)]
        ]) @ U,
    
        '3_001-d' : Uinv @ np.array([
        [-np.conjugate(-eps), 0],
        [0, eps]
        ]) @ U,
    
        'm1b10' : Uinv @ np.array([
        [0, -eps],
        [np.conjugate(eps),0]
        ]) @ U,
    
        'm120' : Uinv @ np.array([
        [0,-1.],
        [1.,0]
        ]) @ U,
    
        'm210' : Uinv @ np.array([
        [0,np.conjugate(-eps)],
        [eps,0]
        ]) @ U,
    
        'm1b10d' : Uinv @ np.array([
        [0,eps],
        [-np.conjugate(eps),0]
        ]) @ U,
    
        'm120d' : Uinv @ np.array([
        [0,1.],
        [-1.,0]
        ]) @ U,
    
        'm210d' : Uinv @ np.array([
        [0,-np.conjugate(-eps)],
        [-eps,0]
        ]) @ U,
    },
    "E3/2_1" : {
        'l' : 1,
        'i' : np.array([
        [1]
        ]),
    
        '3_001+' : np.array([
        [-1]
        ]),
    
        '3_001-' : np.array([
        [-1]
        ]),
    
        'm1b10' : np.array([
        [1.j]
        ]),
    
        'm120' : np.array([
        [1.j]
        ]),
    
        'm210' : np.array([
        [1.j]
        ]),
    
        'id' : np.array([
        [-1.]
        ]),
    
        '3_001+d' : np.array([
        [1.]
        ]),
    
        '3_001-d' : np.array([
        [1.]
        ]),
    
        'm1b10d' : np.array([
        [-1.j]
        ]),
    
        'm120d' : np.array([
        [-1.j]
        ]),
    
        'm210d' : np.array([
        [-1.j]
        ]),
    },
    "E3/2_2" : {
        'l' : 1,
        'i' : np.array([
        [1]
        ]),
    
        '3_001+' : np.array([
        [-1]
        ]),
    
        '3_001-' : np.array([
        [-1]
        ]),
    
        'm1b10' : np.array([
        [-1.j]
        ]),
    
        'm120' : np.array([
        [-1.j]
        ]),
    
        'm210' : np.array([
        [-1.j]
        ]),
    
        'id' : np.array([
        [-1.]
        ]),
    
        '3_001+d' : np.array([
        [1.]
        ]),
    
        '3_001-d' : np.array([
        [1.]
        ]),
    
        'm1b10d' : np.array([
        [1.j]
        ]),
    
        'm120d' : np.array([
        [1.j]
        ]),
    
        'm210d' : np.array([
        [1.j]
        ]),
    }
}

irrep_Cx = {
    "E1/2_1" : {
        'l' : 1,
        'i' : np.array([
            [1]
        ]),
        'm100' : np.array([
            [1.j]
        ]),
        'm1b10' : np.array([
            [1.j]
        ]),
        'm210' : np.array([
            [1.j]
        ]),
        'm120' : np.array([
            [1.j]
        ]),
        'id' : np.array([
            [-1.]
        ]),
        'm100d' : np.array([
            [-1.j]
        ]),
        'm1b10d' : np.array([
            [-1.j]
        ]),
        'm210d' : np.array([
            [-1.j]
        ]),
        'm120d' : np.array([
            [-1.j]
        ])
    },
    "E1/2_2" : {
        'l' : 1,
        'i' : np.array([
            [1]
        ]),
        'm100' : np.array([
            [-1.j]
        ]),
        'm1b10' : np.array([
            [-1.j]
        ]),
        'm210' : np.array([
            [-1.j]
        ]),
        'm120' : np.array([
            [-1.j]
        ]),
        'id' : np.array([
            [-1.]
        ]),
        'm100d' : np.array([
            [1.j]
        ]),
        'm1b10d' : np.array([
            [1.j]
        ]),
        'm210d' : np.array([
            [1.j]
        ]),
        'm120d' : np.array([
            [1.j]
        ])
    }
}

########################################################################################
########################################################################################

# symmetry operation class
class SymmetryOp:
    def __init__(self, mat=np.eye(2), op=None, name=''):
        self.dim = len(op)
        self.name = name
        self.mat = mat
        if op is not None:
            self.op = op
        else:
            self.op = np.eye(self.dim)
        
    #def real_space_repres(self, mesh):
    #    # get the real space representation of this symmetry operator
    #    # if the mesh is invariant under the symmetry operation,
    #    # the real space representation of this operator is a permutation matrix
    #    # operator acting as a reordering of the nodes
    #    self.permutation_matrix = None


def get_symmetry_operator(comps=np.s_[0:8], mesh= None, name = 'i', Jx=Jx, Jy=Jy, Jz=Jz, parity=parity):
    dim = comps.stop - comps.start
    # for each name we get a different rotation operator
    d2pi = 0.0

    if name[-1] == 'd':
        d2pi = 2. * np.pi
        name = name[:-1]
    
    Jz = Jz[comps,comps]
    Jx = Jx[comps,comps]
    Jy = Jy[comps,comps]

    parity = parity[comps,comps]

    if name == 'i':
        # identity operator 
        phi = 0.0 + d2pi
        mat = np.array([
            [1, 0],
            [0, 1]
        ])
        op = expm( - 1.j * Jz * phi)

        return SymmetryOp(mat=mat, op=op, name=name)
        
    if name == '3_001+':
        # rotation of 120 degrees around the 001 axis (positive)
        phi = 2.0 * np.pi / 3. + d2pi
        mat = np.array([
            [0, -1],
            [1, -1]
        ])
        op = expm( - 1.j * Jz * phi)

        return SymmetryOp(mat=mat, op=op, name=name)
    
    if name == '3_001-':
        # rotation of 120 degrees around the 001 axis (negative)
        phi = 2.0 * np.pi / 3. + d2pi
        mat = np.array([
            [-1, 1],
            [-1, 0]
        ])
        op = expm(  1.j * Jz * phi)
        return SymmetryOp(mat=mat, op=op, name=name)
    
    if name == '2_001':
        # rotation of 120 degrees around the 001 axis 
        theta = np.pi + d2pi
        mat = np.array([
            [-1, 0],
            [0, -1]
        ])
        op = expm(- 1.j * Jz * theta)
        return SymmetryOp(mat=mat, op=op, name=name)
    
    if name == '6_001-':
        # rotation of 120 degrees around the 001 axis 
        theta = - 2.0 * np.pi / 6. + d2pi
        mat = np.array([
            [0, 1],
            [-1, 1]
        ])
        op = expm(- 1.j * Jz * theta)
        return SymmetryOp(mat=mat, op=op, name=name)
    
    if name == '6_001+':
        # rotation of 120 degrees around the 001 axis 
        theta = 2.0 * np.pi / 6. + d2pi
        mat = np.array([
            [1, -1],
            [1, 0]
        ])
        op = expm(- 1.j * Jz * theta)
        return SymmetryOp(mat=mat, op=op, name=name)
    
    ##### reflections ######
    if name == 'm1b10':
        #  reflection on plane sigma_v3 
        phi = np.pi + d2pi
        #theta = 11.*np.pi/6.
        gamma = 2.*np.pi/3.
        beta = np.pi 
        alpha = 0.0
        #theta = np.pi/3.
        mat = np.array([
            [0, 1],
            [1, 0]
        ])

        #mat = np.array([
        #    [1, 0],
        #    [-1, 1]
        #])

        op = parity @ expm(- 1.j * (phi) * (Jx * (np.sqrt(3)/2) + Jy * (-1/2)))
        return SymmetryOp(mat=mat, op=op, name=name)
        
    if name == 'm120':
        # reflection on plane sigma_v1
        phi = np.pi + d2pi
        gamma = 0.0  
        beta = np.pi
        alpha = 0.0
        mat = np.array([
            [1, -1],
            [0, -1]
        ])

        #mat = np.array([
        #    [-1, 1],
        #    [0, 1]
        #])

        #op = np.conjugate(parity @ expm(- 1.j * alpha * Jz) @ expm(- 1.j * beta * Jy) @ expm(- 1.j * gamma * Jz))
        op = parity @ expm(- 1.j * (phi) * Jy )
        return SymmetryOp(mat=mat, op=op, name=name)
    
    if name == 'm210':
        # reflection on plane on plane sigma_v2
        phi = np.pi + d2pi
        #theta = 7.* np.pi/6.
        gamma = -2.*np.pi/3.
        beta = -np.pi
        alpha = 0.0 
        #theta = 2. * np.pi/3.
        mat = np.array([
            [-1, 0],
            [-1, 1]
        ])

        #mat = np.array([
        #    [1, 0],
        #    [1, -1]
        #])
        op = parity @ expm(- 1.j * (phi) * (Jx * (-np.sqrt(3)/2) + Jy * (-1/2)))
        #op = np.conjugate(parity @ expm(- 1.j * alpha * Jz) @ expm(- 1.j * beta * Jy) @ expm(- 1.j * gamma * Jz))
        return SymmetryOp(mat=mat, op=op, name=name)
    
    if name == 'm100':
        # reflection on plane orthogonal to x-axis
        phi = np.pi + d2pi
        mat = np.array([
            [-1, 1],
            [0, 1]
        ]) 
        op = parity @ expm(- 1.j * (phi) * Jx )
        return SymmetryOp(mat=mat, op=op, name=name)
    
########################################################################################
########################################################################################

class SymmetryGroup():
    def __init__(self, comps=np.s_[0:8], symops_names=['i'], irrep = None):
        symops = []
        for name in symops_names:
            symops.append(get_symmetry_operator(comps=comps, name = name))
            #symops.append(reflection(comps=np.s_[2:8], name = name))
        self.symops = dict(zip(symops_names,symops))

        self.h = len(symops_names)

        self.irrep = irrep

        ltot=0
        for key in irrep.keys():
            ltot += irrep[key]['l']
        self.ltot=ltot

        irrep_names=[]
        for key in irrep.keys():
            for i in range(irrep[key]['l']):
                lab=''
                if irrep[key]['l']>1:
                    lab = '_' + str(i+1)
                irrep_names.append(key+lab)
        self.irrep_names = irrep_names
    
########################################################################################
########################################################################################

class ClassOperator:
    def __init__(self, comps = np.s_[0:8], symops_names=['i'], alpha=1.0 ):
        symops = []
        for name in symops_names:
            symop = get_symmetry_operator(comps=comps, name = name)
            symop.op *= alpha
            symops.append(symop)
        self.symops = dict(zip(symops_names,symops))   
        self.alpha = alpha


########################################################################################
########################################################################################
#def PTCO(*args, energy_spectrum=None):
#
#    # here we compute the quantity: alphaH * H|psi>. Remember that H|psi> = E|psi>.
#    nk = energy_spectrum.shape[0]
#    neig = energy_spectrum.shape[1]
#
#    V = np.zeros((nk, neig, neig), dtype=complex)
#
#    alphaH_Hpsi_kz = np.zeros((nk,neig))
#    if energy_spectrum is not None:
#        E0 = np.max(energy_spectrum,axis=1)
#        alphaH_Hpsi_kz = (energy_spectrum - E0[:,np.newaxis])/(np.min(energy_spectrum,axis=1)-E0)[:,np.newaxis]
#
#    O = 0.0
#
#    # sum the contribution for each wave function object
#    for (wf, class_operators) in args:
#        O += wf.get_class_operator(class_operators, kz0 = True)
#
#    # find the coefficients at each kz
#    for i in range(nk):
#        # add the hamiltonian action on the eigestates
#        Ok = O[i,:,:] + np.diag(v=alphaH_Hpsi_kz[i,:])
#        _, Vjk = eigh(Ok)
#        mask = np.argsort(np.abs(Vjk)**2, axis=0)[-2::,:].T
#        Vjk = Vjk[:,np.argsort(mask[:,0])]
#        V[i,:,:] = Vjk
#    return V



def get_class_operator_matrix(wf, B,  class_operators):
    
    nk = wf.psi.shape[0]
    neig = wf.psi.shape[-1]
    ncom = wf.psi.shape[2]
    O = np.zeros((nk,neig,neig), dtype = complex)
    # sum the contribution for each wave function object
    psip=0.0
    for cop in class_operators:
        for symop_name in cop.symops.keys():
            symop = cop.symops[symop_name]
            # symmetry operation on the wave function spinor
            psip += wf.apply_symmetry_operator(symop)

    U = wf.psi.reshape(nk, wf.fs.dlnc.sum()*ncom,-1)
    Utd = psip.reshape(nk, wf.fs.dlnc.sum()*ncom,-1)
    for i in range(nk):
        O[i,:,:] = U[i].conj().T @ B @ Utd[i]
    
    return O

def PTCO(*args, energy_spectrum=None):
    O = 0.0
    for (wf, B, class_operators) in args:
        O += get_class_operator_matrix(wf, B,  class_operators)
    nk = O.shape[0]
    neig = O.shape[1]
    V = np.zeros((nk, neig, neig), dtype=complex)
    alphaH_Hpsi_kz = np.zeros((nk,neig))
    if energy_spectrum is not None:
        E0 = np.max(energy_spectrum,axis=1)
        alphaH_Hpsi_kz = (energy_spectrum - E0[:,np.newaxis])/(np.min(energy_spectrum,axis=1)-E0)[:,np.newaxis]
    for i in range(nk):
        # add the hamiltonian action on the eigestates
        Ok = O[i,:,:] + np.diag(v=alphaH_Hpsi_kz[i,:])
        _, Vjk = eigh(Ok)
        # find the correct eigenvectors order by evaluating the eigenvalues at each kz
        mask = np.argsort(np.abs(Vjk)**2, axis=0)[-2::,:].T
        Vjk = Vjk[:,np.argsort(mask[:,0])]
        V[i,:,:] = Vjk
    return V

def project_irrep(wf, B, symg):
    # need to know the dimensions of the weights matrix

    nk = wf.psi.shape[0]
    neig = wf.psi.shape[-1]
    ncom = wf.psi.shape[2]
    
    # the result is stored in a dictionary: irrep_name --> projection coefficients
    projections = {irrep_name:0 for irrep_name in symg.irrep_names}

    # number of operations in the group
    h = symg.h
    
    i=0
    for irrep_label in symg.irrep.keys():
        l = symg.irrep[irrep_label]['l']
        for partner_fun_label in range(symg.irrep[irrep_label]['l']):
            irrep_name = symg.irrep_names[i]
            mu = partner_fun_label

            # projection on the irrep
            psip = 0.0
            for symop_name in symg.symops.keys():
                symop = symg.symops[symop_name]
                psip += wf.apply_symmetry_operator(symop=symop) * np.conjugate(symg.irrep[irrep_label][symop_name][mu,mu]) * (l/h)

            U = wf.psi.reshape(nk, wf.fs.dlnc.sum()*ncom,-1)
            Utd = psip.reshape(nk, wf.fs.dlnc.sum()*ncom,-1)
            proj= np.zeros((nk,neig))
            for ik in range(nk):
                proj[ik,:] = np.diagonal(U[ik].conj().T @ B @ Utd[ik]).real
            projections[irrep_name] = proj
            i+=1
    
    return projections

########################################################################################
########################################################################################


# the following function is used to classify the energy bands according to a given "classifier matrix"
# the classifier can be anything in principle. Usually, the classifyer is the projection of the wave function on
# one of the irreducible representations of the symmetry group of the Hamiltonian.

# the dimension of the returned vector depends on the value of the classifier at kz=0.
# I should consider to modify this, because usually at kz=0 the symmetry group can be different (elevated symmetry at zone center)

def classify_bands(bands, classifier):
    # from kz=0 i select the energy bands that i want to "follow"
    # number of E12 bands
    #num_classified = (mask>0).sum()
    
    # the new bands vector
    bands_classified = np.zeros((bands.shape))
    
    # crossing matrix, considers all bands
    crossing_matrix = np.pad(np.diff(classifier, axis=0), pad_width=(1,0))[:,1:]
    
    # loop over the indices of the bands
    for m in range(bands.shape[1]):
        # the index normally is the one of the band
        idx = m
        # loop on kz. start from one
        if classifier[0,m] == 1:
            for nk in range(bands.shape[0]):
                crossing_label = crossing_matrix[nk,m]
                if crossing_label == -1:
                    # localize the position of the 1
                    if np.argwhere(crossing_matrix[nk,:]>0)[0][0] < m:
                        # the correct band index is the one of the lower band
                        idx += -1
                    else:
                        # the correct band index is the one of the lower band
                        idx += 1
                bands_classified[nk,m] = bands[nk,idx]
    # return only the classified bands
    mask = classifier[0,:] == 1
    bands_classified = bands_classified[:,mask]
    return bands_classified
