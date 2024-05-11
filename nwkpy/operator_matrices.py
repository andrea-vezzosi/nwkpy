import numpy as np

########################################################################################
########################################################################################

# angular momentum matrices written in the winkler basis

# this matrix has to be transposed probably, it is used to transform the
# angular momentum matrices from the basis of winkler (standard textbook) to the one I used in my code
#Qmat = np.diag([1.,-1.j, -1,-1.j,1.j,1,-1,-1.j])

Qmat = np.diag([1.,1.j, -1,1.j,-1.j,1,-1,1.j])

Jz = np.diag(v = np.array([0.5, -0.5, 1.5, -1.5, 0.5, -0.5, 0.5, -0.5]))

Jx = np.array([
    [0,1, 0,0,0,0,0,0],
    [1,0, 0,0,0,0,0,0],
    [0,0, 0,0,np.sqrt(3.),0,0,0],
    [0,0, 0,0,0,np.sqrt(3.),0,0],
    [0,0, np.sqrt(3.),0,0,2.,0,0],
    [0,0, 0,np.sqrt(3.),2.,0,0,0],
    [0,0, 0,0,0,0,0,1],
    [0,0, 0,0,0,0,1,0]
]) * 0.5

Jy = np.array([
    [0,-1, 0,0,0,0,0,0],
    [1,0, 0,0,0,0,0,0],
    [0,0, 0,0,-np.sqrt(3.),0,0,0],
    [0,0, 0,0,0,np.sqrt(3.),0,0],
    [0,0, np.sqrt(3.),0,0,-2.,0,0],
    [0,0, 0,-np.sqrt(3.),2.,0,0,0],
    [0,0, 0,0,0,0,0,-1],
    [0,0, 0,0,0,0,1,0]
]) * 0.5j

Ux = np.array([
    [-np.sqrt(3.), 0.0],
    [0, np.sqrt(3.)],
    [0, -1.],
    [1.,0],
]) * 1./(3*np.sqrt(2))


Uy = np.array([
    [np.sqrt(3.), 0.0],
    [0, np.sqrt(3.)],
    [0, 1.],
    [1.,0],
]) * 1.j/(3*np.sqrt(2))


Uz = np.array([
    [0., 0.],
    [0., 0.],
    [1, 0.],
    [0.,1.],
]) * np.sqrt(2.)/3.


sigma_x = np.array([
    [0.,1.],
    [1.,0.]
])

sigma_y = np.array([
    [0.,-1.],
    [1.,0.]
]) * 1.j

sigma_z = np.array([
    [1.,0.],
    [0.,-1.]
])

# spin matrices
Sblock=np.block([
    [2./3. * Jx[2:6,2:6], -2. * Ux],
    [-2. * Ux.conj().T, -1./3. * sigma_x ] 
])
Sx = np.block([
    [sigma_x, np.zeros((2, 6),dtype=complex)],
    [np.zeros((6, 2),dtype=complex), Sblock ]
])

Sblock=np.block([
    [2./3. * Jy[2:6,2:6], -2. * Uy],
    [-2. * Uy.conj().T, -1./3. * sigma_y ] 
])
Sy = np.block([
    [sigma_y, np.zeros((2, 6),dtype=complex)],
    [np.zeros((6, 2),dtype=complex), Sblock ]
])

Sblock=np.block([
    [2./3. * Jz[2:6,2:6], -2. * Uz],
    [-2. * Uz.conj().T, -1./3. * sigma_z ] 
])
Sz = np.block([
    [sigma_z, np.zeros((2, 6),dtype=complex)],
    [np.zeros((6, 2),dtype=complex), Sblock ]
])

parity = np.diag(v = np.array([1., 1., -1, -1, -1, -1, -1 , -1]))

# transform into my basis (just some phase factors)
Jx = np.conjugate(Qmat) @ Jx @ Qmat.T
Jy = np.conjugate(Qmat) @ Jy @ Qmat.T
Jz = np.conjugate(Qmat) @ Jz @ Qmat.T

Sx = np.conjugate(Qmat) @ Sx @ Qmat.T
Sy = np.conjugate(Qmat) @ Sy @ Qmat.T
Sz = np.conjugate(Qmat) @ Sz @ Qmat.T

#parity = np.conjugate(Qmat) @ parity @ Qmat.T
