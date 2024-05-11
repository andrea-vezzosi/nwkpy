import numpy as np
from nwkpy.fem.gauss import GaussParams

# Function
def Q_1(csi,eta,sigma):
    return csi**2*(3.-2.*csi)+2.*csi*eta*sigma
def Q_2(csi,eta,sigma):
    return eta**2*(3.-2.*eta)+2.*csi*eta*sigma
def Q_3(csi,eta,sigma):
    return sigma**2*(3.-2.*sigma)+2.*csi*eta*sigma

# x deriv
def R_1(csi,eta,sigma):
    return csi**2*(csi-1.)-csi*eta*sigma
def R_2(csi,eta,sigma):
    return csi*eta**2 + 0.5*csi*eta*sigma
def R_3(csi,eta,sigma):
    return csi*sigma**2 + 0.5*csi*eta*sigma

# y deriv
def S_1(csi,eta,sigma):
    return csi**2*eta + 0.5*csi*eta*sigma
def S_2(csi,eta,sigma):
    return eta**2*(eta-1.)-csi*eta*sigma
def S_3(csi,eta,sigma):
    return eta*sigma**2 + 0.5*csi*eta*sigma


###### DERIVATIVES ######

# Function
def Q_1_p_csi(csi,eta,sigma):
    return 2.*(3.-2.* csi)*csi - 2.* csi**2 - 2.* csi* eta + 2.* eta* sigma
def Q_1_p_eta(csi,eta,sigma):
    return -2. *csi *eta + 2.* csi *sigma
def Q_2_p_csi(csi,eta,sigma):
    return -2.* csi *eta + 2.* eta* sigma
def Q_2_p_eta(csi,eta,sigma):
    return -2. *csi* eta + 2.* (3. - 2.* eta)* eta - 2.* eta**2 + 2.* csi* sigma
def Q_3_p_csi(csi,eta,sigma):
    return -2.* csi *eta + 2.* eta* sigma - 2. *(3. - 2.* sigma)* sigma + 2.* sigma**2
def Q_3_p_eta(csi,eta,sigma):
    return -2.* csi* eta + 2.* csi* sigma - 2.* (3. - 2.* sigma)* sigma + 2.* sigma**2

# x deriv
def R_1_p_csi(csi,eta,sigma):
    return 2.* (-1. + csi)* csi + csi**2 + csi* eta - eta* sigma
def R_1_p_eta(csi,eta,sigma):
    return csi* eta - csi *sigma
def R_2_p_csi(csi,eta,sigma):
    return -0.5 *csi* eta + eta**2 + 0.5* eta* sigma
def R_2_p_eta(csi,eta,sigma):
    return 1.5 *csi* eta + 0.5* csi* sigma
def R_3_p_csi(csi,eta,sigma):
    return -0.5* csi* eta - 2.* csi* sigma + 0.5 *eta* sigma + sigma**2
def R_3_p_eta(csi,eta,sigma):
    return -0.5* csi* eta - 1.5* csi* sigma

# y deriv
def S_1_p_csi(csi,eta,sigma):
    return 1.5 *csi* eta + 0.5* eta* sigma
def S_1_p_eta(csi,eta,sigma):
    return csi**2 - 0.5* csi* eta + 0.5* csi* sigma
def S_2_p_csi(csi,eta,sigma):
    return csi* eta - eta* sigma
def S_2_p_eta(csi,eta,sigma):
    return csi* eta + 2.* (-1. + eta)* eta + eta**2 - csi* sigma
def S_3_p_csi(csi,eta,sigma):
    return -0.5* csi* eta - 1.5* eta* sigma
def S_3_p_eta(csi,eta,sigma):
    return -0.5* csi* eta + 0.5* csi* sigma - 2.* eta* sigma + sigma**2

################################################################################

# Lagrange Polynomials
def L_1(csi,eta,sigma):
    return csi
def L_2(csi,eta,sigma):
    return eta
def L_3(csi,eta,sigma):
    return sigma

# x deriv
def L_1_p_csi(csi,eta,sigma):
    return 1.
def L_2_p_csi(csi,eta,sigma):
    return 0.
def L_3_p_csi(csi,eta,sigma):
    return -1.

# y deriv
def L_1_p_eta(csi,eta,sigma):
    return 0.
def L_2_p_eta(csi,eta,sigma):
    return 1.
def L_3_p_eta(csi,eta,sigma):
    return -1.


##################################################################################
# Quadratic Lagrange
#
# number of nodes = 6 (3 geometric + 3 interpolation)
# number of dof = 6 
# 
# 2
# | \
# 4  6
# |   \
# 3--5--1 
# 
# Quadratic Lagrange Polynomials
def LQ_1(csi,eta):
    return -csi + 2*csi**2
def LQ_2(csi,eta):
    return -eta + 2*eta**2
def LQ_3(csi,eta):
    return 1 - 3*csi - 3*eta + 2*csi**2 + 4*csi*eta + 2*eta**2
def LQ_4(csi,eta):
    return 4*eta - 4*csi*eta - 4*eta**2
def LQ_5(csi,eta):
    return 4*csi - 4*csi*eta - 4*csi**2
def LQ_6(csi,eta):
    return 4*csi*eta

# x deriv
def LQ_1_p_csi(csi,eta):
    return -1 + 4*csi
def LQ_2_p_csi(csi,eta):
    return 0
def LQ_3_p_csi(csi,eta):
    return - 3 + 4*csi + 4*eta
def LQ_4_p_csi(csi,eta):
    return -4*eta
def LQ_5_p_csi(csi,eta):
    return 4 - 4*eta - 8*csi
def LQ_6_p_csi(csi,eta):
    return 4*eta

# y deriv
def LQ_1_p_eta(csi,eta):
    return 0
def LQ_2_p_eta(csi,eta):
    return -1 + 4*eta
def LQ_3_p_eta(csi,eta):
    return - 3 + 4*csi + 4*eta
def LQ_4_p_eta(csi,eta):
    return 4 - 4*csi - 8*eta
def LQ_5_p_eta(csi,eta):
    return -4*csi
def LQ_6_p_eta(csi,eta):
    return 4*csi

##################################################################################
# Lagrange-hermite LH6
#
# number of nodes = 4 (3 geometric + 1 interpolation)
# number of dof = 6 
# node with derivatives is always in position (0,0) of the reference triangle
# midside node 4 is always in position (0.5,0.5)
# 2
# | \
# |  4
# |   \
# 3'----1 
# 

# shape functions
def LH6_1(csi,eta):
    return -1. * csi * eta + 1. * csi**2
def LH6_2(csi,eta):
    return -1. * csi * eta + 1. * eta**2
def LH6_3(csi,eta):
    return 1. -2. * csi * eta - 1. * csi**2 - 1. * eta**2
def LH6_3_x(csi,eta):
    return 1.0 * csi - 1. * csi * eta - 1. * csi**2
def LH6_3_y(csi,eta):
    return 1.0 * eta - 1. * csi * eta - 1. * eta**2
def LH6_4(csi,eta):
    return 4. * csi * eta

# derivative with respect to csi
def LH6_1_p_csi(csi,eta):
    return -1. * eta + 2. * csi
def LH6_2_p_csi(csi,eta):
    return -1. * eta
def LH6_3_p_csi(csi,eta):
    return -2. * eta - 2. * csi
def LH6_3_x_p_csi(csi,eta):
    return 1.0  - 1. * eta - 2. * csi
def LH6_3_y_p_csi(csi,eta):
    return - 1. * eta
def LH6_4_p_csi(csi,eta):
    return 4. * eta

# derivative with respect to eta
def LH6_1_p_eta(csi,eta):
    return -1. * csi 
def LH6_2_p_eta(csi,eta):
    return -1. * csi + 2. * eta
def LH6_3_p_eta(csi,eta):
    return -2. * csi  - 2. * eta
def LH6_3_x_p_eta(csi,eta):
    return - 1. * csi 
def LH6_3_y_p_eta(csi,eta):
    return 1.0 - 1. * csi - 2. * eta
def LH6_4_p_eta(csi,eta):
    return 4. * csi

##################################################################################
# Lagrange-hermite LH7
#
# number of nodes = 3 (3 geometric and interp)
# number of dof = 7
# node with derivatives is always in position (1,0) and (0,1) of the reference triangle
# node 3 at (0,0) has only 1 dof, the function value
# 2'
# | \
# |  \
# |   \
# 3----1' 
# 

# shape functions
def LH7_1(csi,eta):
    return   2.*csi -1.*csi**2 -2. * csi * eta**2
def LH7_1_x(csi,eta):
    return -1.*csi +  1.*csi**2 + 1. * csi * eta**2
def LH7_1_y(csi,eta):
    return 1. * csi**2 * eta
def LH7_2(csi,eta):
    return 2. * eta - 1. * eta**2 -2. * csi**2 * eta
def LH7_2_x(csi,eta):
    return 1. * csi * eta**2
def LH7_2_y(csi,eta):
    return -1. * eta + 1.*eta**2  + 1. * csi**2 * eta
def LH7_3(csi,eta):
    return 1. -2. * csi -2.*eta + 1.*csi**2  + 1.*eta**2 + 2.*csi**2 * eta + 2.*csi * eta**2

# derivative with respect to csi
def LH7_1_p_csi(csi,eta):
    return   2. - 2. * csi -2. * eta**2
def LH7_1_x_p_csi(csi,eta):
    return -1. +  2.*csi + 1. * eta**2
def LH7_1_y_p_csi(csi,eta):
    return 2. * csi * eta
def LH7_2_p_csi(csi,eta):
    return  -4. * csi * eta
def LH7_2_x_p_csi(csi,eta):
    return 1.* eta**2
def LH7_2_y_p_csi(csi,eta):
    return  2. * csi * eta
def LH7_3_p_csi(csi,eta):
    return -2. + 2.* csi + 4. * csi * eta + 2. * eta**2

# derivative with respect to eta
def LH7_1_p_eta(csi,eta):
    return  -4. * csi * eta
def LH7_1_x_p_eta(csi,eta):
    return  2. * csi * eta
def LH7_1_y_p_eta(csi,eta):
    return 1. * csi**2
def LH7_2_p_eta(csi,eta):
    return 2. - 2. * eta -2. * csi**2
def LH7_2_x_p_eta(csi,eta):
    return 2. * csi * eta
def LH7_2_y_p_eta(csi,eta):
    return -1. + 2.*eta  + 1. * csi**2
def LH7_3_p_eta(csi,eta):
    return -2.   + 2.*eta + 2.*csi**2 + 4.*csi * eta

class _ShapeFunction:
    def __init__(self, ngauss):
        self.gpar = GaussParams(ngauss)
        
        # matrix of gauss weights (nodelm*ndof x ngauss)
        self.WT = None
        
        # fun and der shape (nodelm*ndof x ngauss)
        self.N = None
        self.N_p_csi = None
        self.N_p_eta = None
        
        # shape integrals (nodelm*ndof x nodelm*ndof)
        self.B1_B1= None
        self.B2_B1= None
        self.B1_B2= None
        self.B2_B2= None
        self.N_B1 = None
        self.B1_N = None
        self.N_B2 = None
        self.B2_N = None
        self.N_N  = None
    
    def _get_shape(self):
        
        # compute shape functions at gauss pts
        
        N = np.zeros((self.ndof, self.gpar.n))
        N_p_csi = np.zeros((self.ndof, self.gpar.n))
        N_p_eta = np.zeros((self.ndof, self.gpar.n))
        for ig in range(self.gpar.n):
            csi = self.gpar.pt_x[ig]
            eta = self.gpar.pt_y[ig]
            
            # build shape function (9x12)
            N[:,ig]       = self.fun(csi, eta).squeeze()
            N_p_csi[:,ig] = self.der_csi(csi,eta).squeeze()
            N_p_eta[:,ig] = self.der_eta(csi,eta).squeeze()
        self.N = N
        self.N_p_csi = N_p_csi
        self.N_p_eta = N_p_eta
        
    def _integrate(self):
        
        # integrate bare shape 
        
        WT = self.gpar.get_weight_matrix(self.ndof)
        self.B1_B1 = np.dot(WT*self.N_p_csi,self.N_p_csi.T)
        self.B2_B1 = np.dot(WT*self.N_p_eta,self.N_p_csi.T)
        self.B1_B2 = np.dot(WT*self.N_p_csi,self.N_p_eta.T)
        self.B2_B2 = np.dot(WT*self.N_p_eta,self.N_p_eta.T)

        self.N_B1 = np.dot(WT*self.N,self.N_p_csi.T)
        self.B1_N = np.dot(WT*self.N_p_csi,self.N.T)
        self.N_B2 = np.dot(WT*self.N,self.N_p_eta.T)
        self.B2_N = np.dot(WT*self.N_p_eta,self.N.T)

        self.N_N = np.dot(WT*self.N,self.N.T)
        
        self.WT = WT
        
    # methods to evaluate derivatives, already integrated
     
    def ddx(self, J, detJ, T):
        ddx = J[1,1]**2 * self.B1_B1 \
        - J[1,1] * J[1,0] * (self.B2_B1 + self.B1_B2) \
        + J[1,0]**2 * self.B2_B2
        ddx = np.dot(np.dot(T.T,ddx),T)/detJ
        return ddx
    
    def ddy(self, J, detJ, T):
        ddy = J[0,1]**2 * self.B1_B1 \
        - J[0,1]*J[0,0]*(self.B2_B1 + self.B1_B2) \
        + J[0,0]**2 * self.B2_B2
        ddy = np.dot(np.dot(T.T,ddy),T)/detJ
        return ddy
    
    def dxl(self, J, detJ, T):
        dxl = J[1,1] * self.B1_N \
        - J[1,0] * self.B2_N 
        dxl = np.dot(np.dot(T.T,dxl),T)
        return dxl
    
    def dxr(self, J, detJ, T):
        dxr = J[1,1] * self.N_B1 \
        - J[1,0] * self.N_B2
        dxr = np.dot(np.dot(T.T,dxr),T)
        return dxr
    
    def dyl(self, J, detJ, T):
        dyl = - J[0,1] * self.B1_N \
        + J[0,0] * self.B2_N 
        dyl = np.dot(np.dot(T.T,dyl),T)
        return dyl
    
    def dyr(self, J, detJ, T):
        dyr = - J[0,1] * self.N_B1 \
        + J[0,0] * self.N_B2 
        dyr = np.dot(np.dot(T.T,dyr),T)
        return dyr
    
    def dxdy(self, J, detJ, T):
        
        dxdy = - J[1,1] * J[0,1] * self.B1_B1 \
        + J[1,1] * J[0,0] * self.B1_B2 \
        + J[1,0] * J[0,1] * self.B2_B1 \
        - J[1,0] * J[0,0] * self.B2_B2
                            
        dxdy = np.dot(np.dot(T.T,dxdy),T)/detJ
        return dxdy
    
    def dydx(self, J, detJ, T):
        
        dydx = - J[1,1] * J[0,1] * self.B1_B1 \
        + J[0,1] * J[1,0] * self.B1_B2 \
        + J[0,0] * J[1,1] * self.B2_B1 \
        - J[1,0] * J[0,0] * self.B2_B2
                            
        dydx = np.dot(np.dot(T.T,dydx),T)/detJ
        return dydx
    
    def I(self, J, detJ, T):
        return np.dot(np.dot(T.T,self.N_N),T)*detJ


    # this method evaluates the Lz operator integrated on this element

    def Lz(self, gauss_coords, J, detJ, T):
        # transform f to (nodelm*ndof x ngauss)
        nd = self.WT.shape[0]
        ng = self.WT.shape[1]

        xg = gauss_coords[:,0]
        xg = np.tile( xg , nd ).reshape( ( nd , ng) )

        yg = gauss_coords[:,1]
        yg = np.tile( yg , nd ).reshape( ( nd , ng) )

        N_x_B1 = np.dot(self.WT * xg * self.N, self.N_p_csi.T)
        N_x_B2 = np.dot(self.WT * xg * self.N, self.N_p_eta.T)
        N_y_B1 = np.dot(self.WT * yg * self.N, self.N_p_csi.T)
        N_y_B2 = np.dot(self.WT * yg * self.N, self.N_p_eta.T)

        Lz = -1.j * ( ( - J[0,1] * N_x_B1 + J[0,0] * N_x_B2 ) - ( J[1,1] * N_y_B1 - J[1,0] * N_y_B2) )

        Lz = np.dot(np.dot(T.T,Lz),T)

        return Lz

class ShapeFunctionHermite(_ShapeFunction):
    def __init__(self, ngauss=12):
        self.nodelm = 3
        self.ndof_per_node = np.array([3,3,3])
        self.ndof = self.ndof_per_node.sum()
        _ShapeFunction.__init__(self, ngauss)
        self._get_shape()
        self._integrate()
        
    def fun(self, csi, eta):

        # Evaluate shape function

        ndim = self.ndof
        N = np.zeros((ndim,1))
        sigma = 1.0-csi-eta

        N[0]= Q_1(csi,eta,sigma)
        N[1]= R_1(csi,eta,sigma)
        N[2]= S_1(csi,eta,sigma)
        N[3]= Q_2(csi,eta,sigma)
        N[4]= R_2(csi,eta,sigma)
        N[5]= S_2(csi,eta,sigma)
        N[6]= Q_3(csi,eta,sigma)
        N[7]= R_3(csi,eta,sigma)
        N[8]= S_3(csi,eta,sigma)
        return N

    def der_csi(self, csi, eta):

        # Evaluate shape derivative
        # with respect to csi

        ndim = self.ndof
        N_p_csi = np.zeros((ndim,1))
        sigma = 1.0-csi-eta
        N_p_csi[0]= Q_1_p_csi(csi,eta,sigma)
        N_p_csi[1]= R_1_p_csi(csi,eta,sigma)
        N_p_csi[2]= S_1_p_csi(csi,eta,sigma)
        N_p_csi[3]= Q_2_p_csi(csi,eta,sigma)
        N_p_csi[4]= R_2_p_csi(csi,eta,sigma)
        N_p_csi[5]= S_2_p_csi(csi,eta,sigma)
        N_p_csi[6]= Q_3_p_csi(csi,eta,sigma)
        N_p_csi[7]= R_3_p_csi(csi,eta,sigma)
        N_p_csi[8]= S_3_p_csi(csi,eta,sigma)
        return N_p_csi

    def der_eta(self, csi, eta):

        # Evaluate shape derivative
        # with respect to eta

        ndim = self.ndof
        N_p_eta = np.zeros((ndim,1))
        sigma = 1.0-csi-eta
        N_p_eta[0]= Q_1_p_eta(csi,eta,sigma)
        N_p_eta[1]= R_1_p_eta(csi,eta,sigma)
        N_p_eta[2]= S_1_p_eta(csi,eta,sigma)
        N_p_eta[3]= Q_2_p_eta(csi,eta,sigma)
        N_p_eta[4]= R_2_p_eta(csi,eta,sigma)
        N_p_eta[5]= S_2_p_eta(csi,eta,sigma)
        N_p_eta[6]= Q_3_p_eta(csi,eta,sigma)
        N_p_eta[7]= R_3_p_eta(csi,eta,sigma)
        N_p_eta[8]= S_3_p_eta(csi,eta,sigma)
        return N_p_eta

    def get_T_matrix(self, J):

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, J[0,0], J[1,0]],
            [0.0, J[0,1], J[1,1]]
        ])

        T = np.kron(np.eye(self.nodelm,dtype=float), A)
        return T


    
    
class ShapeFunctionLagrange(_ShapeFunction):
    def __init__(self, ngauss=3):
        _ShapeFunction.__init__(self, ngauss)
        self.nodelm = 3
        self.ndof_per_node = np.array([1,1,1])
        self.ndof = self.ndof_per_node.sum()
        _ShapeFunction.__init__(self, ngauss)
        self._get_shape()
        self._integrate()
        
    def fun(self, csi,eta):
        ndim = self.ndof
        N = np.zeros((ndim,1))
        sigma = 1.0-csi-eta
        N[0]= L_1(csi,eta,sigma)
        N[1]= L_2(csi,eta,sigma)
        N[2]= L_3(csi,eta,sigma)
        return N
    
    def der_csi(self, csi, eta):
        ndim = self.ndof
        N_p_csi = np.zeros((ndim,1))
        sigma = 1.0-csi-eta
        N_p_csi[0]= L_1_p_csi(csi,eta,sigma)
        N_p_csi[1]= L_2_p_csi(csi,eta,sigma)
        N_p_csi[2]= L_3_p_csi(csi,eta,sigma)
        return N_p_csi
    
    def der_eta(self, csi, eta):
        ndim = self.ndof
        N_p_eta = np.zeros((ndim,1))
        sigma = 1.0-csi-eta
        N_p_eta[0]= L_1_p_eta(csi,eta,sigma)
        N_p_eta[1]= L_2_p_eta(csi,eta,sigma)
        N_p_eta[2]= L_3_p_eta(csi,eta,sigma)
        return N_p_eta

    def get_T_matrix(self, J):
        return np.eye(self.ndof)

class ShapeFunctionLagrangeQuadratic(_ShapeFunction):
    def __init__(self, ngauss=12):
        _ShapeFunction.__init__(self, ngauss)
        self.nodelm = 6
        self.ndof_per_node = np.array([1,1,1,1,1,1])
        self.ndof = self.ndof_per_node.sum()
        _ShapeFunction.__init__(self, ngauss)
        self._get_shape()
        self._integrate()
        
    def fun(self, csi,eta):
        ndim = self.ndof
        N = np.zeros((ndim,1))

        N[0]= LQ_1(csi,eta)
        N[1]= LQ_2(csi,eta)
        N[2]= LQ_3(csi,eta)
        N[3]= LQ_4(csi,eta)
        N[4]= LQ_5(csi,eta)
        N[5]= LQ_6(csi,eta)
        return N
    
    def der_csi(self, csi, eta):
        ndim = self.ndof
        N_p_csi = np.zeros((ndim,1))

        N_p_csi[0]= LQ_1_p_csi(csi,eta)
        N_p_csi[1]= LQ_2_p_csi(csi,eta)
        N_p_csi[2]= LQ_3_p_csi(csi,eta)
        N_p_csi[3]= LQ_4_p_csi(csi,eta)
        N_p_csi[4]= LQ_5_p_csi(csi,eta)
        N_p_csi[5]= LQ_6_p_csi(csi,eta)
        return N_p_csi
    
    def der_eta(self, csi, eta):
        ndim = self.ndof
        N_p_eta = np.zeros((ndim,1))

        N_p_eta[0]= LQ_1_p_eta(csi,eta)
        N_p_eta[1]= LQ_2_p_eta(csi,eta)
        N_p_eta[2]= LQ_3_p_eta(csi,eta)
        N_p_eta[3]= LQ_4_p_eta(csi,eta)
        N_p_eta[4]= LQ_5_p_eta(csi,eta)
        N_p_eta[5]= LQ_6_p_eta(csi,eta)
        return N_p_eta

    def get_T_matrix(self, J):
        return np.eye(self.ndof)


class ShapeFunctionLH6(_ShapeFunction):
    def __init__(self, ngauss=12):
        self.nodelm = 4
        self.ndof_per_node = np.array([1,1,3,1])
        self.ndof = self.ndof_per_node.sum()
        _ShapeFunction.__init__(self, ngauss)
        self._get_shape()
        self._integrate()
        
    def fun(self, csi, eta):

        # Evaluate shape function

        ndim = self.ndof
        N = np.zeros((ndim,1))

        N[0]= LH6_1(csi,eta)
        N[1]= LH6_2(csi,eta)
        N[2]= LH6_3(csi,eta)
        N[3]= LH6_3_x(csi,eta)
        N[4]= LH6_3_y(csi,eta)
        N[5]= LH6_4(csi,eta)
        return N

    def der_csi(self, csi, eta):

        # Evaluate shape derivative
        # with respect to csi

        ndim = self.ndof
        N_p_csi = np.zeros((ndim,1))

        N_p_csi[0]= LH6_1_p_csi(csi,eta)
        N_p_csi[1]= LH6_2_p_csi(csi,eta)
        N_p_csi[2]= LH6_3_p_csi(csi,eta)
        N_p_csi[3]= LH6_3_x_p_csi(csi,eta)
        N_p_csi[4]= LH6_3_y_p_csi(csi,eta)
        N_p_csi[5]= LH6_4_p_csi(csi,eta)

        return N_p_csi

    def der_eta(self, csi, eta):

        # Evaluate shape derivative
        # with respect to eta

        ndim = self.ndof
        N_p_eta = np.zeros((ndim,1))
        N_p_eta[0]= LH6_1_p_eta(csi,eta)
        N_p_eta[1]= LH6_2_p_eta(csi,eta)
        N_p_eta[2]= LH6_3_p_eta(csi,eta)
        N_p_eta[3]= LH6_3_x_p_eta(csi,eta)
        N_p_eta[4]= LH6_3_y_p_eta(csi,eta)
        N_p_eta[5]= LH6_4_p_eta(csi,eta)

        return N_p_eta

    def get_T_matrix(self, J):

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, J[0,0], J[1,0]],
            [0.0, J[0,1], J[1,1]]
        ])
        
        T = np.block([
            [np.eye(2),np.zeros((2,3)),np.zeros((2,1))],
            [np.zeros((3,2)),A,np.zeros((3,1))],
            [np.zeros((1,5)),1]
        ])
        return T


class ShapeFunctionLH7(_ShapeFunction):
    def __init__(self, ngauss=12):
        self.nodelm = 3
        self.ndof_per_node = np.array([3,3,1])
        self.ndof = self.ndof_per_node.sum()
        _ShapeFunction.__init__(self, ngauss)
        self._get_shape()
        self._integrate()
        
    def fun(self, csi, eta):

        # Evaluate shape function

        ndim = self.ndof
        N = np.zeros((ndim,1))

        N[0]= LH7_1(csi,eta)
        N[1]= LH7_1_x(csi,eta)
        N[2]= LH7_1_y(csi,eta)
        N[3]= LH7_2(csi,eta)
        N[4]= LH7_2_x(csi,eta)
        N[5]= LH7_2_y(csi,eta)
        N[6]= LH7_3(csi,eta)
        return N

    def der_csi(self, csi, eta):

        # Evaluate shape derivative
        # with respect to csi

        ndim = self.ndof
        N_p_csi = np.zeros((ndim,1))

        N_p_csi[0]= LH7_1_p_csi(csi,eta)
        N_p_csi[1]= LH7_1_x_p_csi(csi,eta)
        N_p_csi[2]= LH7_1_y_p_csi(csi,eta)
        N_p_csi[3]= LH7_2_p_csi(csi,eta)
        N_p_csi[4]= LH7_2_x_p_csi(csi,eta)
        N_p_csi[5]= LH7_2_y_p_csi(csi,eta)
        N_p_csi[6]= LH7_3_p_csi(csi,eta)

        return N_p_csi

    def der_eta(self, csi, eta):

        # Evaluate shape derivative
        # with respect to eta

        ndim = self.ndof
        N_p_eta = np.zeros((ndim,1))
        N_p_eta[0]= LH7_1_p_eta(csi,eta)
        N_p_eta[1]= LH7_1_x_p_eta(csi,eta)
        N_p_eta[2]= LH7_1_y_p_eta(csi,eta)
        N_p_eta[3]= LH7_2_p_eta(csi,eta)
        N_p_eta[4]= LH7_2_x_p_eta(csi,eta)
        N_p_eta[5]= LH7_2_y_p_eta(csi,eta)
        N_p_eta[6]= LH7_3_p_eta(csi,eta)

        return N_p_eta

    def get_T_matrix(self, J):

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, J[0,0], J[1,0]],
            [0.0, J[0,1], J[1,1]]
        ])
        
        T = np.block([
            [A,np.zeros((3,4))],
            [np.zeros((3,3)),A,np.zeros((3,1))],
            [np.zeros((1,6)),1]
        ])
        return T


class ShapeFunctionProduct(_ShapeFunction):
    def __init__(self, vsh, wsh):
        self.vsh = vsh
        self.wsh = wsh

        self.gpar = GaussParams(np.max([self.vsh.gpar.n, self.vsh.gpar.n ]))
        
        # matrix of gauss weights (nodelm*ndof x ngauss)
        self.WT = None
        
        # fun and der shape (nodelm*ndof x ngauss)
        self.N = None
        self.N_p_csi = None
        self.N_p_eta = None
        
        # shape integrals (nodelm*ndof x nodelm*ndof)
        self.B1_B1= None
        self.B2_B1= None
        self.B1_B2= None
        self.B2_B2= None
        self.N_B1 = None
        self.B1_N = None
        self.N_B2 = None
        self.B2_N = None
        self.N_N  = None

        self.ndofV = self.vsh.ndof
        self.ndofW = self.wsh.ndof

        self.ndof = self.vsh.ndof + self.wsh.ndof

        self._get_shape()
        self._integrate()

    def _get_shape(self):
        
        # compute shape functions at gauss pts
        
        N = np.vstack([self.vsh.N, self.wsh.N ])
        N_p_csi = np.vstack([self.vsh.N_p_csi, self.wsh.N_p_csi])
        N_p_eta = np.vstack([self.vsh.N_p_eta, self.wsh.N_p_eta])

        self.N = N
        self.N_p_csi = N_p_csi
        self.N_p_eta = N_p_eta
    
    def get_T_matrix(self, J):
        T_vsh = self.vsh.get_T_matrix(J)
        T_wsh = self.wsh.get_T_matrix(J)
        T = np.block([
            [T_vsh,np.zeros((self.vsh.ndof,self.wsh.ndof))],
            [np.zeros((self.wsh.ndof,self.vsh.ndof)),T_wsh]
        ])
        return T



###########################################
# geometrical shape functions
#class GeometricShapeFunction():
#    def __init__(self, node_map, x, y):
#        self.x = x
#        self.y = y
#        self.node_map = map
#
#    def _set_geometric_shape_functions_kind(self):
#        if self.node_map == [(0,0),(1,0),(0,1)]:
#            self.kind = '1'
#        if self.node_map == [(0,0),(0,1),(1,0)]:
#            self.kind = '2'
#        if self.node_map == [(0,1),(0,0),(1,0)]:
#            self.kind = '3'
#        if self.node_map == [(1,0),(0,0),(0,1)]:
#            self.kind = '4'
#        if self.node_map == [(1,0),(0,1),(0,0)]:
#            self.kind = '5'
#        if self.node_map == [(0,1),(1,0),(0,0)]:
#            self.kind = '6'
#    
#    def get_jacobian(self):
#        if self.kind == '1':
#            self.J = np.array([
#                [self.x[0]-self.x[2],self.x[1]-self.x[2]],
#                [self.y[0]-self.y[2],self.y[1]-self.y[2]]
#            ])    
    
# test section
if __name__ == '__main__': # When run for testing only
    shape_herm = ShapeFunctionHermite(nodelm = 3, ngauss = 12)
    shape_lag = ShapeFunctionLagrange(nodelm = 3, ngauss = 12)
    print(shape_herm.N_N)


    
