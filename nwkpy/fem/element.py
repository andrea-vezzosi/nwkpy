
import numpy as np
from nwkpy.fem.shape import ShapeFunctionHermite , ShapeFunctionLagrange, ShapeFunctionLH7, ShapeFunctionLH6, ShapeFunctionLagrangeQuadratic, ShapeFunctionProduct

"""
Finite element class
"""
class FiniteElement:
    """
    Finite element class
    
    Init Parameters
    ----------
    iel : int, index element
    mesh : mesh object, finite element mesh
    shape : shape function object
    
    Attributes
    ----------
    nods : ndarray, nodes indices
    nreg : int, region index
    material : object, material object
    
    el_v : 
    x_vert : ndarray, nodes x-coordinates
    y_vert : ndarray, nodes y-coordinates
    J : ndarray, jacobian matrix
    Jinv : ndarray, inverse jacobian matrix
    detJ : float, jacobian
    T : ndarray, derivative matrix
    
    ddi, di : 
    
    """
    def __init__(self, iel, mesh, shape):
        self.iel = iel
        self.mesh = mesh
        self.shape = shape
        
        #self.nods = mesh.triangles[iel]
        self.nods = mesh.triangles[iel][mesh.triangles[iel]>=0]
        self.nodelm = self.nods.shape[0]
        self.dof_per_node = None

        self._set_nod_coord()
        self._set_jacobian()
        self._set_T_matrix()
        self._set_integrated_derivatives()
        self._set_gauss_coords()
        self._set_integrated_Lz()
        
        self.nreg = self.mesh.t_l[iel]
        self.material = self.mesh.material[iel]

    def _set_geometrical_shape(self):
        return 0

    def _set_nod_coord(self):
        self.el_v = self.mesh.vertices[self.nods]
        self.x_vert = self.el_v[:,0][:3]
        self.y_vert = self.el_v[:,1][:3]
        
    def _set_jacobian(self):
        x = self.x_vert
        y = self.y_vert
        
        self.J = np.array([
            [x[0]-x[2],x[1]-x[2]],
            [y[0]-y[2],y[1]-y[2]]
        ])
        
        self.detJ = (x[0]-x[2])*(y[1]-y[2])-(x[1]-x[2])*(y[0]-y[2])
        self.Jinv = np.linalg.inv(self.J)
        
    def _set_T_matrix(self):    
        self.T = self.shape.get_T_matrix(self.J)
        
    def _set_integrated_derivatives(self):
        self.ddx  = self.shape.ddx(self.J, self.detJ, self.T)
        self.ddy  = self.shape.ddy(self.J, self.detJ, self.T) 
        self.dxl  = self.shape.dxl(self.J, self.detJ, self.T) 
        self.dxr  = self.shape.dxr(self.J, self.detJ, self.T) 
        self.dyl  = self.shape.dyl(self.J, self.detJ, self.T) 
        self.dyr  = self.shape.dyr(self.J, self.detJ, self.T) 
        self.dxdy = self.shape.dxdy(self.J, self.detJ, self.T)
        self.dydx = self.shape.dydx(self.J, self.detJ, self.T)
        self.I    = self.shape.I(self.J, self.detJ, self.T)

    def _set_integrated_Lz(self):
        self.Lz = self.shape.Lz(self.gauss_coords, self.J, self.detJ, self.T)

    
    def int_psi_f_psi(self, f):
        
        # method to integrate an arbitrary function
        # int psi* f psi on the real element
        # corresponding mass (potential) matrix is returned
        
        # f is ndarray evaluated at gauss pts
        
        # transform f to (nodelm*ndof x ngauss)
        nd = self.shape.ndof
        ng = self.shape.gpar.n
        f = np.tile( f  , nd ).reshape( ( nd , ng) )
        
        # integrate in reference element
        fint = np.dot( self.shape.WT * self.shape.N * f , self.shape.N.T ) * self.detJ
        
        # T matrix multiply
        fint = np.dot(np.dot( self.T.T , fint ), self.T )
        return fint
    
    def int_psi_f(self, f):
        
        # method to integrate an arbitrary function
        # int psi* f psi on the real element
        # corresponding load vector is returned        
        # f is ndarray evaluated at gauss pts
        
        # integrate in reference element
        fint = np.dot( self.shape.WT * self.shape.N , f ) * self.detJ
        
        # T matrix multiply
        fint = np.dot( self.T.T , fint )
        return fint

    def int_psi_f_psiy(self, f):
        
        # method to integrate an arbitrary function
        # int psi* f psi'_y on the real element
        # corresponding element matrix is returned        
        # f is ndarray evaluated at gauss pts
                
        # transform f to (nodelm*ndof x ngauss)
        nd = self.shape.ndof
        ng = self.shape.gpar.n
        f = np.tile( f  , nd ).reshape( ( nd , ng) )
        
        # integrate in reference element
        fint = np.dot( self.shape.WT * self.shape.N * f , (- self.J[0,1] * self.shape.N_p_csi.T + self.J[0,0] * self.shape.N_p_eta.T) /self.detJ ) * self.detJ
        
        # T matrix multiply
        fint = np.dot(np.dot( self.T.T , fint ), self.T )

        # the function return an element matrix. If you know the values of the function at the 
        # nodes of this element, you can just do psi_row @ matrix # psi_col, and this gives you the 
        # expectation value of the operator. This is ok for one component. If the solution has many component
        # you can just do matrix = np.kron( matrix ,  np.eye(ncom)  ), and then psi = psi.T.flatten and then vector multiply.
        # otherwise you just do the same for each component and in principle for each kz and also each subband, since the integral has already been performed 
        # 
        # Repeat for each element in the mesh to obtain the expectation value.
        return fint

    def int_psi_f_psix(self, f):
        
        # method to integrate an arbitrary function
        # int psi* f psi'_x on the real element
        # corresponding element matrix is returned        
        # f is ndarray evaluated at gauss pts
                
        # transform f to (nodelm*ndof x ngauss)
        nd = self.shape.ndof
        ng = self.shape.gpar.n
        f = np.tile( f  , nd ).reshape( ( nd , ng) )
        
        # integrate in reference element
        fint = np.dot( self.shape.WT * self.shape.N * f , (self.J[1,1] * self.shape.N_p_csi.T - self.J[1,0] * self.shape.N_p_eta.T) /self.detJ ) * self.detJ
        
        # T matrix multiply
        fint = np.dot(np.dot( self.T.T , fint ), self.T )
        return fint

    
    def int_f(self, f):
        
        # integrate arbitrary f(x,y) over real element
        
        # input
        # f : ndarray, function value on gauss pts
        
        # returns
        # fint: float, value of the integral
        
        wt = self.shape.gpar.wt
        fint = f @ wt.T * self.detJ
        
        return fint
        
    
    def get_refcoord(self, x , y):

        p_coords = np.array([[x-self.x_vert[2]],
                            [y-self.y_vert[2]]])
        local_coords = self.Jinv @ p_coords
        csi = local_coords[0]
        eta = local_coords[1]
        return csi, eta
    
    def get_realcoord(self, csi, eta):
        local_coords = np.array([[csi],
                                 [eta]])
        p_coords = self.J @ local_coords + np.array([[self.x_vert[2]],
                                                     [self.y_vert[2]]])
        x = p_coords[0]
        y = p_coords[1]
        return x, y
    
    def _set_gauss_coords(self):
        # compute gauss points on the real element
        gauss_coords = np.zeros((self.shape.gpar.n,2))
        for ig in range(self.shape.gpar.n):
            csi = self.shape.gpar.pt_x[ig]
            eta = self.shape.gpar.pt_y[ig]
            x , y =  self.get_realcoord(csi , eta)
            gauss_coords[ig,0] = x
            gauss_coords[ig,1] = y
            
        self.gauss_coords = gauss_coords
        
    
    def interp_sol(self, x , y , snod):
        # x : float, x coord inside element
        # y : float, y coord inside element
        # s : ndarray, solution on nodal points
        
        csi, eta = self.get_refcoord(x, y)
        N = self.shape.fun(csi, eta)
        #N_p_csi = self.shape.der_csi(csi,eta)
        #N_p_eta = self.shape.der_eta(csi,eta)
        
        s = snod.T @ (self.T.T @ N)
        
        return s
    
    def interp_sol_gauss(self, snod):
        
        # interpolate solution on gauss point
        # snod : ndarray, solution on nodal points
        # s : ndarray, solution on gauss points
        
        N = self.shape.N
        
        s = snod.T @ (self.T.T @ N)
        
        return s
        
    
    def get_adjel(self, enods):
        
        # return adjacent element index
        # enods: array or list, edge's nodes indices
        
        conec = self.mesh.triangles
        testarr = conec[self.iel, enods]
        a = np.zeros(conec.shape[0], dtype=int)
        for i in range(conec.shape[0]):
            if np.isin(conec[i,:],testarr).sum() == 2:
                a[i] = 1
        b = np.where(a)[0]
        mask = np.isin(b , self.iel , invert=True)

        return int(b[mask])


########## FINITE ELEMENT SPACE ############

class FemSpace:
    def __init__(self, mesh, shape_class_name):

        self.shape_class_name = shape_class_name

        if shape_class_name=='Lagrange':
            dlnc = np.full(mesh.ng_nodes,1)
        elif shape_class_name=='Hermite':
            dlnc = np.full(mesh.ng_nodes,3)
        elif shape_class_name=='LagrangeHermite':

            # insert new midside nodes in the mesh 
            #mesh.InsertInterpolationNodes(v_l_to_split=mesh.interfaces)
            self.mesh.InsertInterpolationNodes(v_l_to_split=1)

            # compute the correct "DLNC" table
            dlnc = np.full(mesh.ng_nodes,3)
            #mask = np.argwhere(np.isin(mesh.v_l,mesh.interfaces))[:,0]
            mask = np.argwhere(np.isin(mesh.v_l,1))[:,0]
            dlnc[mask] = 1

            # permute the "CONEC" table of each element for reference element consistency
            self.mesh.triangles = PermuteTri(t=mesh.triangles, dlnc = dlnc)
        elif shape_class_name=='LagrangeQuadratic':

            # insert new midside nodes in the mesh
            # this method also updates the vertices label according to the splitted edges
            mesh.ChangeP1toP2Mesh()

            # compute the correct "DLNC" table
            # all the nodes have only one dof
            dlnc = np.full(mesh.ng_nodes,1)

        # total number of nodes
        self.ng_nodes = mesh.ng_nodes
        
        # "CONEC" table
        self.conec = mesh.triangles

        self.dlnc = dlnc
        
        # store dlnc in a "cumulative" way
        self.dlnc_cumul = np.pad(np.cumsum(dlnc), (1,0), 'constant', constant_values=0)

        self.mesh = mesh
        
        # list of finite elements
        self.felems = []

        self.gauss_coords_global = np.zeros((12,2,self.mesh.nelem)) # hard coded !!

        self.detJ_per_elem = np.zeros(self.mesh.nelem)

        self._set_felems()

        self.total_area = self.get_total_area()

    
    def _set_felems(self):
         
        # loop over elements
        tdof_per_elem=[]
        nodes_per_elem = []
        for iel in self.mesh.elems:

            # nodes of this element
            nodes = self.mesh.triangles[iel][self.mesh.triangles[iel]>=0]
            # total number of dof
            tot_dof = self.dlnc[nodes].sum()

            num_nodes = nodes.shape[0]

            # select shape function constructor
            if num_nodes==3 and tot_dof==3:
                shape = ShapeFunctionLagrange()
            elif num_nodes==6 and tot_dof==6:
                shape = ShapeFunctionLagrangeQuadratic()
            elif num_nodes==3 and tot_dof==9:
                shape = ShapeFunctionHermite()
            elif num_nodes==3 and tot_dof==7:
                shape = ShapeFunctionLH7()
            elif num_nodes==4 and tot_dof==6:
                shape = ShapeFunctionLH6()
                            
            # instantiate element object
            fel = FiniteElement(iel, self.mesh, shape)
            fel.dof_per_node = self.dlnc[nodes]
            self.felems.append(fel)

            self.gauss_coords_global[:,:,iel] = fel.gauss_coords

            self.detJ_per_elem[iel] = fel.detJ

            nodes_per_elem.append(num_nodes)
            tdof_per_elem.append(tot_dof)
        
        self.gauss_coords_global = self.gauss_coords_global.swapaxes(2,1).swapaxes(0,1).reshape((12*self.mesh.nelem,2))

        self.nodes_per_elem = np.array(nodes_per_elem)
        self.tdof_per_elem = np.array(tdof_per_elem)

        self.shape = shape

    
    def get_total_area(self):
        """Return total mesh area in rescaled units 
        """

        total_area = 0.0
        for fel in self.felems:
            total_area += fel.int_f(np.full(fel.shape.gpar.n,1.0))
        return total_area
        
    def get_region_area(self, region_identifiers):
        region_area = 0.0
        for fel in self.felems:
            if fel.nreg in region_identifiers:
                region_area += fel.int_f(np.full(fel.shape.gpar.n,1.0))
        return region_area

    def get_el_areas(self):
        el_areas = []
        for fel in self.felems:
            el_areas.append( fel.int_f(np.full(fel.shape.gpar.n,1.0)) )
        el_areas = np.array(el_areas)
        return el_areas

class FemSpaceProduct():
    def __init__(self, Vfs, Wfs):
        self.Vfs = Vfs
        self.Wfs = Wfs
        self._set_product_felems()
    
    def _set_product_felems(self):
        
        # loop over elements
        self.product_felems = []
        for iel in self.Vfs.mesh.elems: 
            shape = ShapeFunctionProduct(vsh=self.Vfs.felems[iel].shape, wsh=self.Wfs.felems[iel].shape)
            fel = FiniteElement(iel, self.Vfs.mesh, shape)
            self.product_felems.append(fel)
            
################################################

def PermuteTri(t,dlnc):
    for i in range(t.shape[0]):
        triangle = t[i][t[i]>=0]
        nodelm = triangle.shape[0]
        triangle_dof = dlnc[triangle]
        ndof = triangle_dof.sum()
        if ndof == 6 and nodelm==4:
            # LH element with 4 nodes    
            op = np.argwhere(t[i,3:]>=0)[0][0]
            if op == 0:
                nroll=2
            elif op == 1:
                nroll=1
            elif op==2:
                nroll=0
            t[i,:3] = np.roll(t[i,:3], nroll)
        if ndof == 7 and nodelm==3:
            # LH element with 3 nodes
            op = np.argwhere(triangle_dof==1)[0][0]
            if op==2:
                nroll=0
            elif op==1:
                nroll=1
            elif op==0:
                nroll=2
            t[i,:3] = np.roll(t[i,:3], nroll)
    return t




