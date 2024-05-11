import numpy as np
from nwkpy.fem.mesh.ffem_io import ffmsh_2d_size_read, ffmsh_2d_data_read, ffmsh_2d_size_print, ffmsh_2d_data_print
import matplotlib.tri as tri
from nwkpy import _constants
from nwkpy import _common
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import coo_matrix, triu

class BaseMesh:
    def __init__(self, mesh_name, bandwidth_reduction = True):
        self.mesh_name = mesh_name
        self.bandwidth_reduction = bandwidth_reduction
        
        self._read_mesh()
        self._set_trifinder()
        
        if self.bandwidth_reduction:
            self._renumber_nodes()
        
        #self.elems = None
        #self.bn = None
        #self.outn = None
        
        #self.interfaces=None
        #self.itab=None
        
        #self.material = None
    
    def _read_mesh(self):
        
        v_num, e_num, t_num = ffmsh_2d_size_read( self.mesh_name )
        v_xy, v_l, e_v, e_l, t_v, t_l = ffmsh_2d_data_read(self.mesh_name , v_num, e_num, t_num )
        
        self.nelem = t_num
        self.ng_nodes = v_num
        self.vertices = v_xy.T / _constants.length_scale
        self.triangles = t_v.T-1
        self.nodelm = self.triangles.shape[1]
        self.e_v = e_v.T.astype(int)-1
        self.e_l = e_l
        
        self.t_l = t_l

        # all the labels are set to 0
        self.v_l = v_l * 0
        
        # obtain the label values of the border edges
        self.border_labels = np.unique(e_l[e_l>0])

        # obtain the region labels
        self.region_labels = np.unique(t_l)
        
    def _set_borders(self):
        if self.restrict_to is not None:
            self.elems = np.arange(self.nelem)[np.isin(self.t_l,list(range(1,self.restrict_to+1)))]
            self.bn = np.where(self.v_l==self.restrict_to)[0]
            outn = np.unique( self.triangles[self.t_l > self.restrict_to].flatten() )   # all the outer nodes
            self.outn = np.delete( outn , np.isin( outn , self.bn) )   # remove the border nodes
        else:
            self.elems = np.arange(self.nelem)

            # get the border node from the edges label of the outer border edges
            bn =[]
            # obtain the label values of the border edges
            for blab in self.border_labels:
               bn.append(np.unique(self.e_v[np.where(self.e_l==blab)[0]]))
               # some of the border node may come from splitted edges, aadd them too
               bn.append(np.where( np.isin ( self.v_l , blab ) )[0])
            self.bn = np.unique(np.hstack(bn))
            self.outn = np.array([], dtype=int)       
           
    def _set_material(self):
        self.material = np.vectorize(self.reg2mat.get)(self.t_l)
        self.particle_per_elem = np.vectorize(self.mat2partic.get)(self.material)

    def _set_trifinder(self):
        self.triangulation = tri.Triangulation(self.vertices[:,0], self.vertices[:,1], self.triangles)
        self.trifinder = self.triangulation.get_trifinder()
    
    def _renumber_nodes(self):
        
        # node renumbering using reverse Cuthill-McGee algorithm
        
        # mesh edges from triangulation
        edges = self.triangulation.edges
        
        # append duplicate edges
        edges = np.vstack( [ edges , edges[:,::-1] ] )
        ned = edges.shape[0]
        
        # adjacency graph in dense format
        adjgr = np.hstack([edges, np.ones((ned,1))]).astype(int)
        shape = tuple(adjgr.max(axis=0)[:2]+1)
        
        # adjacency matrix in sparse format
        adjmat = coo_matrix((adjgr[:, 2], (adjgr[:, 0], adjgr[:, 1])), shape=shape,
                                dtype=adjgr.dtype).tocsr()
        
        cmgr = reverse_cuthill_mckee(adjmat, symmetric_mode=True)
        cmdict = _common.build_dict(list(cmgr), list(np.arange(self.ng_nodes)))
        
        # renumber nodes using the obtained map dict
        self.triangles = np.vectorize(cmdict.get)(self.triangles)
        self.e_v = np.vectorize(cmdict.get)(self.e_v)
        
        # reshuffle vertices and vert labels according to the node reordering
        self.vertices = self.vertices[cmgr]
        self.v_l = self.v_l[cmgr]
        
        # re-set triangulation object
        self._set_trifinder()

    def InsertInterpolationNodes(self, v_l_to_split):
        edges = Tri2Edge(self.vertices, self.triangles)
        self.edges = edges
        edges_to_split = FindEdgesToSplit(self.triangles, edges, self.v_l, v_l_to_split)
        self.vertices, self.triangles = SplitEdges(self.vertices,self.triangles, edges_to_split)

        # update vertices labels
        self.v_l = np.hstack([self.v_l,np.full(edges_to_split.shape[0],v_l_to_split)])
        self.ng_nodes += edges_to_split.shape[0]

        # update dof per node table
        #self.dlnc = np.hstack([self.dlnc,np.full(edges_to_split.shape[0],1)])
        #self.dlnc[np.argwhere(self.v_l==1)[:,0]] = 1
    
    def ChangeP1toP2Mesh(self):
        self.vertices, self.triangles, self.v_l, self.edges, self.edges_label = ChangeP1toP2Mesh(
            self.vertices, self.triangles, self.v_l, self.e_v, self.e_l
            )

        # update vertices labels
        self.ng_nodes = self.vertices.shape[0]
        self._set_borders()
    
    def size_print(self):
        ffmsh_2d_size_print( 'MESH SIZE DATA', self.ng_nodes, len(self.e_l), self.nelem )

    def delete_cpp_objects(self):
        self.triangulation._cpp_triangulation = None
        self.trifinder = None
        self.triangulation._trifinder = None
        
    def get_cpp_objects(self):
        self.triangulation._cpp_triangulation = self.triangulation.get_cpp_triangulation()
        self.trifinder = self.triangulation.get_trifinder()

    # symmetry operations on the mesh can be represented as permuations matrices 
    # if the mesh is invariant under the symmetry operation

    def represent_symmetry_group(self, symg):
        symops = {}
        for n in symg.symops.keys():
            Utilde = Uinv @ symg.symops[n].mat @ U
            vertices_transf = self.vertices @ Utilde.T
            p = np.zeros(self.ng_nodes,dtype=int)
            for i in range(len(p)):
               p[i] = np.where(np.isclose(vertices_transf,(self.vertices[i,0], self.vertices[i,1])).all(axis=1))[0][0]
            symops[n] = p
        self.symops = symops

class Mesh(BaseMesh):
    def __init__(
            self, 
            mesh_name,
            reg2mat=None, 
            mat2partic=None,
            restrict_to = None, 
            bandwidth_reduction=True
        ):
        BaseMesh.__init__(self, mesh_name, bandwidth_reduction)

        if reg2mat is not None:
            self.reg2mat = reg2mat
        else:
            self.reg2mat = {key: "Unknown material" for key in self.region_labels}

        if mat2partic is not None:
            self.mat2partic = mat2partic
        else:
            self.mat2partic = {key: "Unknown particle" for key in self.region_labels}
        
        self.restrict_to = restrict_to
        
        self._set_borders()
        self._set_material()     

class RectangularMesh:
    def __init__(self, Lx, Ly, nx=64, ny=64, area=None):
        self.Lx = Lx
        self.Ly = Ly

        if area is not None:
            nx = int( np.sqrt( Lx * Ly / area) )
            ny = int( np.sqrt( Lx * Ly / area) )
        self.nx = nx
        self.ny = ny

        self.dx = Lx/nx
        self.dy = Ly/ny

        self.nxs = 2*nx-1
        self.nys = 2*ny-1
        self.dxs = Lx/self.nxs
        self.dys = Ly/self.nys

##############################################################################
##############################################################################

# COMMON FUNCTIONS
def Tri2Edge(p,t):

    nnp = p.shape[0]
    nnt = t.shape[0]
    
    i = t[:,0]
    j = t[:,1]
    k = t[:,2]
    
    A = coo_matrix((np.full(nnt,-1), (j, k)), shape=(nnp, nnp))
    A = A + coo_matrix((np.full(nnt,-1), (i, k)), shape=(nnp, nnp))
    A = A + coo_matrix((np.full(nnt,-1), (i, j)), shape=(nnp, nnp))
    A = ((A + A.T)<0)*-1
    A = triu(A)
    r,c,v = A.row, A.col, A.data
    v = np.arange(len(v))
    A = coo_matrix((v, (r, c)), shape=(nnp, nnp))
    A = A + A.T
    edges = np.zeros((nnt,3))
    for k in range(nnt):
        edges[k,:] = np.array([A[t[k,1],t[k,2]],
                               A[t[k,0],t[k,2]],
                               A[t[k,0],t[k,1]]])
    edges = edges.astype(int)
    return edges

def ChangeP1toP2Mesh_simple(p,t):
    nnp = p.shape[0]
    edges = Tri2Edge(p,t)
    npp_to_add = len(np.unique(edges))
    p = np.vstack([p,np.zeros((npp_to_add,2))])
    
    edges = edges+nnp
    
    i = t[:,0]
    j = t[:,1]
    k = t[:,2]
    
    e=edges[:,0]
    p[e,0] = 0.5*(p[j,0]+p[k,0])
    p[e,1] = 0.5*(p[j,1]+p[k,1])
    
    e=edges[:,1]
    p[e,0] = 0.5*(p[i,0]+p[k,0])
    p[e,1] = 0.5*(p[i,1]+p[k,1])
    
    e=edges[:,2]
    p[e,0] = 0.5*(p[i,0]+p[j,0])
    p[e,1] = 0.5*(p[i,1]+p[j,1])
    
    t = np.hstack([t,np.zeros((t.shape[0],3),dtype=int)])
    
    t[:,3:] = edges
    return p, t

def ChangeP1toP2Mesh(p, t, p_l, e_v, e_l):
    nnp = p.shape[0]
    edges = Tri2Edge(p,t)
    edges_label = EdgeToLabel(edges, t, e_v, e_l)
    npp_to_add = len(np.unique(edges))
    p = np.vstack([p,np.zeros((npp_to_add,2))])
    p_l = np.hstack([p_l,np.zeros(npp_to_add,dtype=int)])
    edges = edges+nnp
    
    i = t[:,0]
    j = t[:,1]
    k = t[:,2]
    
    e=edges[:,0]
    el = edges_label[:,0]
    p[e,0] = 0.5*(p[j,0]+p[k,0])
    p[e,1] = 0.5*(p[j,1]+p[k,1])
    p_l[e] = el
    
    e=edges[:,1]
    el = edges_label[:,1]
    p[e,0] = 0.5*(p[i,0]+p[k,0])
    p[e,1] = 0.5*(p[i,1]+p[k,1])
    p_l[e] = el
    
    e=edges[:,2]
    el = edges_label[:,2]
    p[e,0] = 0.5*(p[i,0]+p[j,0])
    p[e,1] = 0.5*(p[i,1]+p[j,1])
    p_l[e] = el
    
    t = np.hstack([t,np.zeros((t.shape[0],3),dtype=int)])
    
    t[:,3:] = edges

    # update with the new edges
    #edges = Tri2Edge(p,t)
    # update edges label ???
    return p, t, p_l, edges, edges_label

def EdgeToLabel(edges, t, e_v, e_l):
    
    edge_label = np.zeros(edges.shape,dtype=int)
    
    e = edges[:,0]
    v1 = t[:,1]
    v2 = t[:,2]
    edge_vertices_jk = np.vstack([v1,v2]).T
    edge_label_jk = AssignLabel(edge_vertices_jk, e_v, e_l)
    edge_label[:,0] = edge_label_jk
    
    e = edges[:,1]
    v1 = t[:,0]
    v2 = t[:,2]
    edge_vertices_ik = np.vstack([v1,v2]).T
    edge_label_ik = AssignLabel(edge_vertices_ik, e_v, e_l)
    edge_label[:,1] = edge_label_ik
    
    e = edges[:,2]
    v1 = t[:,0]
    v2 = t[:,1]
    edge_vertices_ij = np.vstack([v1,v2]).T
    edge_label_ij = AssignLabel(edge_vertices_ij, e_v, e_l)
    edge_label[:,2] = edge_label_ij

    return edge_label

def AssignLabel(edge_vertices, e_v, e_l):
    edge_labels = np.zeros(edge_vertices.shape[0],dtype=int)
    for i in range(edge_vertices.shape[0]):
        idx = np.where( np.logical_or(e_v==edge_vertices[i,:],e_v==edge_vertices[i,::-1] ).all(axis=1))[0]
        if idx.size>0:
            edge_labels[i] = e_l[idx]
    return np.array(edge_labels)

## con questa procedura ottengo gli edge da splittare

# funziona solo se le label fornite in "v_l_to_split" corrispondono ad interfacce
# fra due materiali. In altre parole devono coincidere con degli esagoni interni alla strutture
# non può essere usata per splittare i triangoli che stanno sul bordo della struttura !
# andrebbe generalizzata... tutto il resto però mi sembra indipendente. Una volta ottenuti
# gli indici dei lati da splittare funziona tutto a meraviglia

def map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val

def FindEdgesToSplit(t, edges, v_l, v_l_to_split):
    a = np.argwhere(np.isin(t,np.argwhere(np.isin(v_l,v_l_to_split)).squeeze()))   
    values, counts = np.unique(a[:,0], return_counts=True)   
    b = a[np.isin(a[:,0],values[np.argwhere(counts==2).squeeze()])]    
    c = np.hstack([np.unique(b[:,0])[:,np.newaxis],b[:,1].reshape((int(b[:,1].shape[0]/2),2)).sum(axis=1)[:,np.newaxis]])
    vfunc  = np.vectorize(map_func)
    d = dict(zip([1,2,3], [2,1,0]))
    c[:,1] = vfunc(c[:,1], d)
    edges_to_split = edges[c[:,0],c[:,1]]
    edges_to_split = np.unique(edges_to_split)
    return edges_to_split

def SplitEdges(p,t, edges_to_split):
    npp = p.shape[0]
    p, t = ChangeP1toP2Mesh_simple(p,t)
    mask = np.logical_not(np.logical_or(t<npp ,np.isin(t,edges_to_split+npp)))
    t[mask] = -1
    p=p[np.unique(t[t>=0]),:]
    vfunc  = np.vectorize(map_func)
    d = dict(zip(list(np.unique(np.sort(edges_to_split+npp))), list(np.arange(npp,npp+len(edges_to_split)))))
    t = vfunc(t, d)
    return p, t

######################################################################################################################
######################################################################################################################

# mesh generating functions
import nwkpy.fem.mesh.pyFreeFem as pyff
from nwkpy.fem.mesh import ffem_script

def test_mesh():
    WorkBlock = pyff.edpBlock(content=ffem_script.test_mesh_script, name = 'WORK')
    script=pyff.edpScript([WorkBlock])
    #script.pprint()
    script.run(verbose=True)


def Hex2regsymm(mesh_name, total_width, shell_width, edges_per_border):

    nC1 = edges_per_border['nC1']
    nC2 = edges_per_border['nC2']
    nC3 = edges_per_border['nC3']
    nC4 = edges_per_border['nC4']
    nC5 = edges_per_border['nC5']
    nC6 = edges_per_border['nC6']

    # Structure
    in1 = pyff.edpInput(name='core_width[nm]', data_type='real', source=total_width,FreeFem_name='tw')
    in2 = pyff.edpInput(name='shell_width[nm]', data_type='real', source=shell_width, FreeFem_name='sw')

    # Number of points for each Border
    inC1 = pyff.edpInput(name='n', data_type='real', source=nC1, FreeFem_name='nC1')
    inC2 = pyff.edpInput(name='n', data_type='real', source=nC2, FreeFem_name='nC2')
    inC3 = pyff.edpInput(name='n', data_type='real', source=nC3, FreeFem_name='nC3')
    inC4 = pyff.edpInput(name='n', data_type='real', source=nC4, FreeFem_name='nC4')
    inC5 = pyff.edpInput(name='n', data_type='real', source=nC5, FreeFem_name='nC5')
    inC6 = pyff.edpInput(name='n', data_type='real', source=nC6, FreeFem_name='nC6')

    
    # Number of points for each Border
    inputs = [in1,in2,inC1,inC2,inC3,inC4,inC5,inC6]

    InputBlock = pyff.edpBlock('', name = 'INPUT', input = inputs) 

    WorkBlock = pyff.edpBlock(content=ffem_script.Hex2reg_symm_ffem_script, name = 'WORK') 

    content = """savemesh(Thwholed," """.strip()+mesh_name+""" ");""".strip()
    SaveBlock = pyff.edpBlock(content=content, name = 'SAVE')
    content = """plot(Thwholed,ps=" """.strip()+mesh_name[:-4]+'.eps'+""" ");""".strip()
    #content = """plot(Thwholed);""".strip()
    PlotBlock = pyff.edpBlock(content=content, name = 'PLOT')

    script=pyff.edpScript([InputBlock,WorkBlock,SaveBlock,PlotBlock])
    script.run()
    return

def Hex3regsymm(mesh_name, reg2mat, mat2partic, tw, sw, bw, np):

    nC1 = np['nC1']
    nC2 = np['nC2']
    nC3 = np['nC3']
    nC4 = np['nC4']
    nC5 = np['nC5']
    nC6 = np['nC6']
    nC7 = np['nC7']
    nC8 = np['nC8']
    nC9 = np['nC9']

    # Structure
    in1 = pyff.edpInput(name='core_width[nm]', data_type='real', source=tw,FreeFem_name='tw')
    in2 = pyff.edpInput(name='shell_width[nm]', data_type='real', source=sw, FreeFem_name='sw')
    in3 = pyff.edpInput(name='barrier_width[nm]', data_type='real', source=bw, FreeFem_name='bw')

    # Number of points for each Border
    inC1 = pyff.edpInput(name='n', data_type='real', source=nC1, FreeFem_name='nC1')
    inC2 = pyff.edpInput(name='n', data_type='real', source=nC2, FreeFem_name='nC2')
    inC3 = pyff.edpInput(name='n', data_type='real', source=nC3, FreeFem_name='nC3')
    inC4 = pyff.edpInput(name='n', data_type='real', source=nC4, FreeFem_name='nC4')
    inC5 = pyff.edpInput(name='n', data_type='real', source=nC5, FreeFem_name='nC5')
    inC6 = pyff.edpInput(name='n', data_type='real', source=nC6, FreeFem_name='nC6')
    inC7 = pyff.edpInput(name='n', data_type='real', source=nC7, FreeFem_name='nC7')
    inC8 = pyff.edpInput(name='n', data_type='real', source=nC8, FreeFem_name='nC8')
    inC9 = pyff.edpInput(name='n', data_type='real', source=nC9, FreeFem_name='nC9')

    
    # Number of points for each Border
    inputs = [in1,in2,in3,inC1,inC2,inC3,inC4,inC5,inC6,inC7,inC8,inC9]

    InputBlock = pyff.edpBlock('', name = 'INPUT', input = inputs) 

    WorkBlock = pyff.edpBlock(content=ffem_script.Hex3reg_symm_ffem_script, name = 'WORK') 

    content = """savemesh(Thwholed," """.strip()+mesh_name+""" ");""".strip()
    SaveBlock = pyff.edpBlock(content=content, name = 'SAVE')

    script=pyff.edpScript([InputBlock,WorkBlock,SaveBlock])
    script.run()

    # at this point a mesh file has been generated... it remains to read it and create a mesh object

    # write the correct path to the mesh to be read
    mesh = Mesh(
        mesh_name,
        reg2mat,
        mat2partic,
        restrict_to = None,
        bandwidth_reduction=True
    )
    return mesh


phiuv = 2.*np.pi/3.
U = np.array([
    [1.0, -(1/np.tan(phiuv))],
    [0.0, 1/np.sin(phiuv)]
])

Uinv = np.array([
    [1.0, np.cos(phiuv)],
    [0.0, np.sin(phiuv)]
])







    
