"""
In this example a set of different meshes is generated using FreeFem++.
The output meshes are saved into the folder OUTDATA
"""

################# import section ######################
import numpy as np
import sys
sys.path.append("../../")
#sys.path.append("/e/ANDREA/projects/python_library_paper/nwkp")
# base
from nwkp.fem import Hex2regsymm, Mesh
from nwkp import tic, toc
from nwkp import Logger

# input file
import indata
################################################################

################### General settings ###############

# where to write output data
path = indata.path

logger = Logger(rank=0, logfile_path=path)
logger.logga(size=1)

# create the mesh object
reg2mat = indata.reg2mat
mat2partic = indata.mat2partic

############# Mesh generation  #############

def main():

    Hex2regsymm(
        mesh_name = indata.mesh_name,
        total_width=indata.total_width,
        shell_width=indata.shell_width,
        edges_per_border=indata.edges_per_border
    )
    logger.write(value='####################################')
    
    # at this point a mesh file has been generated... 
    # it remains to read it, create a mesh object and obtain some geometrical information

    mesh = Mesh(
        indata.mesh_name
    )

    logger.write(key="Number of vertices = ", value=mesh.ng_nodes)
    logger.write(key="Number of boundary edges = ", value=len(mesh.e_l))

    logger.write(key="Number of elements = ", value=mesh.nelem)
    logger.write(key="Boundary labels = ", value=mesh.border_labels)

    logger.write(key="boundary edges with label 1 = ", value=np.count_nonzero(mesh.e_l == 1))
    logger.write(key="boundary edges with label 2= ", value=np.count_nonzero(mesh.e_l == 2))
    logger.write(key="boundary edges with label 3= ", value=np.count_nonzero(mesh.e_l == 3))
    logger.write(key="boundary edges with label 4= ", value=np.count_nonzero(mesh.e_l == 4))
    logger.write(key="boundary edges with label 5= ", value=np.count_nonzero(mesh.e_l == 5))
    logger.write(key="boundary edges with label 6= ", value=np.count_nonzero(mesh.e_l == 6))
    
    logger.write(key="Materials = ", value=mesh.reg2mat)
    logger.write(key="Particles = ", value=mesh.mat2partic)
    logger.write(key="Region labels = ", value=mesh.region_labels)
    logger.write(key="Elements in region 1 = ", value=np.count_nonzero(mesh.t_l == 1))
    logger.write(key="Elements in region 2 = ", value=np.count_nonzero(mesh.t_l == 2))


if __name__=='__main__':
    tic()
    main()
    toc()