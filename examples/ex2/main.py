"""
In this example we show how to compute the band structure of a core-shell nanowire
"""

################# import section ######################
import sys
sys.path.append("../../")

import numpy as np
from mpi4py import MPI

# base
from nwkpy import tic, toc
from nwkpy import Logger

# input file
import indata

# interfaces
from nwkpy.fem import Mesh
from nwkpy import BandStructure, PoissonProblem
from nwkpy import FreeChargeDensity, ElectrostaticPotential, DopingChargeDensity
from nwkpy import  _constants

# numerical
from scipy.sparse import save_npz
import socket
import gc
import os
################################################################

################### General settings ###############
# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# where to write output data
outdata_path = indata.outdata_path
mesh_name = indata.mesh_name
logger = Logger(rank=0, logfile_path=outdata_path)
logger.logga(size=1)

# Display hostname andIP address
def host_IP(mylogger):
   try:
      hname = socket.gethostname()
      hip = socket.gethostbyname(hname)
      mylogger.write(key='Hostname: ', value=hname)
      mylogger.write(key="IP Address: ",value=hip)
   except:
      mylogger.write(value="Unable to get Hostname and IP")


############# A single band structure calculation ############

def main():


    mesh = Mesh(
        mesh_name = indata.mesh_name,
        reg2mat = indata.reg2mat,
        mat2partic = indata.mat2partic
    ) 

    nreg = len(mesh.region_labels)       
    
    # start loop on chemical potentials
    mu = indata.chemical_potential

    # kz values
    kzvals = indata.kzvals

    try:
        dk = kzvals[1] - kzvals[0]
    except IndexError:
        logger.write(value='Warning: dk = 0.0')
        dk = 0.0

    nk = len(kzvals)
    if rank == 0:
        np.save(outdata_path+'/kzvals', kzvals)


    # compute a subset of kzvalues in each node
    kmaxlocal = len(kzvals) // size
    kin = rank * kmaxlocal
    kfin = kin + kmaxlocal
    kzslice = np.s_[kin:kfin]
      
    
    # initialize the poisson problem the Poisson equation with no free-charge included
    p = PoissonProblem(
        mesh, 
        shape_class_name=indata.shape_kind_poisson,
        dirichlet=indata.dirichlet,
        electric_field=indata.electric_field,
        user_defined_parameters=indata.user_defined_params
    ) 
    
    idj = ''
            
    # use an electrostatic potential from a file
    init_pot_name = indata.init_pot_name
    
    if init_pot_name is not None:
        # read initial potential from external csv file
        Vin = np.load(init_pot_name)
        # write initial pot
        p.epot = ElectrostaticPotential(p.fs, V=Vin)
    else:
        # solve poisson problem
        p.run()
        # electrostatic potential energy array
        Vin = p.epot.V
    #if rank==0:
    #    np.save(path_mu+'/OUTPUT_DATA_EPOT'+str(idj), Vin, allow_pickle=False)
        
    logger.logga(kz=kzvals[kzslice])       
        
    bs = BandStructure(
        mesh = mesh,
        kzvals = kzvals[kzslice],
        valence_band_edges= indata.valence_band_edges,
        principal_axis_direction = indata.principal_axis_direction,
        temperature = indata.temperature,
        k = indata.k,
        e_search=indata.e_search,
        shape_functions = indata.shape_kind_kp,
        epot = p.epot,
        logger=logger,
        rescaling=indata.rescaling,
        user_defined_params=indata.user_defined_params
    )
    bs.run()
            
            
    # gathering of the band structure data
    neig = bs.bands.shape[1]

    sendbuf=bs.kzvals
    recvbuf = np.zeros([nk])
    comm.Allgather(sendbuf, recvbuf)
    bs.kzvals = recvbuf
    
    sendbuf=bs.bands
    recvbuf = np.zeros([nk, neig])
    comm.Allgather(sendbuf, recvbuf)
    bs.bands = recvbuf
            
    sendbuf = bs.spinor_distribution
    recvbuf = np.zeros([nk, 8, neig])
    comm.Allgather(sendbuf, recvbuf) 
    bs.spinor_distribution = recvbuf

    sendbuf = bs.norm_sum_region
    recvbuf = np.zeros([nk, nreg, neig])
    comm.Allgather(sendbuf, recvbuf)
    bs.norm_sum_region = recvbuf
            
    sendbuf = bs.psi_el
    recvbuf = np.zeros([nk, bs.psi_el.shape[1], bs.psi_el.shape[2],bs.psi_el.shape[3]],dtype='complex')
    comm.Allgather(sendbuf, recvbuf) 
    bs.psi_el = recvbuf

    sendbuf = bs.psi_h
    recvbuf = np.zeros([nk, bs.psi_h.shape[1], bs.psi_h.shape[2],bs.psi_h.shape[3]],dtype='complex')
    comm.Allgather(sendbuf, recvbuf) 
    bs.psi_h = recvbuf
    del(sendbuf)  
    del(recvbuf)

    # compute the charge on the root process
    rho_el = FreeChargeDensity(bs.fs_el)
    rho_h = FreeChargeDensity(bs.fs_h)
                
    rho_el.add_charge(bs.psi_el, 
                   np.array(bs.bands),
                   dk=dk,
                   mu=mu, 
                   temp=indata.temperature,
                   modified_EFA=indata.modified_EFA, 
                   particle=indata.particle_s_components,
                   norm_sum_region = bs.norm_sum_region,
                   thr_el=indata.thr_el,
                   thr_h=indata.thr_h)

    rho_h.add_charge(bs.psi_h, 
                   np.array(bs.bands),
                   dk=dk ,
                   mu=mu, 
                   temp=indata.temperature, 
                   modified_EFA=indata.modified_EFA,
                   particle=indata.particle_p_components,
                   norm_sum_region = bs.norm_sum_region,
                   thr_el=indata.thr_el,
                   thr_h=indata.thr_h)
    
    ntot_el, ptot_el = rho_el.get_total_charge() 
    ntot_h, ptot_h = rho_h.get_total_charge() 

    ntot = ntot_el + ntot_h
    ptot = ptot_el + ptot_h
    total_charge = ntot + ptot
    logger.logga(total_charge=total_charge)

    # save output and produce figures
    logger.write(value='Writing output.....')
    if rank==0:
        # (kz,  neig)
        np.save(outdata_path+'/bands', bs.bands)
        # (kz,  neig)
        np.save(outdata_path+'/spinor_dist', bs.spinor_distribution)
        # (kz, char, neig)
        np.save(outdata_path+'/norm_sum_region', bs.norm_sum_region)

        np.save(outdata_path+'/envelope_el', bs.psi_el)  
        np.save(outdata_path+'/envelope_h', bs.psi_h) 
        
        np.save(outdata_path+'/total_charge'+idj, total_charge)
        # save the overlap matrix
        save_npz(outdata_path+"/B.npz", bs.solver[0].bgl)

    # plot the results
    if rank==0:
        figure_bands = bs.plot_bands(**indata.plotting_preferencies_bands)
        figure_bands.savefig('./outdata/energy_bands.png',bbox_inches="tight")
    
        figure_density = bs.plot_density(rho_el,rho_h,**indata.plotting_preferencies_density)
        figure_density.savefig('./outdata/carrier_density.png',bbox_inches="tight")
    
        figure_potential = p.epot.plot(**indata.plotting_preferencies_potential)
        figure_potential.savefig('./outdata/potential.png',bbox_inches="tight")
    

    
if __name__ == '__main__':
    tic() 
    main()
    toc()
