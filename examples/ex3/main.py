"""
In this example we show how to compute the band structure of a core-shell nanowire self-consistently
"""

################# import section ######################
import sys
sys.path.append("../../")

import numpy as np
import copy
import pickle as pl
from mpi4py import MPI

# base
from nwkp import tic, toc
from nwkp import Logger

# input file
import indata

# interfaces
from nwkp.fem import Mesh
from nwkp import BandStructure, PoissonProblem
from nwkp import FreeChargeDensity, ElectrostaticPotential, Broyden
from nwkp import  _constants

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

# Display hostname andIP address
def host_IP(mylogger):
   try:
      hname = socket.gethostname()
      hip = socket.gethostbyname(hname)
      mylogger.write(key='Hostname: ', value=hname)
      mylogger.write(key="IP Address: ",value=hip)
   except:
      mylogger.write(value="Unable to get Hostname and IP")

############# Schrodinger-Poisson at fixed chemical potential  ############

def main():

    nreg = len(indata.reg2mat.keys())

    mesh = Mesh(
        mesh_name = indata.mesh_name,
        reg2mat = indata.reg2mat,
        mat2partic = indata.mat2partic,
        restrict_to = None,
        bandwidth_reduction = True
    )        
    
    # chemical potential
    chemical_potential = indata.chemical_potential

    # electric field values
    electric_field = indata.electric_field

    # kz values
    kzvals = indata.kzvals

    try:
        dk = kzvals[1] - kzvals[0]
    except IndexError:
        #logger.write(value='Warning: dk = 0.0')
        dk = 0.0

    nk = len(kzvals)
    if rank == 0:
        np.save(outdata_path+'/kzvals', kzvals)


    # compute a subset of kzvalues in each node
    kmaxlocal = len(kzvals) // size
    kin = rank * kmaxlocal
    kfin = kin + kmaxlocal
    kzslice = np.s_[kin:kfin]

    ############################################################
    ############ Loop in external electric fields ##############
    if rank==0:
        np.save(outdata_path+'/electric_field', electric_field)
    ii=0
    for ef in electric_field[indata.efield_slice]:
        id_ef_curr_out = indata.efield_slice[ii] 
        directory_ef = "OUT_"+str(id_ef_curr_out)
        parent_dir = outdata_path
        path_ef = os.path.join(parent_dir, directory_ef)
        if rank==0:
            if not os.path.exists(path_ef):
                os.mkdir(path_ef)
            np.save(path_ef+'/electric_field_value', ef[0])
            np.save(path_ef+'/electric_field_direction', ef[1])
        comm.Barrier()

        #########################################################
        ############ Loop on chemcial potentials ################
        if rank==0:
            np.save(path_ef+'/chemical_potential', chemical_potential)
        i=0
        init_pot_name = indata.init_pot_name
        for mu in chemical_potential:
            # create directory for output files
            idmu_curr_out = i
            directory_mu = "OUT_"+str(idmu_curr_out)
            parent_dir = path_ef
            path_mu = os.path.join(parent_dir, directory_mu)
            path_log = os.path.join(path_mu, "LOGFILES")
            if rank==0:
                if not os.path.exists(path_mu):
                    os.mkdir(path_mu)
                if not os.path.exists(path_log):
                    os.mkdir(path_log)
                np.save(path_mu+'/mu', mu)
            comm.Barrier()
            logger = Logger(rank=rank, logfile_path=path_log)
            host_IP(mylogger=logger)
            logger.logga(size=size)
            logger.write(key='Number of chemical potentials = ', value=len(chemical_potential))
            logger.logga(chemical_potential=mu)
      
            #############################################################################################################
            ############ Solve Poisson equation without free charge, including an external electric field ###############
            p = PoissonProblem(
                mesh, 
                shape_class_name=indata.shape_kind_poisson,
                dirichlet=indata.dirichlet, 
                electric_field=ef,
                user_defined_parameters=indata.user_defined_params
            )
    
            idj = '_init'
            
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
    
            if rank==0:
                np.save(path_mu+'/epot'+str(idj), Vin, allow_pickle=False)

            ###############################################
            ########### SELF CONSISTENT CYCLE #############
    
            # create broyden updater    
            up = Broyden(N=Vin.shape[0], M=indata.maxter, beta=indata.betamix, w0=indata.w0, use_wm=indata.use_wm)
    
            # resid 
            n_resid_lst = []
            n_resid_rel_lst = []
            p_resid_lst = []
            p_resid_rel_lst = []
    
            # self-consistent cycle starts
            for j in range(indata.maxiter):
                
                # create the external potential (on poisson fem space)
                # at the first iteration is the full potential, including the external field
                # at the second, third, ... iterations it is going to be relaxed using the broyden method
                epot = ElectrostaticPotential(p.fs, V=Vin)
        
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
                    epot = epot,
                    logger=logger,
                    rescaling=indata.rescaling,
                    user_defined_params=indata.user_defined_params
                )
                bs.run()
            
                ############# MPI SECTION ###########
            
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
                logger.write(value='Writing output.....')

                # save output and produce figures
                logger.write(value='Writing output.....')
                if rank==0:
                    # (kz,  neig)
                    np.save(path_mu+'/bands', bs.bands)
                    # (kz,  neig)
                    np.save(path_mu+'/spinor_dist', bs.spinor_distribution)
                    # (kz, char, neig)
                    np.save(path_mu+'/norm_sum_region', bs.norm_sum_region)
            
                    np.save(path_mu+'/envelope_el', bs.psi_el)  
                    np.save(path_mu+'/envelope_h', bs.psi_h) 
                    
                    np.save(path_mu+'/total_charge'+idj, total_charge)
                    # save the overlap matrix
                    save_npz(path_mu+"/B.npz", bs.solver[0].bgl)
                    
                # plot the results
                logger.write(value='Plotting the results........')
                if rank==0:
                    figure_bands = bs.plot_bands(**indata.plotting_preferencies_bands)
                    figure_bands.savefig(path_mu+'/energy_bands.png',bbox_inches="tight")

                    figure_density = bs.plot_density(rho_el,rho_h,**indata.plotting_preferencies_density)
                    figure_density.savefig(path_mu+'/carrier_density.png',bbox_inches="tight")

                    figure_potential = p.epot.plot(**indata.plotting_preferencies_potential)
                    figure_potential.savefig(path_mu+'/potential.png',bbox_inches="tight")
   
                ############ CONVERGENCE CHECK #############
                if j>0: 
                    # density convergence check
                    logger.logga(iteration_number = j)
                    logger.write(value='Convergence check.....')
                    
                    # total density change--> is defined as the int(abs(n1-n2))
                    n_resid, p_resid = get_density_resid(rho_el, rho_h, rho_el_prev, rho_h_prev)
    
                    if n_resid>1e-10: 
                        n_resid_rel = n_resid / np.abs(ntot_prev)
                    else:
                        n_resid_rel=0.0
                    if p_resid>1e-10:
                        p_resid_rel = p_resid / np.abs(ptot_prev)
                    else:
                        p_resid_rel=0.0
    
                    logger.write(key='Negative density resid [cm^-1]: ', value=n_resid)
                    logger.write(key='Negative density resid [relative]: ', value=n_resid_rel)
                    logger.write(key='Positive density resid [cm^-1]: ', value=p_resid)
                    logger.write(key='Positive density resid [relative]: ', value=p_resid_rel)
    
                    # save resid and relative resid
                    n_resid_lst.append(n_resid)
                    n_resid_rel_lst.append(n_resid_rel)
                    p_resid_lst.append(p_resid)
                    p_resid_rel_lst.append(p_resid_rel)
                    if rank==0:
                        np.save(path_mu+"/n_resid", np.array(n_resid_lst))
                        np.save(path_mu+"/n_resid_rel", np.array(n_resid_rel_lst))
                        np.save(path_mu+"/p_resid", np.array(p_resid_lst))
                        np.save(path_mu+"/p_resid_rel", np.array(p_resid_rel_lst))
    
                    if (n_resid_rel <= indata.maxchargeerror and p_resid_rel <= indata.maxchargeerror):
                        logger.write(value='CONVERGENCE REACHED')
                        # EXIT SP CYCLE
                        break                
                # actual dens becomes prev
                rho_el_prev = copy.copy(rho_el)
                rho_h_prev = copy.copy(rho_h)
                ntot_prev = ntot
                ptot_prev = ptot
                
                ############### POISSON ###################
                
                p = PoissonProblem(
                    mesh, 
                    shape_class_name=indata.shape_kind_poisson,
                    dirichlet=indata.dirichlet,
                    electric_field=ef, 
                    user_defined_parameters=indata.user_defined_params,
                    rho_el = rho_el,
                    rho_h = rho_h 
                )
                p.run()
                Vout = p.epot.V
                logger.logga(max_pot_variation = np.max( np.abs(Vin-Vout) ))
                # MAE potential
                mae_pot = np.sum(np.abs(Vin-Vout)) / Vin.shape[0]
                logger.write(key='Mean absolute error potential [eV]: ',value=mae_pot)
        
                # mixing procedure
                if j in indata.toreset:
                    reset = True
                else:
                    reset = False
                Vout = up.update( xin=Vin[:,np.newaxis], xout=Vout[:,np.newaxis], reset=reset ).squeeze()
                idj='_conv'
                Vin = np.copy(Vout)
                
                np.save(path_mu+'/epot'+str(idj), Vin)
                if j < indata.maxiter - 1 :
                    del(bs)
                    gc.collect()
                if j == indata.maxiter-1:
                    logger.write(value='REACHED MAX INTERATION NUMBER')
             
            # plot the results
            logger.write(value='Plotting the results........')
            if rank==0:
                figure_bands = bs.plot_bands(**indata.plotting_preferencies_bands)
                figure_bands.savefig(path_mu+'/energy_bands.png',bbox_inches="tight")
            
                figure_density = bs.plot_density(rho_el,rho_h,**indata.plotting_preferencies_density)
                figure_density.savefig(path_mu+'/carrier_density.png',bbox_inches="tight")
            
                figure_potential = p.epot.plot(**indata.plotting_preferencies_potential)
                figure_potential.savefig(path_mu+'/potential.png',bbox_inches="tight")
            logger.write(value='Going to next chemical potential........')
    
            # use the potential energy from the last chemical potential
            #init_pot_name = path_mu+"/epot_conv.npy"
    
            #idmu_curr_out+=1
            i+=1
        logger.write(value='All chemical potentials have been considered...')
        ii+=1
    logger.write(value='All electric fields have been considered...')


def get_density_resid(rho1_el, rho1_h, rho2_el, rho2_h):
    n_resid = 0.0
    p_resid = 0.0
    for iel in range(rho1_el.fs.mesh.nelem):
        fel_el = rho1_el.fs.felems[iel]
        fel_h = rho1_el.fs.felems[iel]
        gauss_coords_el = fel_el.gauss_coords
        gauss_coords_h = fel_h.gauss_coords
        n1_el, p1_el = rho1_el.interp(gauss_coords_el, total=False)
        n1_h, p1_h = rho1_h.interp(gauss_coords_h, total=False)
        n2_el, p2_el = rho2_el.interp(gauss_coords_el, total=False)
        n2_h, p2_h = rho2_h.interp(gauss_coords_h, total=False)
        n_resid += fel_el.int_f(np.abs(n1_el + n1_h - n2_el - n2_h)) * _constants.length_scale**2 * 1e-16
        p_resid += fel_el.int_f(np.abs(p1_el + p1_h - p2_el - p2_h)) * _constants.length_scale**2 * 1e-16

    return n_resid, p_resid
    
    
if __name__ == '__main__':
    tic() 
    main()
    toc()
