import copy
import numpy as np
from nwkpy import _constants
from nwkpy import  _database
from nwkpy._common import Logger
from nwkpy.fem import FemSpace, FemSpaceProduct
from nwkpy.hamiltonian import HamiltonianZB
from nwkpy.hamiltonian_wz import HamiltonianWZ
from nwkpy.physics import FreeChargeDensity
from nwkpy.fem.mesh.mesh import RectangularMesh
from nwkpy.fem.solver import GenEigenProblem
from nwkpy.fem.problem import Schrodinger

import matplotlib.pyplot as plt
from matplotlib import figure
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.tri import UniformTriRefiner
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mpi4py import MPI
import logging
import os

logger = logging.getLogger(__name__)
    
class BandStructure:
    def __init__(
        self,
        mesh,
        kzvals, 
        valence_band_edges, 
        principal_axis_direction='001',
        crystal_form='ZB',
        temperature=4.0,
        k=60, 
        shape_functions={
            'el': 'Hermite',
            'h' : 'LagrangeQuadratic',
        },
        epot=None, 
        logger=None,
        user_defined_params=None,
        rescaling=None,
        decouple_split_off=None,
        decouple_conduction=None,
        spherical_approximation=None,
        e_search=0.0,
        eigenvalue_shift=None
    ):
        
        # mesh object
        self.mesh = mesh

        # finite element support for the electron components
        self.fs_el = FemSpace(copy.copy(mesh), shape_class_name=shape_functions['el'])

        # finite element support for the hole components
        self.fs_h = FemSpace(copy.copy(mesh), shape_class_name=shape_functions['h'])

        # finite element support for the product of the two
        self.fsP = FemSpaceProduct(self.fs_el, self.fs_h)

        # free wave vectors 1/Angstrom.
        self.kzvals = kzvals 

        # temperature of the system
        self.temperature = temperature

        # direction of the z' axis along which the nanowire is translationally invariant
        self.principal_axis = principal_axis_direction

        # the energy value in eV at which the eigenvalue are sought.
        self.e_search = e_search 

        # electrostatic potential energy object in eV
        self.epot = epot
        
        # number of eigenvalues and eigenvectors to be sought
        self.k = k

        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(rank=0,logfile_path='./')
            
        material_list = list(self.mesh.reg2mat.values())
        # Ensure rescaling is a dict for all materials
        # GG 16-08-2025: aggiunto per evitare errore 'not unsubscriptable' piu' sotto
        if rescaling is None:
            rescaling = dict(zip(material_list, [None]*len(material_list)))
        if spherical_approximation is None:
            spherical_approximation = dict(zip(material_list, [None]*len(material_list)))
        if decouple_conduction is None:
            decouple_conduction = dict(zip(material_list, [None]*len(material_list)))        
        if decouple_split_off is None:
            decouple_split_off = dict(zip(material_list, [None]*len(material_list)))
            
        # bulk hamiltonian constructor of this structure
        if crystal_form=='ZB':
            Hamiltonian = HamiltonianZB
        elif crystal_form=='WZ':
            Hamiltonian = HamiltonianWZ
        else:
            raise ValueError("invalid crystal form")
        
        # for each material of the nanowire store an hamiltonian
        bulk_hamiltonian_list = []
        for material in material_list:
            bulk_hamiltonian = Hamiltonian(
                material=material,
                valence_band_edge=valence_band_edges[material],
                principal_axis_direction=principal_axis_direction,
                rescaling=rescaling[material],
                temperature=temperature,
                rembands=True,
                spherical_approx=spherical_approximation[material],
                decouple_split_off=decouple_split_off[material],
                decouple_conduction=decouple_conduction[material],
                user_defined_params=user_defined_params
            )
            bulk_hamiltonian_list.append(bulk_hamiltonian)

        # the set of hamiltonians is stored in the form of a dict.
        # keys are the material names, values are the hamiltonians
        self.bulk_hamiltonian_dict = dict(zip(list(self.mesh.reg2mat.values()), bulk_hamiltonian_list))

        # list for energy bands
        self.bands = []
        self.spinor_distribution = []
        
        # list for fem solutions
        self.femsol_lst = []

        # array with eigenvectors at each kz 
        self.psi = None
        
        # fermi occupation
        self.fermi = np.zeros(k)

        # eigenvalue shift preferencies
        # this must be a list with an "eigenvalue_shift" dictionary
        # containing the data for the eigenvalue shift at each wave-vector
        # if given, at each kz wave vector the data are read
        if eigenvalue_shift is not None:
            self.eigenvalue_shift = eigenvalue_shift
        else:
            self.eigenvalue_shift=[None]*len(self.kzvals)

        self.solver = []

    def run(self):

        psi_el = []
        psi_h = []
        bands = []
        spinor_distribution = []
        norm_sum_region_lst = []

        # create solver
        solver_kp = GenEigenProblem()
        
        i = 0 # index for k-cycle      

        # with MPI_debug = True, write debug information to file
        debug_write(f"Starting computation for {len(self.kzvals)} k-points \n")

        for kz in self.kzvals:
            
            # with MPI_debug = True, write debug information to file
            debug_write(f"Computing for kz={kz}\n") 
            
            # k dot p variational form
            v = Schrodinger(
                kz = kz,
                hamiltonian_dict = self.bulk_hamiltonian_dict,
                epot=self.epot,
            )

            # assembly global matrices
            solver_kp = GenEigenProblem()
            solver_kp.assembly(self.fsP, v)
            

            # self.logger.logga(search_energy = self.e_search) ##########################
            # logging.info(f'search energy = {self.e_search}') # log the search energy

            eigvals, eigvecs_el, eigvecs_h, comp_dist, norm_sum_region = solver_kp.solve(
                    k=self.k,
                    which='LM',
                    v0=None,
                    sigma=self.e_search,
                    eigenvalue_shift=self.eigenvalue_shift[i]
                )

            # with MPI_debug = True, write debug information to file
            debug_write(f"Eigenvalues = {eigvals * 1e3}\n") 

            # store current solution
            bands.append(list(eigvals))
            spinor_distribution.append(list(comp_dist))
            norm_sum_region_lst.append(list(norm_sum_region))

            # complete envelope function
            psi_el.append(eigvecs_el)
            psi_h.append(eigvecs_h)
            self.solver.append(solver_kp)            

            i+=1

        self.psi_el = np.array(psi_el)
        self.psi_h = np.array(psi_h)
        self.bands = np.array(bands)
        self.spinor_distribution = np.array(spinor_distribution)
        self.norm_sum_region = np.array(norm_sum_region_lst)

    #######################################################################
        

    # methods for plotting

    ############### PLOT ENERGY BANDS ###############
    def plot_bands(
        self,
        threshold_el=None, 
        threshold_h=None, 
        chemical_potential=None, 
        character_to_show=None, 
        figsize=(5, 5), 
        lw=4,
        xlim=None, 
        ylim=None, 
        cmap_in='Blues',
        loc_cbar=1,
        spines_lw=4,
        fontsize=20):

        fig = figure.Figure( figsize =figsize )
        gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0,0])
        
        title=''
        # spinorial characters
        EL = self.spinor_distribution[:,0:2,:].sum(axis=1)
        HH = self.spinor_distribution[:,2:4,:].sum(axis=1)
        LH = self.spinor_distribution[:,4:6,:].sum(axis=1)
        SO = self.spinor_distribution[:,6:8,:].sum(axis=1)

        norm5 = plt.Normalize(0., 1.)
        if character_to_show is not None:
            if character_to_show=='EL':
                color = EL
                cmap=cmap_in
                ticklabels = ['0', '1']
                ticks=[0,1]
            elif character_to_show=='HH':
                color= HH
                cmap=cmap_in
                ticklabels = ['0', '1']
                ticks=[0,1]
            elif character_to_show=='LH':
                color= LH
                cmap=cmap_in
                ticklabels = ['0', '1']
                ticks=[0,1]
            elif character_to_show=='SO':
                color= SO
                cmap=cmap_in
                ticklabels = ['0', '1']
                ticks=[0,1]
            elif character_to_show=='LH-HH':
                color= LH - HH
                cmap=cmap_in
                ticklabels = ['HH', 'LH']
                ticks=[-0.9,0.9]
                norm5 = plt.Normalize(-1., 1.0)
            elif character_to_show=='H-EL':
                color= (HH+LH+SO)-EL
                cmap=cmap_in
                ticklabels = ['EL', 'H']
                ticks=[-1,1]
                norm5 = plt.Normalize(-1., 1.0)
            elif character_to_show=='H-EL-reg':
                if self.norm_sum_region is None:
                    raise ValueError("Must provide norm per region vector")
                color= self.norm_sum_region[:,1,:] - self.norm_sum_region[:,0,:]
                cmap=cmap_in
                ticklabels = ['EL', 'H']
                ticks=[-1,1]
                norm5 = plt.Normalize(-1., 1.0)  
            elif character_to_show=='H-EL-reg-threshold':
                if self.norm_sum_region is None:
                    raise ValueError("Must provide norm per region vector")
                if threshold_el is None:
                    raise ValueError("Must provide electron threshold")
                if threshold_h is None:
                    raise ValueError("Must provide hole threshold")
                color = np.zeros((self.norm_sum_region.shape[0],self.norm_sum_region.shape[2]))
                color[self.norm_sum_region[:,1,:]>threshold_h] = 1.0 #pure holes
                color[self.norm_sum_region[:,0,:]>threshold_el] = -1.0 #pure electrons
                cmap='jet'
                ticklabels = ['EL', 'H']
                ticks=[-1,1]
                norm5 = plt.Normalize(-1., 1.0) 
            else:
                raise ValueError("invalid character string")
        else:
            ticks=[-1,1]
            ticklabels = ["0", "1"]
            color = EL+HH+LH+SO
            cmap=cmap_in
    
        xlim_l = xlim[0]
        xlim_r = xlim[1]
        ylim_l = ylim[0]
        ylim_r = ylim[1]
        ax.set_xlim(xlim_l,xlim_r)
        ax.set_ylim(ylim_l,ylim_r) 
    
        ax.set_ylabel('$E$ [meV]', size = fontsize+7)
        ax.tick_params('y', labelsize=fontsize+3)
        ax.tick_params('x', labelsize=fontsize+3)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(spines_lw)
            ax.spines[axis].set_color("black")
        ax.tick_params(direction='out', length=8, width=3, colors='black',
               grid_color='black', grid_alpha=1.)

        for j in range(0,self.spinor_distribution.shape[2],1):
                
            x = self.kzvals * 10 # plot in nanometers
            y = self.bands[:,j] * 1e3 # plot in meV
            
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1) 
            lc = LineCollection(segments, cmap=cmap, norm=norm5)
            lc.set_array(color[:,j])
            lc.set_linewidth(lw)
            line5 = ax.add_collection(lc)
            
        cbaxes = inset_axes(ax, width="40%", height="6%", loc=loc_cbar) 
        
        cbar = fig.colorbar(line5,
                     cax=cbaxes,
                     ticks=ticks,
                     orientation='horizontal')
        
        cbar.ax.set_xticklabels(ticklabels,size=15)
        cbar.ax.tick_params(size=0)
        #cbar.ax.set_title(title, fontsize=20, loc="right")
        cbar.ax.set_xlabel(xlabel=title, fontsize=fontsize)

        if character_to_show is None:
            cbar.remove()
    
        # plot chemical potentials
        if chemical_potential is not None:
            ax.axhline(y=chemical_potential, color='black', ls='--', lw=lw)
        
        ax.set_xlabel('$k_{z}[nm^{-1}]$',size=fontsize+7)
        return fig
    
    ############### PLOT CARRIER DENSITY ###############

    def plot_density(self, *density, xlim, ylim, figsize=(5,5), subdiv=1, cmapin='rainbow', levels=21, fontsize=20, polygons=None):
    
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1, left=0.00, right=1, top=1.0, bottom=0.0, hspace=0.0) 
        ax = fig.add_subplot(gs[0,0])

        xlim_l = xlim[0]
        xlim_r = xlim[1]
        ylim_l = ylim[0]
        ylim_r = ylim[1]

    
        triangles = self.mesh.triangles[:,:3]
        x = self.mesh.vertices[:,0] #*staticdata.br / 10 # nm
        y = self.mesh.vertices[:,1] #*staticdata.br / 10 # nm
        trigrid = tri.Triangulation(x, y, triangles )
        ref_trigrid = UniformTriRefiner(trigrid)
        trigrid = ref_trigrid.refine_triangulation(subdiv=subdiv)
        vert = np.vstack([trigrid.x, trigrid.y]).T
        triangles = trigrid.triangles
        
        ch = np.zeros(len(vert))
        for rho in density:
            ch += rho.interp(vert) / 1e16
    
        ax.tick_params('both', labelsize=15) 
        
        ch2D = ax.tricontourf(
            vert[:,0] * _constants.length_scale / 10, 
            vert[:,1] * _constants.length_scale / 10,
            triangles,
            ch,
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
        cbar.ax.set_ylabel(r'Carrier density $[10^{16} \mathrm{cm}^{-3}]$', fontsize=fontsize+2)
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

_out_directory = None

def MPI_debug_setup(valore):
    """
    Receive and set up the output directory for MPI debug files.
    
    Args:
        valore (str): The directory path where debug files will be written.
    """
    global _out_directory
    _out_directory = valore
    os.makedirs(_out_directory, exist_ok=True)

def debug_write(message, rank=None):
    """
    Funzione semplice per scrivere debug su file separati per ogni processo MPI.
    
    Args:
        message (str): Messaggio da scrivere
        rank (int, optional): Rank MPI (se None, viene determinato automaticamente)
    """
    
    import os
 
    # Import the flag from the calling script
    try:
        import __main__
        MPI_debug = getattr(__main__, 'MPI_debug', False)
    except:
        MPI_debug = False
    
    if not MPI_debug:
        return
    
    if rank is None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    
    filename = os.path.join(_out_directory, f"MPI_debug_{rank}.txt")
      
    # # Aggiunge timestamp al messaggio
    # import datetime
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # formatted_message = f"[{timestamp}] RANK-{rank}: {message}\n"
    
    # Scrive su file (append mode)
    with open(filename, 'a') as f:
        # f.write(formatted_message)
        f.write(message)