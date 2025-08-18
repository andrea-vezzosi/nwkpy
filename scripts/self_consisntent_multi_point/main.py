#!/usr/bin/env python
"""
Self-consistent Schrödinger-Poisson band structure calculation for core-shell nanowires

This script performs multi-parameter sweeps over chemical potentials and external electric fields,
solving the coupled Schrödinger-Poisson equations self-consistently using the 8-band k·p method
with finite element discretization.

Key features:
- Multi-physics coupling between electronic structure and electrostatics
- Broyden mixing for accelerated convergence
- MPI parallelization over k-points
- Comprehensive output including band structures, charge densities, and potentials
"""

# =============================================================================
# GENERAL IMPORTS AND SETUP
# =============================================================================

# Standard library imports for system operations
import sys                              # System-specific parameters and functions
import os                               # Operating system interface for file operations

# Numerical computing libraries
import numpy as np                      # Fundamental numerical computing package
import copy                             # Deep copy operations for object duplication

# Message Passing Interface for parallel computing
from mpi4py import MPI                  # Python bindings for MPI parallelization

# Input/output and logging utilities
import logging                          # Logging library for structured output control

# Scientific computing utilities
from scipy.sparse import save_npz       # Sparse matrix I/O operations for efficiency

import socket                           # Network info for debugging
import gc                               # Garbage collection for memory management

# =============================================================================
# CORE LIBRARY IMPORTS
# =============================================================================

# High-level timing and diagnostic utilities from nwkpy
from nwkpy import tic, toc              # High-resolution timing functions for performance analysis
from nwkpy import library_header        # Display library version and build information

# Core computational classes for nanowire band structure calculations
from nwkpy.fem import Mesh              # Finite element mesh handling and region assignment
from nwkpy import BandStructure         # k·p band structure solver with MPI parallelization
from nwkpy import PoissonProblem        # Electrostatic potential solver via Poisson equation
from nwkpy import FreeChargeDensity     # Free carrier density calculator with Fermi statistics
from nwkpy import ElectrostaticPotential # Potential field container and manipulation
from nwkpy import Broyden               # Broyden mixing for accelerated convergence
from nwkpy import _constants            # Physical constants (fundamental and material-specific)
from nwkpy import MPI_debug_setup        # MPI debugging utility function

# Material parameter database access
from nwkpy._database import params      # Comprehensive material parameter database

# =============================================================================
# LOCAL IMPORTS 
# =============================================================================

# Import local utility functions for logging and error handling
from nwkpy.utilities import *

# Import local configuration constants for consistent formatting
from nwkpy.config import *

# =============================================================================
# SCRIPT PARAMETERS
# =============================================================================

# This script identifier for logging and header generation
SCRIPT_NAME = 'Self-consistent Schrödinger-Poisson band structure calculation'  

# =============================================================================
# IMPORT INPUT PARAMETERS
# =============================================================================

# Import all simulation parameters from the input configuration file
from indata import *                   

# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================

# Construct file system paths for output data storage
cdir = os.getcwd()                     # Current working directory (script location)

# Construct output data directory path
outdata_path = os.path.join(cdir, directory_name)

# Construct full path to log file for structured output
log_file = os.path.join(outdata_path, LOG_FILE_NAME + ".log")

# =============================================================================
# MPI SETUP AND PROCESS INITIALIZATION
# =============================================================================

# Initialize MPI communicator for parallel k-point calculations
comm = MPI.COMM_WORLD                  # Global communicator including all MPI processes
rank = comm.Get_rank()                 # Process rank (0 to size-1), 0 is the master process
size = comm.Get_size()                 # Total number of MPI processes in the calculation

# Creates the directory tree if it doesn't exist, no error if it already exists
# Only rank 0 creates directories to avoid race conditions
if rank == 0:
    os.makedirs(outdata_path, exist_ok=True)

# Synchronize all processes after directory creation
comm.Barrier()

# =============================================================================
# MESH FILES
# =============================================================================

# Construct paths to required mesh files (input data)
mesh_file = os.path.join(cdir, mesh_name + ".msh")  # GMSH mesh geometry file
mesh_data = os.path.join(cdir, mesh_name + ".dat")  # Mesh metadata and region definitions

# Verify that required mesh files exist before proceeding
# Only rank 0 checks files to avoid race conditions
if rank == 0:
    try:
        if not os.path.exists(mesh_file):  # Check for primary mesh file
            raise FileNotFoundError(f"Mesh file '{mesh_file}' not found")
        if not os.path.exists(mesh_data):  # Check for mesh metadata file
            raise FileNotFoundError(f"Mesh data file '{mesh_data}' not found")
        file_check_passed = True
    except FileNotFoundError as f:
        execution_aborted(str(f))

# =============================================================================
# CONFIGURE LOGGING SYSTEM
# =============================================================================

# NOTE: This logging configuration must come after nwkpy imports to avoid 
# duplicate library headers in the log output
logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',  # Timestamp and level for each message
    filename=log_file,                 # Direct all log output to file (not console)
    datefmt='%H:%M:%S',                # Time format: hours:minutes:seconds (no date)
    filemode='w',                      # Overwrite existing log file on each run
    level=logging.INFO                 # Minimum logging level (INFO and above)
)
logger = logging.getLogger(__name__)   # Get logger instance for this module

# Display log file location on stdout for user reference
if rank == 0:
    print(f'\nAll log messages sent to file: {log_file}\n')

# Synchronize all processes after logging setup
comm.Barrier()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_density_resid(rho1_el, rho1_h, rho2_el, rho2_h):
    """
    Calculate charge density residuals for convergence check
    
    Computes the integrated absolute difference between current and previous
    charge densities for both electrons and holes. This is used to determine
    when the self-consistent cycle has converged.
    
    Parameters:
        rho1_el, rho1_h: Current iteration charge densities
        rho2_el, rho2_h: Previous iteration charge densities
        
    Returns:
        n_resid: Negative charge density residual in cm⁻¹
        p_resid: Positive charge density residual in cm⁻¹
    """
    n_resid = 0.0
    p_resid = 0.0
    
    # Loop over all finite elements in the mesh
    for iel in range(rho1_el.fs.mesh.nelem):
        # Get finite element objects for electrons and holes
        fel_el = rho1_el.fs.felems[iel]
        fel_h = rho1_el.fs.felems[iel]
        
        # Get Gaussian quadrature coordinates
        gauss_coords_el = fel_el.gauss_coords
        gauss_coords_h = fel_h.gauss_coords
        
        # Interpolate charge densities at Gauss points
        n1_el, p1_el = rho1_el.interp(gauss_coords_el, total=False)
        n1_h, p1_h = rho1_h.interp(gauss_coords_h, total=False)
        n2_el, p2_el = rho2_el.interp(gauss_coords_el, total=False)
        n2_h, p2_h = rho2_h.interp(gauss_coords_h, total=False)
        
        # Integrate absolute differences with proper units conversion
        # Length scale conversion and cm⁻¹ units
        n_resid += fel_el.int_f(np.abs(n1_el + n1_h - n2_el - n2_h)) * _constants.length_scale**2 * 1e-16
        p_resid += fel_el.int_f(np.abs(p1_el + p1_h - p2_el - p2_h)) * _constants.length_scale**2 * 1e-16

    return n_resid, p_resid

def consistency_checks():
    """
    Perform comprehensive consistency checks on input parameters from indata_updated.py.
    Raises:
        ValueError: If any parameter is invalid, inconsistent, or out of range.
    """
    # Check output directory and mesh name
    if not isinstance(directory_name, str) or not directory_name:
        logger.error(f"You entered directory_name = {directory_name}")
        raise ValueError("directory_name must be a non-empty string.")
    if not isinstance(mesh_name, str) or not mesh_name:
        logger.error(f"You entered mesh_name = {mesh_name}")
        raise ValueError("mesh_name must be a non-empty string.")

    # Check material list
    if not (isinstance(material, list) and len(material) == 2):
        logger.error(f"You entered material = {material}")
        raise ValueError("material must be a list of two material names (core, shell).")
    if any(m not in params for m in material):
        logger.error(f"You entered material = {material}")
        logger.error(f"One material in {material} not available")
        logger.error(f'Available materials: {list(params.keys())}')
        raise ValueError(f"Unavailable material(s) - Choose an available material or update the parameters database")

    # Check valence_band
    if not (isinstance(valence_band, list) and len(valence_band) == 2):
        logger.error(f"You entered valence_band = {valence_band}")
        raise ValueError("valence_band must be a list of two values (core, shell).")

    # Check principal_axis_direction
    if principal_axis_direction not in PRINCIPAL_AXIS_DIRECTIONS:
        logger.error(f"You entered principal_axis_direction = {principal_axis_direction}")
        raise ValueError(f"principal_axis_direction is not valid.")

    # Check lattice_constant
    if not (isinstance(lattice_constant, (float, int)) and lattice_constant > 0):
        logger.error(f"You entered lattice_constant = {lattice_constant}")
        raise ValueError("lattice_constant must be a positive number.")

    # Check temperature
    if not (isinstance(temperature, (float, int)) and temperature > 0):
        logger.error(f"You entered temperature = {temperature}")
        raise ValueError("temperature must be a positive number.")

    # Check carrier types
    if not (isinstance(carrier, list) and len(carrier) == 2):
        logger.error(f"You entered carrier = {carrier}")
        raise ValueError("carrier must be a list of two values (core, shell).")

    if any(c not in CARRIER_TYPES for c in carrier):
        logger.error(f"You entered carrier = {carrier}")
        raise ValueError(f"carrier must be in {CARRIER_TYPES}.")

    # Check chemical_potential_set
    if not (isinstance(chemical_potential_set, list) and all(isinstance(mu, (float, int)) for mu in chemical_potential_set)):
        logger.error(f"You entered chemical_potential_set = {chemical_potential_set}")
        raise ValueError("chemical_potential_set must be a list of numbers.")

    # Check electric_field_set
    if not (isinstance(electric_field_set, list) and all(isinstance(t, tuple) and len(t) == 2 for t in electric_field_set)):
        logger.error(f"You entered electric_field_set = {electric_field_set}")
        raise ValueError("electric_field_set must be a list of (magnitude, angle) tuples.")

    # Check e_search
    if not isinstance(e_search, (float, int)):
        logger.error(f"You entered e_search = {e_search}")
        raise ValueError("e_search must be a number.")

    # Check number_eigenvalues
    if not (isinstance(number_eigenvalues, int) and number_eigenvalues > 0):
        logger.error(f"You entered number_eigenvalues = {number_eigenvalues}")
        raise ValueError("number_eigenvalues must be a positive integer.")

    # Check k_range
    if not (isinstance(k_range, list) and len(k_range) == 2 and k_range[0] < k_range[1]):
        logger.error(f"You entered k_range = {k_range}")
        raise ValueError("k_range must be a list of two numbers [start, end] with start < end.")

    # Check number_k_pts
    if not (isinstance(number_k_pts, int) and number_k_pts > 0):
        logger.error(f"You entered number_k_pts = {number_k_pts}")
        raise ValueError("number_k_pts must be a positive integer.")

    # Check maxiter
    if not (isinstance(maxiter, int) and maxiter > 0):
        logger.error(f"You entered maxiter = {maxiter}")
        raise ValueError("maxiter must be a positive integer.")

    # Check maxchargeerror and maxchargeerror_dk
    if not (0 < maxchargeerror < 1):
        logger.error(f"You entered maxchargeerror = {maxchargeerror}")
        raise ValueError("maxchargeerror must be between 0 and 1.")
    if not (0 < maxchargeerror_dk < 1):
        logger.error(f"You entered maxchargeerror_dk = {maxchargeerror_dk}")
        raise ValueError("maxchargeerror_dk must be between 0 and 1.")

    # Check rescale
    if not (isinstance(rescale, list) and len(rescale) == 2):
        logger.error(f"You entered rescale = {rescale}")
        raise ValueError("rescale must be a list of two values (core, shell).")

    # Check character_threshold
    if not (isinstance(character_threshold, list) and len(character_threshold) == 2):
        logger.error(f"You entered character_threshold = {character_threshold}")
        raise ValueError("character_threshold must be a list of two values [electron, hole].")
    if any(not (0.0 <= x <= 1.0) for x in character_threshold):
        logger.error(f"You entered character_threshold = {character_threshold}")
        raise ValueError("Each character_threshold value must be in [0.0, 1.0].")

    # Check shape_function_kp
    if not (isinstance(shape_function_kp, list) and len(shape_function_kp) == 2):
        logger.error(f"You entered shape_function_kp = {shape_function_kp}")
        raise ValueError("shape_function_kp must be a list of two values.")
    if any(s not in SHAPE_FUNCTION_TYPES for s in shape_function_kp):
        logger.error(f"You entered shape_function_kp = {shape_function_kp}")
        raise ValueError(f"shape_function_kp values must be in {SHAPE_FUNCTIONS_TYPES}.")

    # Check shape_function_poisson
    if shape_function_poisson not in SHAPE_FUNCTION_TYPES:
        logger.error(f"You entered shape_function_poisson = {shape_function_poisson}")
        raise ValueError(f"shape_function_poisson must be in {SHAPE_FUNCTION_TYPES}.")

    # Check betamix, maxter, w0
    if not (0 < betamix < 1):
        logger.error(f"You entered betamix = {betamix}")
        raise ValueError("betamix must be between 0 and 1.")
    if not (isinstance(maxter, int) and maxter > 0):
        logger.error(f"You entered maxter = {maxter}")
        raise ValueError("maxter must be a positive integer.")
    if not (0 < w0 < 1):
        logger.error(f"You entered w0 = {w0}")
        raise ValueError("w0 must be between 0 and 1.")

    # Check dirichlet boundary conditions
    if not isinstance(dirichlet, dict):
        logger.error(f"You entered dirichlet = {dirichlet}")
        raise ValueError("dirichlet must be a dictionary.")

    # If all checks pass
    return True

def plot_production(bs,p,rho_el,rho_h,path_mu):

    """ Gemerate dofferemt plots according to input settings in png format"""
    
    # Band structure plot
    figure_bands = bs.plot_bands(**plotting_preferencies_bands)
    figure_bands.savefig(path_mu+'/energy_bands.png', bbox_inches="tight")

    # Carrier density distribution plot
    figure_density = bs.plot_density(rho_el, rho_h, **plotting_preferencies_density)
    figure_density.savefig(path_mu+'/carrier_density.png', bbox_inches="tight")

    # Electrostatic potential plot
    figure_potential = p.epot.plot(**plotting_preferencies_potential)
    figure_potential.savefig(path_mu+'/potential.png', bbox_inches="tight")

# =============================================================================
# MAIN CALCULATION FUNCTION
# =============================================================================

def main():
    """
    Main self-consistent Schrödinger-Poisson calculation routine
    
    This function orchestrates the entire calculation, including:
    1. System setup and parameter validation
    2. Mesh initialization and k-point distribution
    3. Multi-parameter sweeps over electric fields and chemical potentials
    4. Self-consistent solution of coupled Schrödinger-Poisson equations
    5. Output generation and convergence monitoring
    """
    
    # =============================================================================
    # HEADER AND INITIALIZATION
    # =============================================================================
    
    if rank == 0:
        
        # Display formatted script header with execution information
        print_header(SCRIPT_NAME)
        
        # Display nwkpy library version and build information
        library_header()
    
        # Perform comprehensive validation of all input parameters    
        try:                              # Attempt parameter validation
            consistency_checks()  
        except ValueError as e:           # Handle validation failures gracefully
            execution_aborted(e)          # Log error and terminate with proper cleanup
        else:                             # Validation successful - proceed with calculation
            logger.info("")  
            logger.info(f'Input parameters consistency checks passed')       
        
    # Synchronize after header display
    comm.Barrier()
    
    # =========================================================================
    # PHYSICAL SYSTEM CONFIGURATION
    # =========================================================================
    
    # Map finite element mesh regions to physical materials
    # Typically: Region 1 = core, Region 2 = shell for core-shell nanowires
    reg2mat = {
        1: material[0],    # Core region gets first material 
        2: material[1]     # Shell region gets second material 
    }

    # Define valence band edge offsets for proper band alignment
    # These values set the energy reference for each material in eV
    # Critical for correct band lineup in heterostructures
    valence_band_edges = {
        material[0]: valence_band[0],    # Core material VB edge
        material[1]: valence_band[1]     # Shell material VB edge
    }

    # Spurious solution suppression parameters for 8-band k·p model
    # The P parameter rescaling is required to avoid unphysical solutions
    # See: B. A. Foreman, Elimination of spurious solutions from eight-band k·p theory,
    # Phys. Rev. B 56, R12748 (1997)
    #
    # Options for rescaling parameter:
    # - 'S=0': Use Eq. 6.158 from Birner's reference
    # - 'S=1': Use Eq. 6.159 from Birner's reference  
    # - Numerical value: Fraction to reduce Ep by (e.g., 0.26 = 26% reduction)
    rescaling = {
        material[0]: rescale[0],
        material[1]: rescale[1]
    }

    # Material-to-carrier type mapping (legacy parameter, may be unused)
    mat2partic = {
        material[0]: carrier[0],    # Core material carrier type
        material[1]: carrier[1]     # Shell material carrier type
    }

    # =========================================================================
    # K-SPACE SAMPLING CONFIGURATION
    # =========================================================================

    # Configure k-space sampling along the nanowire axis (z-direction)
    # Only positive k-values are needed due to inversion symmetry
    kzin_rel = k_range[0]     # Initial relative k_z value
    kzfin_rel = k_range[1]    # Final relative k_z value  
    numkz = number_k_pts      # Number of k-points for sampling

    # Convert relative k-values to absolute values in Å⁻¹
    # Scaling factor: π/√3/a where a is the lattice constant
    # This gives the correct k-space sampling for the Brillouin zone
    kzvals = np.linspace(kzin_rel, kzfin_rel, numkz) * np.pi / np.sqrt(3.0) / lattice_constant

    # =========================================================================
    # NUMERICAL SOLVER CONFIGURATION
    # =========================================================================

    # Select shape functions for different equation types
    # SAFE (Schrödinger Approximation for Finite Elements) method
    shape_kind_kp = {
        'el': shape_function_kp[0],    # Electrons: typically 'Hermite' for high accuracy
        'h': shape_function_kp[1]      # Holes: typically 'LagrangeQuadratic' for efficiency
    }
    shape_kind_poisson = shape_function_poisson  # Shape functions for Poisson equation

    # Number of eigenvalues to compute (number of subbands to resolve)
    k = number_eigenvalues

    # =========================================================================
    # CHARGE CHARACTERIZATION PARAMETERS
    # =========================================================================

    # Threshold values for classifying states as electron-like or hole-like
    # Based on the dominant character of the wavefunction
    thr_el = character_threshold[0]   # Electron character threshold
    thr_h = character_threshold[1]    # Hole character threshold

    # Legacy parameters for orbital components (may be unused in current version)
    particle_s_components = 'electron'
    particle_p_components = 'electron'

    # =========================================================================
    # MATERIAL PARAMETER DATABASE ACCESS
    # =========================================================================
        
    # Create user-defined material parameter dictionary for computational classes
    # user_defined_params = {
    #     material[0]: params[material[0]], # Core material parameters from database
    #     material[1]: params[material[1]]  # Shell material parameters from database
    # }
    user_defined_params = get_parameters(material)
    
    # Log material parameters for each region
    if rank == 0:
        for m in material:
            log_material_params(m, user_defined_params[m])
            
    # =========================================================================
    # FINITE ELEMENT MESH INITIALIZATION
    # =========================================================================
    
    nreg = len(reg2mat.keys())

    # Load and initialize the finite element mesh
    mesh = Mesh(
        mesh_name=mesh_file,
        reg2mat=reg2mat,
        mat2partic=mat2partic,
        restrict_to=None,
        bandwidth_reduction=True  # Optimize matrix bandwidth for efficiency
    )        
    
    # the following mesh studff could be extracted to a funcion in utilities and used in other scripts
    
    # Log mesh statistics for verification and debugging
    if rank == 0:
        logger.info("")
        logger.info(f'Mesh loaded successfully from {mesh_name}.msh')
        logger.info(f"{DLM}Total vertices (nodes) : {mesh.ng_nodes}")
        logger.info(f"{DLM}Total elements         : {mesh.nelem}")
        logger.info(f"{DLM}Boundary edges         : {len(mesh.e_l)}")
    
    # Display mesh metadata from accompanying .dat file
    if rank == 0:
        logger.info("")
        logger.info(f'Mesh metadata from {mesh_name}.dat')
        with open(mesh_data, "r") as f:
            content = f.read()
            for line in content.splitlines():
                logger.info(f'{DLM} {line}')

    # Synchronize after mesh initialization and logging
    comm.Barrier()

    # =========================================================================
    # K-POINT DISTRIBUTION FOR MPI PARALLELIZATION
    # =========================================================================
    
    # Calculate k-point spacing for numerical integration
    try:
        dk = kzvals[1] - kzvals[0]
    except IndexError:
        # Handle single k-point case
        dk = 0.0

    nk = len(kzvals)
    
    # Save k-point array (only rank 0 writes to avoid conflicts)
    if rank == 0:
        np.save(outdata_path+'/kzvals', kzvals)
        if generate_txt_files:
            np.savetxt(outdata_path + '/kzvals.txt', kzvals, fmt='%.5g', delimiter=DLM)

    # Distribute k-points across MPI processes for parallel computation
    # Each process handles a subset of k-points
    kmaxlocal = len(kzvals) // size
    kin = rank * kmaxlocal
    kfin = kin + kmaxlocal
    kzslice = np.s_[kin:kfin]

    try:
        if size*kmaxlocal != number_k_pts:
            logger.error(f"the number of k points {number_k_pts} not a multple of the number of process {size}")
            raise ValueError("Inconsitent number of processes")
    except ValueError as e:           # Handle validation failures gracefully
        execution_aborted(e)          # Log error and terminate with proper cleanup
           
    # =========================================================================
    # PARAMETER SWEEP SETUP
    # =========================================================================
    
    # Convert parameter lists to numpy arrays for easier handling
    electric_field = np.array(electric_field_set)
    chemical_potential = np.array(chemical_potential_set)

    # Save parameter arrays for later reference
    if rank == 0:
        np.save(outdata_path+'/electric_field', electric_field)
    
    # Synchronize after parameter setup
    comm.Barrier()
    
    # =========================================================================
    # LOG IMPUT PARAMETERS
    # =========================================================================
    
    # Log computational system and MPI configuration
    if rank == 0:
        # Display system configuration in tabular format
        log_system_configuration_summary(material,valence_band,rescale,
                        carrier,shape_function_kp,character_threshold)    
        # Log physical parameters for the simulation
        log_physical_parameters(temperature,chemical_potential_set,principal_axis_direction)
        # Log electrostatic potential solver configuration
        log_electrostatic_potential_configuration(shape_function_poisson,
                        dirichlet,electric_field_set,init_pot_name)
        # Log band structure calculation parameters
        log_band_structure_calculation_parameters(number_eigenvalues,e_search,kzvals,number_k_pts)
        # Log computational system and MPI configuration
        log_computational_system_information(size,kmaxlocal,MPI_debug)
        # Log self-consistent cycle parameters"
        log_self_consistent_cycle_parameters(maxiter, maxchargeerror)
        # Log Broyden mixing parameters
        log_broyden_mixing_parameters(betamix, maxter, w0, use_wm, toreset)

        # the following stuff could be extracted as a utility function
        if rank == 0:
            logger.info("")
            logger.info("File management")
            logger.info(f"{DLM}Current directory         : {cdir}")
            logger.info(f"{DLM}Output root directory     : {directory_name}")
            logger.info(f"{DLM}Text data file generation : {generate_txt_files}")
            logger.info(f"{DLM}Graph generation          : {generate_png_graphs}")
        
    # Synchronize after logging parameters
    comm.Barrier()
        
    # =========================================================================
    # OUTER LOOP: ELECTRIC FIELD SWEEP
    # =========================================================================

    # Log the start of the multi-parameter sweep (if more than one parameter set is entered)
    if rank == 0 and (len(electric_field) > 1 or len(chemical_potential) > 1):
        logger.info("")
        logger.info("" + "-"*80)
        logger.info("Starting multi-parameter sweep over electric fields and chemical potentials")
        logger.info("" + "-"*80)

    # Electric field sweep
    for ii, ef in enumerate(electric_field):
        
        # Create directory structure for this electric field value
        directory_ef = "OUT_EF_" + str(ii)
        parent_dir = outdata_path
        path_ef = os.path.join(parent_dir, directory_ef)
        
        if rank == 0:
            # create output directory
            os.makedirs(path_ef, exist_ok=True)
            # Save electric field parameters
            np.save(path_ef+'/electric_field_value', ef[0])
            np.save(path_ef+'/electric_field_direction', ef[1])
        
        # Synchronize all processes before proceeding
        comm.Barrier()

        # =====================================================================
        # INNER LOOP: CHEMICAL POTENTIAL SWEEP
        # =====================================================================
        
        if rank == 0:
            np.save(path_ef+'/chemical_potential', chemical_potential)

        # Chemical potential sweep
        for i, mu in enumerate(chemical_potential):
            
            # Create directory structure for this chemical potential
            directory_mu = "OUT_CP_" + str(i)
            parent_dir = path_ef
            path_mu = os.path.join(parent_dir, directory_mu)
            
            if rank == 0:
                os.makedirs(path_mu, exist_ok=True)
                np.save(path_mu+'/mu', mu)
            if MPI_debug:                           # to all ranks!
                MPI_debug_setup(path_mu+'/DEBUG')   # create DEBUG directory

            # Synchronize before all processes proceed
            comm.Barrier()
            
            # Log iteration information using standard logging
            if rank == 0:
                logger.info("")
                logger.info("Current configuration instance")
                logger.info(f'{DLM}Electric field {ii+1}/{len(electric_field_set)}      : {ef[0]} kV/m at {ef[1]:.3f} rad')
                logger.info(f'{DLM}Chemical potential {i+1}/{len(chemical_potential_set)}  : {mu:<.5f} eV')
                logger.info(f'{DLM}Relative output path    : {directory_ef}/{directory_mu}')
                if MPI_debug:
                    logger.info(f'{DLM}MPI debug files         : {directory_ef}/{directory_mu}/DEBUG')

            # =================================================================
            # INITIAL POISSON SOLUTION (WITHOUT FREE CHARGE)
            # =================================================================
            
            # Solve Poisson equation with only the external electric field
            # This provides the initial electrostatic potential before self-consistency
            p = PoissonProblem(
                mesh, 
                shape_class_name=shape_kind_poisson,
                dirichlet=dirichlet, 
                electric_field=ef,
                user_defined_parameters=user_defined_params
            )
    
            idj = '_init'
            
            # Initialize electrostatic potential
            if init_pot_name is not None:
                # Use external file for initial potential (for restart calculations)
                Vin = np.load(init_pot_name)
                p.epot = ElectrostaticPotential(p.fs, V=Vin)
            else:
                # Solve Poisson equation from scratch
                p.run()
                Vin = p.epot.V
    
            # Save initial potential for reference
            if rank == 0:
                np.save(path_mu+'/epot'+str(idj), Vin, allow_pickle=False)
                logger.info("")
                logger.info(f'Initial electrostatic potential saved to /epot{idj}.npy')

            # =================================================================
            # SELF-CONSISTENT CYCLE INITIALIZATION
            # =================================================================
    
            # Initialize Broyden mixing for accelerated convergence
            # This method helps achieve faster convergence by mixing old and new potentials
            up = Broyden(N=Vin.shape[0], M=maxter, beta=betamix, w0=w0, use_wm=use_wm)
    
            # Initialize arrays to track convergence progress
            n_resid_lst = []      # Negative charge density residuals
            n_resid_rel_lst = []  # Relative negative charge density residuals
            p_resid_lst = []      # Positive charge density residuals
            p_resid_rel_lst = []  # Relative positive charge density residuals
    
            # =================================================================
            # MAIN SELF-CONSISTENT LOOP
            # =================================================================
            
            if rank == 0:
                logger.info("")
                logger.info('Starting SCF Schrödinger-Poisson cycle')
            
            if rank == 0:
                logger.info("")
                logger.info("Legend: 1-Iteration 2-Total charge 3-Max potential  4-Mean abs error potential  ")
                logger.info("5-Negative density residue [cm^-1] 6-[%] 7-Positive density residue [cm^-1]  8-[%]  9-Data saved  10-Graph generated")
                logger.info(" " + "-"*80)
                logger.info(" (1)    (2)          (3)          (4)          (5)          (6)        (7)          (8)         (9) (10)")

            for j in range(maxiter):
                
                # Create electrostatic potential object for this iteration
                # First iteration: uses initial potential from Poisson solution
                # Later iterations: uses potential updated by Broyden mixing
                epot = ElectrostaticPotential(p.fs, V=Vin)
        
                # =============================================================
                # SCHRÖDINGER EQUATION SOLUTION (BAND STRUCTURE CALCULATION)
                # =============================================================
                
                # Solve the 8-band k·p Schrödinger equation for current potential
                bs = BandStructure(
                    mesh=mesh,
                    kzvals=kzvals[kzslice],  # This process's k-point subset
                    valence_band_edges=valence_band_edges,
                    principal_axis_direction=principal_axis_direction,
                    temperature=temperature,
                    k=k,  # Number of eigenvalues to find
                    e_search=e_search,
                    shape_functions=shape_kind_kp,
                    epot=epot,
                    logger=logger,
                    rescaling=rescaling,
                    # user_defined_parameters=user_defined_params
                    user_defined_params=user_defined_params
                )
                bs.run()
            
                # =============================================================
                # MPI DATA GATHERING AND SYNCHRONIZATION
                # =============================================================
                
                # Collect results from all MPI processes to reconstruct full band structure
                neig = bs.bands.shape[1]
                
                # Gather k-point values
                sendbuf = bs.kzvals
                recvbuf = np.zeros([nk])
                comm.Allgather(sendbuf, recvbuf)
                bs.kzvals = recvbuf
    
                # Gather band energies
                sendbuf = bs.bands
                recvbuf = np.zeros([nk, neig])
                comm.Allgather(sendbuf, recvbuf)
                bs.bands = recvbuf
            
                # Gather spinor character distributions
                sendbuf = bs.spinor_distribution
                recvbuf = np.zeros([nk, 8, neig])
                comm.Allgather(sendbuf, recvbuf) 
                bs.spinor_distribution = recvbuf
            
                # Gather regional norm distributions (for core vs shell analysis)
                sendbuf = bs.norm_sum_region
                recvbuf = np.zeros([nk, nreg, neig])
                comm.Allgather(sendbuf, recvbuf)
                bs.norm_sum_region = recvbuf
            
                # Gather electron wavefunction envelopes
                sendbuf = bs.psi_el
                recvbuf = np.zeros([nk, bs.psi_el.shape[1], bs.psi_el.shape[2], bs.psi_el.shape[3]], dtype='complex')
                comm.Allgather(sendbuf, recvbuf) 
                bs.psi_el = recvbuf
    
                # Gather hole wavefunction envelopes
                sendbuf = bs.psi_h
                recvbuf = np.zeros([nk, bs.psi_h.shape[1], bs.psi_h.shape[2], bs.psi_h.shape[3]], dtype='complex')
                comm.Allgather(sendbuf, recvbuf) 
                bs.psi_h = recvbuf
                
                # Clean up temporary communication buffers
                del(sendbuf)  
                del(recvbuf)
    
                # =============================================================
                # CHARGE DENSITY CALCULATION
                # =============================================================
                
                # Calculate free charge densities from the computed wavefunctions
                # This step converts the quantum mechanical wavefunctions into
                # classical charge densities for the Poisson equation
                rho_el = FreeChargeDensity(bs.fs_el)
                rho_h = FreeChargeDensity(bs.fs_h)
                
                # Add electron contributions to charge density
                rho_el.add_charge(bs.psi_el, 
                               np.array(bs.bands),
                               dk=dk,
                               mu=mu,  # Chemical potential
                               temp=temperature,
                               modified_EFA=modified_EFA, 
                               particle=particle_s_components,
                               norm_sum_region=bs.norm_sum_region,
                               thr_el=thr_el,
                               thr_h=thr_h)
            
                # Add hole contributions to charge density
                rho_h.add_charge(bs.psi_h, 
                               np.array(bs.bands),
                               dk=dk,
                               mu=mu,  # Chemical potential
                               temp=temperature, 
                               modified_EFA=modified_EFA,
                               particle=particle_p_components,
                               norm_sum_region=bs.norm_sum_region,
                               thr_el=thr_el,
                               thr_h=thr_h)
    
                # Calculate integrated total charges
                ntot_el, ptot_el = rho_el.get_total_charge() 
                ntot_h, ptot_h = rho_h.get_total_charge() 
            
                # Sum contributions from both electron and hole calculations
                ntot = ntot_el + ntot_h
                ptot = ptot_el + ptot_h
                total_charge = ntot + ptot
                
                # if rank == 0:
                #     logger.info(f'Total charge balance: {total_charge:.6e} e/cm³')

                # =============================================================
                # DATA OUTPUT AND VISUALIZATION
                # =============================================================
                
                try:
                    if rank == 0:
                        # Core band structure data
                        np.save(path_mu+'/bands', bs.bands)
                        np.save(path_mu+'/spinor_dist', bs.spinor_distribution)
                        np.save(path_mu+'/norm_sum_region', bs.norm_sum_region)
                        if generate_txt_files:
                            np.savetxt(path_mu+'/bands.txt', bs.bands, fmt='%.5g', delimiter=DLM)
                            # Save each k-point as separate text file due to 3D array structure
                            for i in range(bs.spinor_distribution.shape[0]):
                                slice_2d = bs.spinor_distribution[i]
                                np.savetxt(path_mu+f'/spinor_dist_{i}.txt', slice_2d, fmt='%.5g', delimiter=DLM)  
                            # Save each k-point as separate text file due to 3D array structure
                            for i in range(bs.norm_sum_region.shape[0]):
                                slice_2d = bs.norm_sum_region[i]
                                np.savetxt(path_mu+f'/norm_sum_region_{i}.txt', slice_2d, fmt='%.5g', delimiter=DLM)  

                        # Wavefunction envelope functions
                        np.save(path_mu+'/envelope_el', bs.psi_el)  
                        np.save(path_mu+'/envelope_h', bs.psi_h) 
                        
                        # Charge and matrix data
                        np.save(path_mu+'/total_charge'+idj, total_charge)
                        if generate_txt_files:
                            np.savetxt(path_mu+'/total_charge'+idj+'.txt', [total_charge])
                            
                        # Save the overlap matrix for post-processing
                        save_npz(path_mu+"/B.npz", bs.solver[0].bgl)
                        
                        saved_data = "✓" 
                except Exception as e:
                    logger.error(f"Error saving output files: {e}")
                    execution_aborted(e)  # Handle file writing errors gracefully
                
                try: 
                    # Generate visualization plots
                    if rank == 0:
                        if generate_png_graphs:
                            # generate plots
                            plot_production(bs,p,rho_el,rho_h,path_mu)                            
                            saved_graphs = "✓"
                        else:
                            saved_graphs = "✗"
                except Exception as e:
                    logger.error(f"Error generating plots: {e}")
                    execution_aborted(e)
                    
                # =============================================================
                # CONVERGENCE ASSESSMENT
                # =============================================================
                
                n_resid = None
                p_resid = None
                n_resid_rel = None
                p_resid_rel = None
                if j > 0: 
                    # Only check convergence after first iteration
                    # if rank == 0:
                    #     logger.info(f'Convergence check for iteration {j+1}...')
                    
                    # Calculate charge density residuals between iterations
                    n_resid, p_resid = get_density_resid(rho_el, rho_h, rho_el_prev, rho_h_prev)
    
                    # Calculate relative residuals to assess convergence
                    if n_resid > 1e-10: 
                        n_resid_rel = n_resid / np.abs(ntot_prev)
                    else:
                        n_resid_rel = 0.0
                    if p_resid > 1e-10:
                        p_resid_rel = p_resid / np.abs(ptot_prev)
                    else:
                        p_resid_rel = 0.0
    
                    # Log convergence metrics
                    # if rank == 0:
                    #     logger.info(f'Negative density residual [cm^-1]: {n_resid:.6e}')
                    #     logger.info(f'Negative density residual [relative]: {n_resid_rel:.6e}')
                    #     logger.info(f'Positive density residual [cm^-1]: {p_resid:.6e}')
                    #     logger.info(f'Positive density residual [relative]: {p_resid_rel:.6e}')
    
                    # Store convergence history for analysis
                    n_resid_lst.append(n_resid)
                    n_resid_rel_lst.append(n_resid_rel)
                    p_resid_lst.append(p_resid)
                    p_resid_rel_lst.append(p_resid_rel)
                    
                    if rank == 0:
                        np.save(path_mu+"/n_resid", np.array(n_resid_lst))
                        np.save(path_mu+"/n_resid_rel", np.array(n_resid_rel_lst))
                        np.save(path_mu+"/p_resid", np.array(p_resid_lst))
                        np.save(path_mu+"/p_resid_rel", np.array(p_resid_rel_lst))
    
                    # Check if convergence criteria are met
                    if (n_resid_rel <= maxchargeerror and p_resid_rel <= maxchargeerror):
                        if rank == 0:
                            logger.info("")
                            logger.info("Convergence criteria met")
                        break  # Exit self-consistent loop
                        
                # Store current densities for next iteration's convergence check
                rho_el_prev = copy.copy(rho_el)
                rho_h_prev = copy.copy(rho_h)
                ntot_prev = ntot
                ptot_prev = ptot
                
                # =============================================================
                # POISSON EQUATION UPDATE
                # =============================================================
                
                # Solve Poisson equation with updated free charge densities
                # This creates the new electrostatic potential for the next iteration
                p = PoissonProblem(
                    mesh, 
                    shape_class_name=shape_kind_poisson,
                    dirichlet=dirichlet,
                    electric_field=ef, 
                    user_defined_parameters=user_defined_params,
                    rho_el=rho_el,
                    rho_h=rho_h 
                )
                p.run()
                Vout = p.epot.V
                
                # Monitor potential changes between iterations
                if rank == 0:
                    max_pot_variation = np.max(np.abs(Vin-Vout))
                    mae_pot = np.sum(np.abs(Vin-Vout)) / Vin.shape[0]
                    # logger.info(f'Maximum potential variation: {max_pot_variation:.6e} eV')
                    # logger.info(f'Mean absolute error potential: {mae_pot:.6e} eV')

                # Tabular output for iteration summary
                if rank == 0:
                    # Handle None values for first iteration
                    n_resid_str = f"{n_resid:>12.4e}" if n_resid is not None else "     N/A    "
                    n_resid_rel_str = f"{n_resid_rel:>12.4e}" if n_resid_rel is not None else "     N/A    "
                    p_resid_str = f"{p_resid:>12.4e}" if p_resid is not None else "     N/A    "
                    p_resid_rel_str = f"{p_resid_rel:>12.4e}" if p_resid_rel is not None else "     N/A    "
                    
                    logger.info(f'{j+1:>3} {total_charge:>12.4e} {max_pot_variation:>12.4e} '
                               f'{mae_pot:>12.4e} {n_resid_str} {n_resid_rel_str} {p_resid_str} '
                               f'{p_resid_rel_str}   {saved_data}   {saved_graphs}')

                # Manual flush of log handlers
                if rank == 0:
                    for handler in logger.handlers:
                        handler.flush()
                        if hasattr(handler, 'stream'): 
                            handler.stream.flush()
                            try: os.fsync(handler.stream.fileno())
                            except: pass
                
                # Apply Broyden mixing to accelerate convergence
                # This combines the input and output potentials to create a better guess
                if j in toreset:
                    reset = True
                else:
                    reset = False
                Vout = up.update(xin=Vin[:,np.newaxis], xout=Vout[:,np.newaxis], reset=reset).squeeze()
                
                # Update input potential for next iteration
                idj = '_conv'
                Vin = np.copy(Vout)
                
                # Save the updated potential
                np.save(path_mu+'/epot'+str(idj), Vin)
                
                # Memory management to prevent memory leaks
                if j < maxiter - 1:
                    del(bs)
                    gc.collect()
                if j == maxiter - 1:
                    if rank == 0:
                        logger.warning('Max iteration number reached - convergence not achieved')

            # =================================================================
            # FINAL PLOTS AND CLEANUP FOR THIS CHEMICAL POTENTIAL
            # =================================================================
            
            # probably redundent, last iteration si good as well
            
            # Generate final plots with converged results
            if rank == 0 and generate_png_graphs:
                
                logger.info("")
                logger.info('Generating final plots with converged results')

                plot_production(bs,p,rho_el,rho_h,path_mu)

                logger.info(FMT_STR.format('Band structure plot', f'./energy_bands.png'))
                logger.info(FMT_STR.format('Carrier density plot', f'./carrier_density.png'))
                logger.info(FMT_STR.format('Potential plot', f'./potential.png'))

            # Optional: Use converged potential as starting point for next chemical potential
            # init_pot_name = path_mu+"/epot_conv.npy"
    
        if rank == 0:
            logger.info("")
            logger.info(f'All chemical potentials completed for electric field {ef}')
    
    if rank == 0:
        logger.info("")
        logger.info('All electric field calculations completed successfully')
    
    # Log successful completion of the entire calculation
    if rank == 0:
        execution_successful()

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == '__main__':
    """
    Script entry point with timing instrumentation.
    """
    
    # Execute main calculation with high-resolution timing instrumentation
    tic()                              # Start precision timer (from nwkpy)
    main()                             # Execute complete band structure calculation
    toc()                              # Stop timer and display total elapsed time