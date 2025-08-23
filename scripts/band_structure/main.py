#!/usr/bin/env python

"""
Scrit for Core-Shell Nanowire Band Structure Calculator
using the 8-band k·p method with finite element discretization.
Based on the nwkpy library.

The calculation includes:
- Band structure computation along the nanowire axis
- Electrostatic potential calculation via Poisson equation
- Charge density calculation with proper carrier statistics
- Parallel processing using MPI for k-point calculations
- Conditional plotting based on input flags

Physical Model:
- Uses 8-band k·p Hamiltonian for semiconductor heterostructures
- Includes both electron and hole states (conduction and valence bands)
- Accounts for strain, spin-orbit coupling, and band mixing effects
- Solves coupled k.p-Poisson equations self-consistently

Numerical Methods:
- Finite Element Method (FEM) for spatial discretization
- Sparse matrix eigenvalue solvers for band structure
- MPI parallelization over k-points for efficiency

Author: [A Vezzosi, G Goldoni]
Date: [August 2025]
"""

# =============================================================================
# GENERAL IMPORTS AND SETUP
# =============================================================================

# Standard library imports for system operations
import sys                              # System-specific parameters and functions
import os                               # Operating system interface for file operations

# Numerical computing libraries
import numpy as np                      # Fundamental numerical computing package

# Message Passing Interface for parallel computing
from mpi4py import MPI                  # Python bindings for MPI parallelization

# Input/output and logging utilities
import logging                          # Logging library for structured output control

# Scientific computing utilities
from scipy.sparse import save_npz, load_npz  # Sparse matrix I/O operations for efficiency

# import socket                         # Network info for debugging (commented - used in utilities)
import gc                               # Garbage collection for memory management

# =============================================================================
# CORE LIBRARY IMPORTS
# =============================================================================

# High-level timing and diagnostic utilities from nwkpy
from nwkpy import tic, toc              # High-resolution timing functions for performance analysis
from nwkpy import library_header        # Display library version and build information
# from nwkpy import Logger             # Legacy logger (deprecated in favor of Python logging)

# Core computational classes for nanowire band structure calculations
from nwkpy.fem import Mesh              # Finite element mesh handling and region assignment
from nwkpy import BandStructure         # k·p band structure solver with MPI parallelization
from nwkpy import PoissonProblem        # Electrostatic potential solver via Poisson equation
from nwkpy import FreeChargeDensity     # Free carrier density calculator with Fermi statistics
from nwkpy import ElectrostaticPotential # Potential field container and manipulation
from nwkpy import DopingChargeDensity   # Doping charge density (not used in this intrinsic case)
from nwkpy import _constants            # Physical constants (fundamental and material-specific)
from nwkpy import MPI_debug_setup  # MPI debugging utilities for parallel development

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
SCRIPT_NAME = 'Core-Shell Nanowire band structure calculation'  

# =============================================================================
# IMPORT INPUT PARAMETERS
# =============================================================================

# Import all simulation parameters from the input configuration file
indata = INPUT_FILE_NAME+'.py'
if not os.path.exists(indata):  
    execution_aborted(f"Input file '{indata}' not found")  # Graceful termination with error logging
else:
    from indata import *

# Execution control logic based on user-specified flags
# Determines whether to run calculations or only generate plots from existing data
run_calculation = True if not plot_only_mode else False

# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================
 
# Construct file system paths for output data storage
cdir = os.getcwd()                     # Current working directory (script location)

# Construct output data directory path
outdata_path = os.path.join(cdir, directory_name)

# Construct full path to log file for structured output
log_file = os.path.join(outdata_path, LOG_FILE_NAME + ".log")

# Creates the directory tree if it doesn't exist, no error if it already exists
os.makedirs(outdata_path, exist_ok=True)

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
print(f'\nAll log messages sent to file: {log_file}\n')
    
# =============================================================================
# USER PARAMETERS
# =============================================================================

# Read band parameters provided by the user, if any
if user_parameters_file is not None:
    try:
        with open(user_parameters_file, 'r') as f:
            user_params = eval(f.read())
    except FileNotFoundError:
        execution_aborted(f"User parameters file '{user_parameters_file}' not found.")
else:
    user_params = None

# =============================================================================
# MESH FILES
# =============================================================================

# Construct paths to required mesh files (input data)
mesh_file = os.path.join(cdir, mesh_name + ".msh")  # GMSH mesh geometry file
mesh_data = os.path.join(cdir, mesh_name + ".dat")  # Mesh metadata and region definitions

# Verify that required mesh files exist before proceeding
try:
    if not os.path.exists(mesh_file):  # Check for primary mesh file
        raise FileNotFoundError(f"Mesh file '{mesh_file}' not found")
    if not os.path.exists(mesh_data):  # Check for mesh metadata file
        raise FileNotFoundError(f"Mesh data file '{mesh_data}' not found")
except FileNotFoundError as f:
    execution_aborted(f)               # Graceful termination with error logging

# =============================================================================
# MPI SETUP AND PROCESS INITIALIZATION
# =============================================================================

# Initialize MPI communicator for parallel k-point calculations
comm = MPI.COMM_WORLD                  # Global communicator including all MPI processes
rank = comm.Get_rank()                 # Process rank (0 to size-1), 0 is the master process
size = comm.Get_size()                 # Total number of MPI processes in the calculation
        
# Synchronize all processes after logging setup
comm.Barrier()

# =============================================================================
# INPUTS CONSISTENCY CHECKS
# =============================================================================

def consistency_checks(indata):
    """
    Perform comprehensive consistency checks on input parameters.

    Args:
        indata: The input data module containing all parameters.

    Raises:
        ValueError: If any parameter is invalid, inconsistent, or out of range
        
    Note:
        This function only validates parameter values, not file existence.
        File checks are performed separately in the main execution flow.
    """

    # log input file
    logger.info("")
    logger.info(f"Input read from file {indata}")

    # Check if all specified materials exist in the nwkpy parameter database
    if any(m not in params and m not in user_params for m in material):
        logger.warning(f"material = {material} in not in the database or user parameters")
        logger.warning(f"Available materials in database : {list(params.keys())}")
        if not(user_params is None):
            logger.warning(f"Available materials in user parameters: {list(user_params.keys())}")
        raise ValueError("Invalid material - Choose available materials or update the parameter database")

    # Validate crystallographic growth direction specification
    if principal_axis_direction not in PRINCIPAL_AXIS_DIRECTIONS:
        logger.warning(f"principal_axis_direction = {principal_axis_direction} not available.")
        logger.warning(f"Valid directions: {PRINCIPAL_AXIS_DIRECTIONS}")
        raise ValueError("Invalid direction - Choose a valid principal axis direction")

    # Verify character thresholds are within valid probability range [0,1]
    if any(x < 0 or x > 1 for x in character_threshold):
        logger.warning(f"character_threshold = {character_threshold} out of range")
        logger.warning(f"Character thresholds must be in the range [0, 1]")
        raise ValueError("Invalid character threshold - Choose values in the correct range")

    # Validate carrier type specifications for each material region
    if any(c not in CARRIER_TYPES for c in carrier):
        logger.warning(f"carrier = {carrier} not available")
        logger.warning(f"Valid carrier types: {CARRIER_TYPES}")
        raise ValueError(f"Invalid carrier type - Choose a valid carrier type for each particle")

    # Check shape function specifications for k·p Hamiltonian
    if any(s not in SHAPE_FUNCTION_TYPES for s in shape_function_kp):
        logger.warning(f"shape_function_kp = {shape_function_kp} not available")
        logger.warning(f"Valid shape function types: {SHAPE_FUNCTION_TYPES}")
        raise ValueError(f"Invalid shape function type - Choose a valid value")

    # Validate shape function for Poisson equation solver
    if (shape_function_poisson not in SHAPE_FUNCTION_TYPES):
        logger.warning(f"shape_function_poisson = {shape_function_poisson} not available")
        logger.warning(f"Valid shape function types: {SHAPE_FUNCTION_TYPES}")
        raise ValueError(f"Invalid shape function type - Choose a valid value")

    # Ensure temperature is physically meaningful (positive Kelvin)
    if temperature <= 0:
        logger.warning(f"Temperature = {temperature} is negative")
        logger.warning(f"Temperatures in K must be positive")
        raise ValueError("Invalid temperature - Choose a positive value")
    
    # Validate eigenvalue count for diagonalization
    if number_eigenvalues < 1:
        logger.warning(f"number_eigenvalues = {number_eigenvalues} is zero or negative")
        logger.warning(f"The number of eigenvalues for diagonalization should be at least 1")
        raise ValueError("Invalid number of eigenvalues - Choose a positive value")
 
    # Check k-space range ordering (start < end)
    if k_range[0] > k_range[1]:
        logger.warning(f"k_range = {k_range} has invalid order of extrema")
        logger.warning("k-point start must be less than end")
        raise ValueError("Invalid k-value range extrema - Choose correct order")
        
    # Validate k-point sampling density
    if number_k_pts < 1:
        logger.warning(f"number_k_pts = {number_k_pts} is zero or negative")
        logger.warning(f"The number of k points should be at least 1")
        raise ValueError("Number of k-points must be at least 1")

# =========================================================================
# PHYSICAL SYSTEM CONFIGURATION
# =========================================================================

reg2mat = {}
valence_band_edges = {}
rescaling = {}
mat2partic = {}

# iterate over the materials
for i, m in enumerate(material):
    
# Define material assignment mapping from mesh regions to materials
# reg2mat = {
#     1: material[0],                    # Region 1 (typically inner core) → first material
#     2: material[1]                     # Region 2 (typically outer shell) → second material
# }
    reg2mat[i+1] = m

# Define valence band edge alignment for heterostructure band offsets
# These values set the energy reference level for each material, determining
# whether the heterostructure has Type I (nested) or Type II (staggered) alignment
# valence_band_edges = {
#     material[0]: valence_band[0],      # Core material valence band edge (eV)
#     material[1]: valence_band[1]       # Shell material valence band edge (eV)
# }
    valence_band_edges[m] = valence_band[i]

# Configure P-parameter rescaling for spurious solution suppression
# The 8-band k·p model can produce unphysical high-energy solutions
# Reference: B. A. Foreman, Phys. Rev. B 56, R12748 (1997)
# Available rescaling methods (from S. Birner thesis):
# - 'S=0': Use standard Ep evaluation according to Eq. 6.158
# - 'S=1': Use modified Ep evaluation according to Eq. 6.159
# - Numerical value: Apply fractional reduction of Ep (e.g., 0.26 = 26% reduction)
# rescaling = {
#     material[0]: rescale[0],           # Core material rescaling method
#     material[1]: rescale[1]            # Shell material rescaling method
# }
    rescaling[m] = rescale[i]

# Define dominant carrier types for each material region
# This is particularly important for broken-gap heterostructures 
# mat2partic = {
#     material[0]: carrier[0],           # Core material dominant carrier type
#     material[1]: carrier[1]            # Shell material dominant carrier type
# }
    mat2partic[m] = carrier[i]

# =============================================================================
# K-SPACE SAMPLING CONFIGURATION
# =============================================================================

# Extract k-space sampling range from input parameters
# Note: Only positive k-values are computed due to inversion symmetry
kzin_rel = k_range[0]                  # Initial relative k_z value (dimensionless)
kzfin_rel = k_range[1]                 # Final relative k_z value (dimensionless)

# Convert relative k-values to absolute units (Å⁻¹)  
# Scaling factor: π/√3/a relates k-space to the first Brillouin zone
# where 'a' is the lattice constant in Angstroms
kzvals = np.linspace(kzin_rel, kzfin_rel, number_k_pts) * np.pi / np.sqrt(3.0) / lattice_constant

# =============================================================================
# NUMERICAL SOLVER CONFIGURATION
# =============================================================================

# Configure shape function selection for finite element basis
# Different shape functions provide trade-offs between accuracy and computational cost
shape_kind_kp = {
    'el': shape_function_kp[0],        # Electron states
    'h': shape_function_kp[1]          # Hole states
}
shape_kind_poisson = shape_function_poisson  # Shape functions for electrostatic potential

# Set eigenvalue solver parameters
# WARNING: This line appears redundant but is required for compatibility
# with the nwkpy library's internal variable naming conventions
k = number_eigenvalues                 # Number of eigenstates to compute per k-point

# =============================================================================
# CHARGE CHARACTERIZATION PARAMETERS
# =============================================================================

# Define character thresholds for electron/hole state classification
# States with character above these thresholds contribute to charge densities
# This provides physical filtering of mixed states in heterostructures
thr_el = character_threshold[0]        # Electron character threshold (0.0-1.0)
thr_h = character_threshold[1]         # Hole character threshold (0.0-1.0)

# Configure particle component specifications for charge density calculations
# These parameters control which orbital components contribute to densities
# Legacy parameters maintained for compatibility with nwkpy library
particle_s_components = 'electron'     # S-orbital contributions (conduction band character)
particle_p_components = 'electron'     # P-orbital contributions (valence band character)

# =============================================================================
# PLOT PRODUCTION FUNCTION
# =============================================================================
def plot_production(bs,p,rho_el,rho_h):
    """
    Generate comprehensive visualization plots for the nanowire calculation.
    
    Args:
        bs (BandStructure): Band structure object containing eigenvalues and eigenvectors
        p (PoissonProblem): Poisson solver object with electrostatic potential
        rho_el (FreeChargeDensity): Electron charge density distribution
        rho_h (FreeChargeDensity): Hole charge density distribution
        
    Side Effects:
        - Saves PNG files to output directory
        - Logs file creation to the main logger
        
    Note:
        Plotting preferences are imported from indata.py and control
        figure size, axis ranges, colormaps, and other visual parameters.
    """

    logger.info("")
    logger.info('Generating plots')
    print('output directory:', outdata_path)
    
    # Generate band structure dispersion plot showing E(k) relationships
    # Displays energy bands vs k-space with color-coded carrier character
    figure_file = os.path.join(outdata_path, 'energy_bands.png')
    figure_bands = bs.plot_bands(**plotting_preferencies_bands)
    figure_bands.savefig(figure_file, bbox_inches="tight")
    logger.info(FMT_STR.format('Band structure plot', f'{directory_name}/energy_bands.png'))
    
    # Generate 2D charge density contour plot in the nanowire cross-section
    # Shows spatial distribution of free carriers with material interface overlay
    figure_file = os.path.join(outdata_path, 'carrier_density.png')
    figure_density = bs.plot_density(rho_el, rho_h, **plotting_preferencies_density)
    figure_density.savefig(figure_file, bbox_inches="tight")
    logger.info(FMT_STR.format('Charge density plot', f'{directory_name}/carrier_density.png'))
    
    # Generate electrostatic potential contour plot showing built-in fields
    # Displays potential landscape created by charge redistribution and external fields
    figure_file = os.path.join(outdata_path, 'potential.png')
    figure_potential = p.epot.plot(**plotting_preferencies_potential)
    figure_potential.savefig(figure_file, bbox_inches="tight")
    logger.info(FMT_STR.format('Electrostatic potential plot', f'{directory_name}/potential.png'))

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def generate_png_graphs_from_data():
    """
    Generate visualization plots from previously saved calculation data.
    
    This function enables plot-only mode, allowing users to create or modify
    visualizations without re-running expensive band structure calculations.
    Particularly useful for:
    - Adjusting plot parameters or visual styling
    - Generating additional figure formats
    - Creating plots for presentations or publications
    - Troubleshooting visualization issues
    
    The function loads all necessary data from .npy files in the output directory,
    reconstructs the computational objects needed for plotting, and generates
    the complete set of visualization plots.
    
    Returns:
        bool: True if plots were successfully generated, False if errors occurred
        
    Raises:
        SystemExit: If required data files are missing or corrupted
        
    """
    
    logger.info("")
    logger.info('Generating plots from saved data')
    logger.info(f'Looking for data files in {directory_name}')
    
    # Define list of required data files for plot generation
    # These files contain the complete numerical results from a previous calculation
    required_files = ['bands.npy', 'spinor_dist.npy', 'norm_sum_region.npy', 
                     'envelope_el.npy', 'envelope_h.npy', 'kzvals.npy']
    
    # Check for existence of all required data files
    missing_files = []
    for file in required_files:
        file_name = os.path.join(outdata_path, file)
        if not os.path.exists(file_name):
            missing_files.append(file)
    
    # Abort plot generation if any required files are missing
    if missing_files:
        logger.error(f'Missing data files {missing_files} in {directory_name}')
        logger.error('Cannot generate plots without complete data set')
        execution_aborted('Missing data files')

    # Load all saved numerical results from binary NumPy files
    # These files contain the complete state of a finished calculation
    try:
        bands = np.load(os.path.join(outdata_path,'bands.npy'))                    # Energy eigenvalues
        spinor_dist = np.load(os.path.join(outdata_path,'spinor_dist.npy'))        # Spinor characters
        norm_sum_region = np.load(os.path.join(outdata_path,'norm_sum_region.npy')) # Regional normalizations
        envelope_el = np.load(os.path.join(outdata_path,'envelope_el.npy'))        # Electron envelopes
        envelope_h = np.load(os.path.join(outdata_path,'envelope_h.npy'))          # Hole envelopes
        kzvals = np.load(os.path.join(outdata_path,'kzvals.npy'))                  # k-point values
        logger.info('All required data files loaded successfully')
        
    except Exception as e:
        logger.error(f'Error loading data files: {str(e)}')
        return False
    
    # Reconstruct computational objects needed for plotting functionality

    # Recreate finite element mesh with material and particle assignments
    mesh = Mesh(
        mesh_name=mesh_file,           # GMSH mesh geometry file
        reg2mat=reg2mat,               # Region to material mapping
        mat2partic=mat2partic          # Material to particle type mapping
    )

    # Recreate material parameter dictionary for computational classes
    # user_defined_params = {
    #     material[0]: params[material[0]], 
    #     material[1]: params[material[1]]
    # }
    
    if user_params is not None:
        user_defined_params = get_parameters(material, user_params)
    else:
        user_defined_params = get_parameters(material)
        
    # Reconstruct band structure object with loaded data
    bs = BandStructure(
        mesh=mesh,
        kzvals=kzvals,
        valence_band_edges=valence_band_edges,
        principal_axis_direction=principal_axis_direction,
        temperature=temperature,
        k=number_eigenvalues,
        e_search=e_search,
        shape_functions=shape_kind_kp,
        epot=None,                     # Will be handled separately for potential plots
        logger=logger,
        rescaling=rescaling,
        user_defined_params=user_defined_params
    )
    
    # Assign loaded numerical data to the band structure object
    # This restores the complete state from the previous calculation
    bs.bands = bands                   # Energy eigenvalues (k-point, eigenvalue)
    bs.spinor_distribution = spinor_dist  # Spinor character (k-point, spinor, eigenvalue)
    bs.norm_sum_region = norm_sum_region  # Regional normalization (k-point, region, eigenvalue)
    bs.psi_el = envelope_el            # Electron envelope functions (complex)
    bs.psi_h = envelope_h              # Hole envelope functions (complex)
    bs.kzvals = kzvals                 # k-point sampling values
    
    # Calculate k-point spacing for charge density integration
    # Required for proper normalization of charge density calculations
    try:
        dk = kzvals[1] - kzvals[0]     # Uniform k-point spacing
    except IndexError:
        dk = 0.0                       # Single k-point case (no integration)
    
    # Recreate free carrier charge density objects for plotting
    # These objects handle the spatial distribution of electrons and holes
    rho_el = FreeChargeDensity(bs.fs_el)  # Electron charge density calculator
    rho_h = FreeChargeDensity(bs.fs_h)    # Hole charge density calculator
    
    # Reconstruct charge density distributions from loaded envelope functions
    # This reproduces the charge density calculation results
    rho_el.add_charge(
        envelope_el,                   # Electron envelope functions
        bands,                         # Band structure energies
        dk=dk,                         # k-point spacing for integration
        mu=chemical_potential,         # Chemical potential (Fermi level)
        temp=temperature,              # Temperature for Fermi-Dirac statistics
        modified_EFA=modified_EFA,     # Modified envelope function approximation
        particle=particle_s_components, # Particle type for s-orbitals
        norm_sum_region=norm_sum_region, # Regional normalization factors
        thr_el=thr_el,                 # Electron character threshold
        thr_h=thr_h                    # Hole character threshold
    )
    
    rho_h.add_charge(
        envelope_h,                    # Hole envelope functions
        bands,                         # Band structure energies
        dk=dk,                         # k-point spacing for integration
        mu=chemical_potential,         # Chemical potential (Fermi level)
        temp=temperature,              # Temperature for Fermi-Dirac statistics
        modified_EFA=modified_EFA,     # Modified envelope function approximation
        particle=particle_p_components, # Particle type for p-orbitals
        norm_sum_region=norm_sum_region, # Regional normalization factors
        thr_el=thr_el,                 # Electron character threshold
        thr_h=thr_h                    # Hole character threshold
    )
    
    # Attempt to recreate electrostatic potential for potential plots
    # This requires solving the Poisson equation or loading saved potential
    try:
        p = PoissonProblem(
            mesh, 
            shape_class_name=shape_kind_poisson,
            dirichlet=dirichlet,       # Boundary condition specifications
            electric_field=electric_field, # External electric field
            user_defined_parameters=user_defined_params
        )
        
        # Load pre-computed potential if available, otherwise solve Poisson equation
        if init_pot_name is not None:
            Vin = np.load(init_pot_name)
            p.epot = ElectrostaticPotential(p.fs, V=Vin)
        else:
            p.run()                    # Solve Poisson equation for electrostatic potential
        
    except Exception as e:
        logger.warning(f'Could not generate potential plot: {str(e)}')

    # =====================================================================
    # PLOT GENERATION
    # =====================================================================

    # Generate all visualization plots using the reconstructed objects
    plot_production(bs,p,rho_el,rho_h)
    
    logger.info("")
    logger.info('Plot generation completed successfully')
    return True
        
# =============================================================================
# MAIN CALCULATION FUNCTION
# =============================================================================

def main():
    """
    Main orchestration function for the complete nanowire band structure calculation.
    
    1. Initialization and Validation:
       - Display script header and library information
       - Validate all input parameters for consistency
       - Load and validate finite element mesh
       
    2. Physical System Setup:
       - Configure material properties and band alignments
       - Set up k-space sampling for band structure calculation
       - Initialize numerical solver parameters
       
    3. Electrostatic Potential Calculation:
       - Solve Poisson equation for built-in potential
       - Handle boundary conditions and external fields
       - Broadcast results to all MPI processes
       
    4. Band Structure Calculation:
       - Distribute k-points among MPI processes
       - Solve 8-band k·p Hamiltonian at each k-point
       - Collect results from all processes
       
    5. Charge Density Analysis:
       - Calculate free carrier densities using envelope functions
       - Apply Fermi-Dirac statistics at specified temperature
       - Verify charge balance for system consistency
       
    6. Output Generation:
       - Save numerical results in multiple formats
       - Generate visualization plots if requested
       - Log completion status and timing information
    
    The function supports different execution modes:
    - Full calculation: Complete band structure computation with all outputs
    - Plot-only mode: Generate plots from existing data files without recalculation
    
    The calculation is fully parallelized using MPI, with automatic load balancing
    of k-point calculations across available processes.
    
    Raises:
        SystemExit: If critical errors occur during any phase of the calculation
        
    Note:
        All MPI processes execute this function, but certain operations (file I/O,
        logging) are restricted to rank 0 to avoid conflicts and duplicate output.
    """
    
    # =============================================================================
    # HEADER AND INITIALIZATION
    # =============================================================================
    
    if rank == 0:
        # Display formatted script header with execution information
        print_header(SCRIPT_NAME)
        
        # Display nwkpy library version and build information
        library_header()

    # Synchronize after header display
    comm.Barrier()
    
    # =========================================================================
    # MESH LOADING AND VALIDATION
    # =========================================================================

    # Load finite element mesh with material and particle type assignments
    mesh = Mesh(
        mesh_name=mesh_file,           # Path to GMSH mesh geometry file
        reg2mat=reg2mat,               # Mapping from mesh regions to materials
        mat2partic=mat2partic          # Mapping from materials to dominant carrier types
    ) 

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
                logger.info(f'{DLM}{line}')

    comm.Barrier()

    # =============================================================================
    # VARIOUS SETTINGS
    # =============================================================================
    
    # Extract number of distinct material regions from mesh
    nreg = len(mesh.region_labels)     

    # Set chemical potential for this calculation
    mu = chemical_potential

    if MPI_debug:                               # to all ranks!
        MPI_debug_setup(outdata_path+'/DEBUG')  # create DEBUG directory

    # =============================================================================
    # MATERIAL PARAMETER DEFINITION AND LOGGING
    # =============================================================================

    # Create user-defined material parameter dictionary for computational classes
    # user_defined_params = {
    #     material[0]: params[material[0]], # Core material parameters from database
    #     material[1]: params[material[1]]  # Shell material parameters from database
    # }
    user_defined_params = None

    if rank == 0:
        logger.info("")

        # Create user-defined material parameter dictionary for computational classes
        if user_params is not None:
            logger.info(f"User parameters read from file {user_parameters_file}")
            user_defined_params = get_parameters(material, user_params)
        else:
            user_defined_params = get_parameters(material)

        # Log material parameters for each region
        for m in material:
            log_material_params(m, user_defined_params[m])

    # Broadcast of parameters to all processes
    user_defined_params = comm.bcast(user_defined_params, root=0)

    # Synchronization after broadcast
    comm.Barrier()

    if rank == 0 and size > 1:
        logger.info(f'Material parameters broadcasted to all {size} MPI processes')

    # =============================================================================
    # VALIDATION OF INPUT PARAMETERS
    # =============================================================================
    if rank == 0:

        try:                              # Attempt parameter validation
            consistency_checks(indata)  
        except ValueError as e:           # Handle validation failures gracefully
            execution_aborted(e)          # Log error and terminate with proper cleanup
        else:                             # Validation successful - proceed with calculation
            logger.info("")
            logger.info(f'Input parameters consistency checks passed')

    comm.Barrier()

    # =============================================================================
    # PLOT GENERATION MODE
    # =============================================================================
    
    # Handle plot-only mode: generate visualizations from saved data without calculation
    if plot_only_mode:
        try:
            # Check MPI compatibility - plot generation requires single process
            if size > 1 and rank == 0:
                logger.error(f'plot_only_mode = {plot_only_mode} is incompatible with MPI parallelization')
                raise ValueError('MPI incompatible - Run plot-only mode on a single process')
        except ValueError as e:
            execution_aborted(e)

        logger.info("")
        logger.info('Running in plot-only mode - no calculations performed')

        # Attempt to generate plots from existing data files
        success = generate_png_graphs_from_data()
        
        try:
            if not success:
                raise ValueError("Plot generation failed")
        except ValueError as e:
            execution_aborted(e)
        else:
            execution_successful()    # Log successful completion
        return                        # Exit main function after plot generation
    
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
    # LOGGING INPUTS AND CONFIGURATIONS
    # =========================================================================
    
    # Log comprehensive configuration summary for documentation
    if rank == 0:
        # Display system configuration in tabular format
        log_system_configuration_summary(material,valence_band,rescale,
                        carrier,shape_function_kp,character_threshold)    
        # Log physical parameters for the simulation
        log_physical_parameters(temperature,chemical_potential,principal_axis_direction)
        # Log electrostatic potential solver configuration
        log_electrostatic_potential_configuration(shape_function_poisson,
                        dirichlet,electric_field,init_pot_name)
        # Log band structure calculation parameters
        log_band_structure_calculation_parameters(number_eigenvalues,e_search,kzvals,number_k_pts)
        # Log computational system and MPI configuration
        log_computational_system_information(size,kmaxlocal,MPI_debug)

        # the following stuff could be extracted as a utility function
        if rank == 0:
            logger.info("")
            logger.info("File management")
            logger.info(f"{DLM}Current directory         : {cdir}")
            logger.info(f"{DLM}Output root directory     : {directory_name}")
            logger.info(f"{DLM}Text data file generation : {generate_txt_files}")
            logger.info(f"{DLM}Graph generation          : {generate_png_graphs}")

    comm.Barrier()

    # =========================================================================
    # ELECTROSTATIC POTENTIAL CALCULATION
    # =========================================================================
    
    # Initialize Poisson equation solver for electrostatic potential calculation
    # Handles boundary conditions, external fields, and material interfaces
    p = PoissonProblem(
        mesh, 
        shape_class_name=shape_kind_poisson,   # Finite element shape functions
        dirichlet=dirichlet,                   # Dirichlet boundary condition specification
        electric_field=electric_field,         # External electric field configuration
        user_defined_parameters=user_defined_params  # Material-specific parameters
    ) 
    
    # Solve Poisson equation with MPI coordination to avoid redundant computation
    Vin = None                        # Initialize potential array
    if init_pot_name is not None:
        # Load pre-computed electrostatic potential from file
        if rank == 0:
            Vin = np.load(init_pot_name)
            logger.info(f'Loaded initial potential from {init_pot_name}')
    else:
        # Solve Poisson equation de novo (rank 0 only to avoid redundant computation)
        if rank == 0:
            logger.info("")
            logger.info('Solving Poisson equation for electrostatic potential')
            p.run()                   # Execute Poisson solver
            Vin = p.epot.V            # Extract potential values

    # Broadcast electrostatic potential to all MPI processes
    Vin = comm.bcast(Vin, root=0)     # MPI broadcast from rank 0 to all processes
    if rank == 0 and size > 1:
        logger.info(f'Electrostatic potential broadcasted to all MPI processes')
    
    # Create electrostatic potential object on all processes
    p.epot = ElectrostaticPotential(p.fs, V=Vin)
        
    # =========================================================================
    # BAND STRUCTURE CALCULATION 
    # =========================================================================
    
    # Initialize band structure solver with this process's assigned k-point subset
    # Each MPI process handles a different range of k-points 
    bs = BandStructure(
        mesh=mesh,                             # Finite element mesh
        kzvals=kzvals[kzslice],                # k-points assigned to this process
        valence_band_edges=valence_band_edges, # Band alignment configuration
        principal_axis_direction=principal_axis_direction,  # Crystal growth direction
        temperature=temperature,               # Simulation temperature
        k=number_eigenvalues,                  # Number of eigenvalues per k-point
        e_search=e_search,                     # Energy search center for eigenvalues
        shape_functions=shape_kind_kp,         # Shape functions for k·p Hamiltonian
        epot=p.epot,                          # Electrostatic potential field
        logger=logger,                         # Logger for this process
        rescaling=rescaling,                   # P-parameter rescaling for spurious solutions
        user_defined_params=user_defined_params  # Material parameter database
    )
    
    # Execute band structure calculation for this process's assigned k-points
    if rank == 0:
        logger.info("")
        logger.info('Solving k·p Hamiltonian for electronic band structure')
        
    bs.run()                          # Solve 8-band k·p eigenvalue problem
            
    # =========================================================================
    # MPI DATA COLLECTION AND SYNCHRONIZATION
    # =========================================================================
    
    # Collect and synchronize results from all MPI processes using collective communication
    # This phase gathers distributed k-point results into complete arrays
    neig = bs.bands.shape[1]          # Number of eigenvalues computed per k-point

    # Gather k-point values from all processes into complete array
    sendbuf = bs.kzvals               # This process's k-values
    recvbuf = np.zeros([nk])          # Buffer for complete k-value array
    comm.Allgather(sendbuf, recvbuf)  # MPI collective gather operation
    bs.kzvals = recvbuf               # Store complete k-value array
    
    # Gather band structure energies: shape (k-point, eigenvalue_index)
    sendbuf = bs.bands                # This process's band energies
    recvbuf = np.zeros([nk, neig])    # Buffer for complete band structure
    comm.Allgather(sendbuf, recvbuf)  # Collect all band energies
    bs.bands = recvbuf                # Store complete band structure
            
    # Gather spinor character distributions: shape (k-point, spinor_component, eigenvalue)
    sendbuf = bs.spinor_distribution  # This process's spinor distributions
    recvbuf = np.zeros([nk, 8, neig]) # Buffer for all spinor data (8 components in k·p)
    comm.Allgather(sendbuf, recvbuf)  # Collect spinor character from all processes
    bs.spinor_distribution = recvbuf  # Store complete spinor character data

    # Gather regional charge distributions: shape (k-point, region, eigenvalue)
    sendbuf = bs.norm_sum_region      # This process's regional distributions
    recvbuf = np.zeros([nk, nreg, neig])  # Buffer for all regional data
    comm.Allgather(sendbuf, recvbuf)  # Collect regional data from all processes
    bs.norm_sum_region = recvbuf      # Store complete regional distribution data
            
    # Gather electron envelope functions: shape (k-point, spatial_point, spinor_component, eigenvalue)
    sendbuf = bs.psi_el               # This process's electron envelopes
    recvbuf = np.zeros([nk, bs.psi_el.shape[1], bs.psi_el.shape[2], bs.psi_el.shape[3]], dtype='complex')
    comm.Allgather(sendbuf, recvbuf)  # Collect electron envelopes from all processes
    bs.psi_el = recvbuf               # Store complete electron envelope functions

    # Gather hole envelope functions: shape (k-point, spatial_point, spinor_component, eigenvalue)
    sendbuf = bs.psi_h                # This process's hole envelopes
    recvbuf = np.zeros([nk, bs.psi_h.shape[1], bs.psi_h.shape[2], bs.psi_h.shape[3]], dtype='complex')
    comm.Allgather(sendbuf, recvbuf)  # Collect hole envelopes from all processes
    bs.psi_h = recvbuf                # Store complete hole envelope functions
    
    # Clean up temporary communication buffers to free memory
    del sendbuf  
    del recvbuf

    # Log successful completion of data collection phase
    if rank == 0 and size > 1:
        logger.info("Band structure results collected from all MPI processes")
    
    # =========================================================================
    # CHARGE DENSITY CALCULATION
    # =========================================================================
    
    # Initialize free carrier charge density calculators for electrons and holes
    rho_el = FreeChargeDensity(bs.fs_el)  # Electron charge density calculator
    rho_h = FreeChargeDensity(bs.fs_h)    # Hole charge density calculator
                
    # Calculate electron charge density using envelope function approximation
    # Integrates over k-space and applies Fermi-Dirac statistics
    rho_el.add_charge(
        bs.psi_el,                     # Electron envelope functions (complex)
        np.array(bs.bands),            # Band structure energies for all k-points
        dk=dk,                         # k-point spacing for Brillouin zone integration
        mu=mu,                         # Chemical potential (Fermi level position)
        temp=temperature,              # Temperature for Fermi-Dirac distribution
        modified_EFA=modified_EFA,     # Use modified envelope function approximation
        particle=particle_s_components, # Particle type for s-orbital contributions
        norm_sum_region=bs.norm_sum_region, # Regional normalization factors
        thr_el=thr_el,                 # Electron character threshold for state inclusion
        thr_h=thr_h                    # Hole character threshold for state inclusion
    )

    # Calculate hole charge density using envelope function approximation
    # Similar to electron calculation but for valence band states
    rho_h.add_charge(
        bs.psi_h,                      # Hole envelope functions (complex)
        np.array(bs.bands),            # Band structure energies for all k-points
        dk=dk,                         # k-point spacing for Brillouin zone integration
        mu=mu,                         # Chemical potential (Fermi level position)
        temp=temperature,              # Temperature for Fermi-Dirac distribution
        modified_EFA=modified_EFA,     # Use modified envelope function approximation
        particle=particle_p_components, # Particle type for p-orbital contributions
        norm_sum_region=bs.norm_sum_region, # Regional normalization factors
        thr_el=thr_el,                 # Electron character threshold for state inclusion
        thr_h=thr_h                    # Hole character threshold for state inclusion
    )
    
    # =========================================================================
    # CHARGE BALANCE ANALYSIS
    # =========================================================================
    
    # Calculate total charges for system verification and validation
    ntot_el, ptot_el = rho_el.get_total_charge()  # Electrons: (negative, positive contributions)
    ntot_h, ptot_h = rho_h.get_total_charge()     # Holes: (negative, positive contributions)

    # Sum all charge contributions to assess overall charge balance
    ntot = ntot_el + ntot_h            # Total negative charge density (e/cm³)
    ptot = ptot_el + ptot_h            # Total positive charge density (e/cm³)
    total_charge = ntot + ptot         # Net charge (should ≈ 0 for neutral system)
    
    # Log charge balance analysis for system verification
    # Large charge imbalances may indicate numerical issues or physical inconsistencies
    if rank == 0:
        logger.info("")
        logger.info(f'Charge balance analysis')
        logger.info(f'{DLM}Total charge = {total_charge:.6e} e/cm³')
        if abs(total_charge) > 1e-10:
            logger.warning("Significant charge imbalance detected - Charge neutrality" + 
                           "expected in intrinsic systems")
            logger.warning('Check calculation parameters')

    # =========================================================================
    # DATA OUTPUT AND STORAGE
    # =========================================================================
    
    # Save numerical results to files for later analysis and plotting
    if rank == 0:
        
        logger.info("")
        logger.info('Saving numerical results to output files')
        
        # Save band structure energies: shape (k-point, eigenvalue_index)
        base_file_name = outdata_path + '/bands'
        np.save(base_file_name + '.npy', bs.bands)              # Binary format (efficient)
        if generate_txt_files:
            np.savetxt(base_file_name + '.txt', bs.bands, fmt='%.5g', delimiter=DLM)  # Text format (readable)
        logger.info(FMT_STR.format('Band structure energies', f'{directory_name}/bands(.npy/.txt)'))

        # TEMPORARY: Save test output in CSV format (remove in production)
        np.savetxt('./outdata/test.txt', bs.bands, fmt='%.5g', delimiter=',')

        # Save spinor character distributions: shape (k-point, spinor_component, eigenvalue)
        base_file_name = outdata_path + '/spinor_dist'
        np.save(base_file_name + '.npy', bs.spinor_distribution)  # Binary format
        if generate_txt_files:
            # Save each k-point as separate text file due to 3D array structure
            for i in range(bs.spinor_distribution.shape[0]):
                slice_2d = bs.spinor_distribution[i]
                np.savetxt(f'{base_file_name}_{i}.txt', slice_2d, fmt='%.5g', delimiter=DLM)  
        logger.info(FMT_STR.format('Spinor character distribution', f'{directory_name}/spinor_dist(.npy/_I.txt)'))        

        # Save regional charge distributions: shape (k-point, region, eigenvalue)
        base_file_name = outdata_path + '/norm_sum_region'
        np.save(base_file_name + '.npy', bs.norm_sum_region)     # Binary format
        if generate_txt_files:
            # Save each k-point as separate text file due to 3D array structure
            for i in range(bs.norm_sum_region.shape[0]):
                slice_2d = bs.norm_sum_region[i]
                np.savetxt(f'{base_file_name}_{i}.txt', slice_2d, fmt='%.5g', delimiter=DLM)  
        logger.info(FMT_STR.format('Regional charge distribution', f'{directory_name}/norm_sum_region(.npy/_I.txt)'))
                
        # Save envelope functions: shape (k-point, spatial_point, spinor_component, eigenvalue)
        # These are the complete spatial wavefunctions (complex-valued)
        # Note: Only saved in binary format due to complex values and large file sizes
        base_file_name = outdata_path + '/envelope_el'
        np.save(base_file_name + '.npy', bs.psi_el)            # Electron envelopes
        logger.info(FMT_STR.format('Electron envelope functions', f'{directory_name}/envelope_el(.npy)'))
        
        base_file_name = outdata_path + '/envelope_h'
        np.save(base_file_name + '.npy', bs.psi_h)             # Hole envelopes
        logger.info(FMT_STR.format('Hole envelope functions', f'{directory_name}/envelope_h(.npy)'))
    
        # Save charge balance verification data
        base_file_name = outdata_path + '/total_charge'
        np.save(base_file_name + '.npy', total_charge)         # Binary format
        if generate_txt_files:
            np.savetxt(base_file_name + '.txt', [total_charge]) # Text format
        logger.info(FMT_STR.format('Total charge balance', f'{directory_name}/total_charge(.npy/.txt)'))
        
        # Save overlap matrix in sparse format for memory efficiency
        save_npz(outdata_path + "/B.npz", bs.solver[0].bgl)
        logger.info(FMT_STR.format('Overlap matrix (sparse)', f'{directory_name}/B.npz'))
        
    # =========================================================================
    # VISUALIZATION AND PLOTTING
    # =========================================================================

    # Generate visualization plots if requested by user
    if generate_png_graphs and rank == 0:
        plot_production(bs,p,rho_el,rho_h)
        
    # =========================================================================
    # CALCULATION COMPLETION
    # =========================================================================

    # Log successful completion of the entire calculation
    if rank==0:
        execution_successful()             
             
# =============================================================================
# SCRIPT EXECUTION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    """
    Script entry point with timing instrumentation.
    """
    
    # Execute main calculation with high-resolution timing instrumentation
    tic()                              # Start precision timer (from nwkpy)
    main()                             # Execute complete band structure calculation
    toc()                              # Stop timer and display total elapsed time