"""
Utility Functions for scripts based on the nwkpy library.

This module provides essential utility functions for logging, error handling,
and system information display to ensure consistency and reusability across different scripts.

Functions:
    execution_aborted: Handle fatal errors and exit gracefully
    execution_successful: Log successful completion
    host_IP: Log hostname and IP for distributed debugging
    print_header: Create formatted header for the script
    log_material_params: Display material parameters
    log_system_configuration_summary: Show system config in tabular form
    log_physical_parameters: Display physical simulation parameters
    log_electrostatic_potential_configuration: Show Poisson equation setup
    log_band_structure_calculation_parameters: Display k-point and eigenvalue settings
    log_computational_system_information: Show MPI and debugging configuration
    log_broyden_mixing_parameters: Log parameters for Broyden mixing.
    log_self_consistent_cycle_parameters: Log parameters for the self-consistent cycle.
    get_parameters: Get parameters for the k.p Hamiltonian of the specified materials.
"""

from datetime import datetime            # Date/time stamps for logs
import logging                          # For structured log output
import socket                           # For network debugging information
# from config import *                    # Local configuration constants
from nwkpy.config import *
from mpi4py import MPI
import sys                              # System-specific parameters and functions

from nwkpy._database import params      # Comprehensive material parameter database

# Initialize module logger for consistent formatting
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def execution_aborted(e,rank=None):
    """
    Handle execution errors by logging the exception and exiting gracefully.
    This function is called when a critical error occurs, preventing further calculations.

    Args:
        e (Exception or str): The exception or error message that caused the execution to abort
        
    Note:
        - All MPI processes must reach this function for proper termination
        - Only rank 0 logs detailed error messages to avoid duplicates
        - Uses sys.exit(1) to indicate error termination
    """
    
    # Only rank 0 logs the error (rank is global variable)
    if rank == 0:
        logger.error(f"{str(e)}")
        logger.error(f"Execution aborted.")
    
    # ALL processes exit with error code
    sys.exit(1)

def execution_successful(rank=None):
    """
    Log successful completion of the simulation.
    
    Provides consistent success messaging both to log files and console output.
    Used as the final step in successful calculation runs.
    
    NOTE: All MPI processes must reach this function for proper termination.
    """
    
    # Only rank 0 logs the success message (rank is global variable)
    if rank == 0:
        logger.info("")
        logger.info('Normal successful completion.')
        print('Normal successful completion.')
    
    # ALL processes exit here - this is the key fix
    sys.exit(0)
    
def host_IP():
    """
    Log hostname and IP address for debugging distributed calculations.
    
    Note:
        Gracefully handles network resolution failures with warning messages
        rather than crashing the simulation.
    """
    try:
        # Get local hostname and resolve to IP address
        hname = socket.gethostname()
        hip = socket.gethostbyname(hname)
        logger.info(f'{DLM}Hostname       : {hname}')
        logger.info(f"{DLM}IP Address     : {hip}")
    except Exception:
        # Network resolution can fail in some cluster environments
        logger.warning("Unable to get hostname and IP address")

def print_header(script_name):
    """
    Print formatted header with script name and execution date.
    
    Creates a centered, bordered header for log files to clearly
    identify calculation runs. 
      
    Args:
        script_name (str): Name of the script being executed
        
    """
    # Header formatting constants
    BAR_LENGTH = 80                      # Total width of header bar
    
    # Get current date for logging
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    # Calculate header bar length and center the script name
    title_length = len(script_name)
    shift = (BAR_LENGTH - title_length) // 2  # Center alignment offset
    
    # Print the header with centered script name and date
    logger.info('=' * BAR_LENGTH)
    logger.info(f'{" " * shift}{script_name}')
    logger.info(f'{" " * shift}Running on {current_date}')
    logger.info('=' * BAR_LENGTH)

def log_material_params(material_name, material_params):
    """
    Log material parameters in a formatted table structure.
    
    Displays material parameters in organized rows for easy reading
    and verification of input values. Groups related parameters
    (band structure, Luttinger, and other physical parameters)
    on separate lines for clarity.
    
    Args:
        material_name (str): Name of the material (e.g., 'GaAs', 'InAs')
        material_params (dict): Dictionary of material parameters from database
    """
    # Column width for consistent parameter display formatting
    cell_width = 10                      # Column width for parameter display

    # Group parameters for organized display
    line1_params = ['Eg', 'delta', 'Ep', 'me', 'lu1', 'lu2', 'lu3']  # Band parameters
    line2_params = ['L', 'M', 'Nm', 'Np', 'N', 'P', 'Ac']            # Luttinger parameters
    line3_params = ['B', 'eps', 'alc', 'alTpar', 'alpha', 'beta']     # Other parameters
    
    # Log material name with spacing for readability
    logging.info("")
    logging.info(f'Material : {material_name}')
    logging.info(f'Note     : {material_params.get("note", "")}')

    # Format each line of parameters with consistent spacing
    line1_items = [f"{key}={value}".ljust(cell_width) 
                   for key, value in material_params.items() if key in line1_params]
    line2_items = [f"{key}={value}".ljust(cell_width) 
                   for key, value in material_params.items() if key in line2_params]
    line3_items = [f"{key}={value}".ljust(cell_width) 
                   for key, value in material_params.items() if key in line3_params]
    
    # Join parameters with separators for table-like appearance
    formatted_params_line1 = " | ".join(line1_items)
    formatted_params_line2 = " | ".join(line2_items)
    formatted_params_line3 = " | ".join(line3_items)
    
    # Log the formatted parameter lines with indentation
    logging.info(f'   {formatted_params_line1}')
    logging.info(f'   {formatted_params_line2}')
    logging.info(f'   {formatted_params_line3}')

def log_system_configuration_summary(material,valence_band,rescale,carrier,shape_function_kp,character_threshold):
    """
    Log physical parameters of the nanowire and k.p configuration in tabular form.
    
    Creates a comprehesive summary table showing the configuration differences
    between core and shell materials. 
    
    Args:
        material (list): Material names for [core, shell]
        valence_band (list): Valence band edge offsets in eV for [core, shell]
        rescale (list): P-parameter rescaling methods for [core, shell]
        carrier (list): Dominant carrier types for [core, shell]
        shape_function_kp (list): Shape functions for [electrons, holes]
        character_threshold (list): Character thresholds for [electrons, holes]
        
    """
    logger.info("")
    logger.info('System Configuration Summary')
    logger.info("----------------------------")
    # Table header with column descriptions
    logger.info('       Material   Valence Edge (eV)  Rescaling   Carrier type   Shape fct           Char. thres.')
    logger.info( '-' * 88)
    # Core material configuration row
    logger.info(f'core   {material[0]:<10} {valence_band[0]:<18} {rescale[0]:<11} {carrier[0]:<14} {shape_function_kp[0]:<19} {character_threshold[0]:<5}')
    # Shell material configuration row
    logger.info(f'shell  {material[1]:<10} {valence_band[1]:<18} {rescale[1]:<11} {carrier[1]:<14} {shape_function_kp[1]:<19} {character_threshold[1]:<5}')
    logger.info('-' * 88)

def log_physical_parameters(temperature,chemical_potential,principal_axis_direction):
    """
    Log physical parameters for the simulation.
    
    Args:
        temperature (float): Simulation temperature in Kelvin
        mu (float): Chemical potential (Fermi level) in eV
        principal_axis_direction (str): Crystallographic growth direction ('100', '110', '111')
    """
    logger.info("")
    logger.info('Physical Parameters')
    logger.info(f'{DLM}Temperature           : {temperature} K')
    if isinstance(chemical_potential, list):
        potentials = ', '.join(f"{mu:.5f}" for mu in chemical_potential)
    else:
        potentials = f"{chemical_potential:.5f}"
    logger.info(f'{DLM}Chemical potential(s) : {potentials} eV')
    logger.info(f'{DLM}Growth direction      : {principal_axis_direction}')

def log_electrostatic_potential_configuration(shape_function_poisson,dirichlet,electric_field,init_pot_name):
    """
    Log electrostatic potential calculation configuration enterning Poisson equation.
    
    Args:
        shape_function_poisson (str): Shape function type for Poisson equation
        dirichlet (dict): Dirichlet boundary condition specification
        electric_field (tuple): External electric field (magnitude, angle)
        init_pot_name (str or None): Path to initial potential file, if any
    """
    logger.info("")
    logger.info('Electrostatic Potential Configuration')
    if isinstance(electric_field, list):
        fields = ' '.join(f"[{x:.4f}, {y:.4f}]" for x, y in electric_field)
    else:
        fields = f"[{electric_field[0]:.4f}, {electric_field[1]:.4f}]"
    logger.info(f'{DLM}Electric field(s) [V/m, rad]     : {fields}')
    logger.info(f'{DLM}Initial potential file           : {init_pot_name if init_pot_name else "None"}')
    logger.info(f'{DLM}Dirichlet boundary conditions    : {dirichlet}')
    logger.info(f'{DLM}Poisson shape function           : {shape_function_poisson}')

def log_band_structure_calculation_parameters(number_eigenvalues,e_search,kzvals,number_k_pts):
    """
    Log band structure calculation parameters used in kp calculations.
    
    Args:
        k (int): Number of eigenvalues to compute per k-point
        e_search (float): Energy center for eigenvalue search in eV
        kzvals (array): Array of k-point values along nanowire axis
        number_k_pts (int): Total number of k-points in calculation
    """
    logger.info("")
    logging.info('Band Structure Calculation Parameters')
    logging.info(f'{DLM}Number of eigenvalues : {number_eigenvalues}')
    logging.info(f'{DLM}Energy search center  : {e_search:<.5f} eV')
    # logging.info(f'{DLM}K-point range         : [{kzvals[0]:.5f}, {kzvals[number_k_pts-1]:.5f}] Å⁻¹')
    logging.info(f'{DLM}K-point range         : [{10*kzvals[0]:.5f}, {10*kzvals[number_k_pts-1]:.5f}] nm⁻¹')
    logging.info(f'{DLM}K-points grid         : {number_k_pts} k-points')
    # logging.info(f'{DLM}dk                    : {kzvals[1] - kzvals[0]:.5f} Å⁻¹')
    logging.info(f'{DLM}dk                    : {10*(kzvals[1] - kzvals[0]):.5f} nm⁻¹')

def log_computational_system_information(size,kmaxlocal,MPI_debug):
    """
    Log computational system configuration and MPI setup.
    
    Args:
        size (int): Total number of MPI processes
        outdata_path (str): Path to output directory
        MPI_debug (bool): Whether MPI debugging is enabled
        
    Side Effects:
        - Calls host_IP() to log network information
        - May call MPI_debug_setup() if debugging is enabled
    """
    logger.info("")
    logging.info('Computational System Configuration')
    host_IP()                      # Log hostname and IP for debugging
    logger.info(f'{DLM}MPI processes  : {size}')
    logger.info(f"{DLM}Points/process : {kmaxlocal}")
    logger.info(f'{DLM}MPI_debug      : {MPI_debug}')
        
        
def log_broyden_mixing_parameters(betamix, maxter, w0, use_wm, toreset):
    """
    Log parameters for Broyden mixing.

    Args:
        betamix (float): Mixing parameter for Broyden method
        maxter (int): Maximum number of stored iterations
        w0 (float): Weight parameter for mixing
        use_wm (bool): Whether to use weight matrix
        toreset (bool): Whether to reset iterations
    """
    logger.info("")
    logger.info("Broyden mixing parameters")
    logger.info(f'{DLM}Mixing parameter (beta)       : {betamix}')
    logger.info(f'{DLM}Maximum stored iterations (M) : {maxter}')
    logger.info(f'{DLM}Weight parameter (w0)         : {w0}')
    logger.info(f'{DLM}Use weight matrix             : {use_wm}')
    logger.info(f'{DLM}Reset iterations              : {toreset}')

def log_self_consistent_cycle_parameters(maxiter, maxchargeerror):
    """
    Log parameters for the self-consistent cycle.

    Args:
        maxiter (int): Maximum number of iterations
        maxchargeerror (float): Convergence criterion for charge density
    """
    logger.info("")
    logger.info("Self-consistent cycle parameters")
    logger.info(f'{DLM}Maximum iterations    : {maxiter}')
    logger.info(f'{DLM}Convergence criterion : {maxchargeerror}')

def get_parameters(material):
    """
    Get parameters for the k.p Hamiltonian of the specified materials.

    Args:
        material (list): List of material names (e.g., ['InAs', 'GaSb'])

    returns:
        dict: Dictionary of material parameters
    """

    # set parameters dictionary
    parameters = {}                 

    # # set parameters from user input (priority)
    # if user_params is not None:
    #     for m in material:
    #         parameters[m] = user_params[m]
    # # set parameters from internal database
    # else:
    #     for m in material:
    #         parameters[m] = params[m]   

    for m in material:
        parameters[m] = params[m]
    return parameters