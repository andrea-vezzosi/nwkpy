"""
Input Parameters for Self-Consistent Core-Shell Nanowire Band Structure Calculations

This configuration file contains all parameters needed for self-consistent 
Schrödinger-Poisson calculations of core-shell nanowire band structures with 
multiple chemical potentials and external electric fields.

Key Features:
    - Multi-parameter sweeps over chemical potentials and electric fields
    - Self-consistent coupling between electronic structure and electrostatics
    - Broyden mixing for accelerated convergence
    - Advanced convergence monitoring and control

Parameter Categories:
    - File Configuration: Output directories and mesh files
    - Material Properties: Core and shell material specifications
    - Crystal Structure: Lattice parameters and growth direction
    - Simulation Conditions: Temperature ranges
    - Multi-Parameter Sweeps: Chemical potentials and electric fields
    - Band Structure: Energy ranges and k-point sampling
    - Self-Consistent Cycle: Convergence criteria and iteration control
    - Numerical Methods: Shape functions and mixing parameters
    - Advanced Features: EFA modifications and thresholds
    - Boundary Conditions: Poisson equation constraints
    - Plotting: Visualization preferences for all outputs
"""

import numpy as np                # Importing numpy for numerical operations
                                  # e.g. you can use np.pi for π in calculations

# =====================
# FILE CONFIGURATION
# =====================

# Output directory where all results will be saved
directory_name = "outdata"        # Directory where all output files will be saved
                                  # Created automatically if it doesn't exist
                                  # Hierarchical structure: outdata/OUT_EF_X/OUT_CP_Y/

# Mesh file specification
mesh_name = "mesh"                # Base name of the mesh file to be generated
                                  # .msh and .dat extensions will be added automatically
                                  # Requires pre-generated GMSH mesh files:
                                  # - .msh: Finite element mesh geometry
                                  # - .dat: Mesh metadata and region information

# =====================
# EXECUTION CONTROL FLAGS
# =====================

# Data output control
generate_txt_files = False         # If True, save results in human readable format (.txt)
                                  # in addition to binary format (.npy) for some files

generate_png_graphs = False        # If True, generate plots in .png format

# Debugging and development options
MPI_debug = False                 # If True, enable MPI debugging output
                                  # Creates separate debug files for each MPI process

# =====================
# MATERIAL PROPERTIES
# =====================

# Core-shell material specification
material = ["InAs+", "GaSb"]       # Materials for [core, shell] of the nanowire
                                  # Must exist in the nwkpy material parameter database
                                  # Use same material for homogeneous samples

# Band alignment and energy references
valence_band = [0.0, 0.56]        # Valence band edge offsets in eV for [core, shell]
                                  # Sets the energy reference for each material
                                  # Difference determines band offset (Type I/II alignment)

user_parameters_file = "user_parameters.dict"  # file with user-defined parameters (if any)
                                  # If None, internal database is used
                                  # Keys should match those in material list

# =====================
# CRYSTAL STRUCTURE
# =====================

# Crystallographic configuration
principal_axis_direction = '111'  # Nanowire growth direction (crystallographic axis)
                                  # Options: '100', '110', '111'

lattice_constant = 6.0583         # Lattice constant in Angstroms (InAs value used for scaling)

# =====================
# SIMULATION CONDITIONS
# =====================

# Thermodynamic conditions
temperature = 4.0                 # Simulation temperature in Kelvin (affects energy gaps)

# =====================
# CARRIER TYPES
# =====================

# Dominant carrier specification
carrier = ["electron", "hole"]    # Dominant carrier type for [core, shell] materials
                                  # Important for broken-gap heterostructures
                                  # Affects charge density calculations and plotting

# =====================
# MULTI-PARAMETER SWEEPS
# =====================

# Chemical potential sweep - multiple values for systematic study
chemical_potential_set = [0.539144444444444, 0.55]  # Chemical potentials in eV
                                  # Each value will be calculated self-consistently
                                  # Zero energy reference is set by valence_band_edges
                                  # Determines carrier concentrations via Fermi-Dirac statistics

# External electric field sweep - multiple field configurations
# Creates separate output directories for each field
electric_field_set = [
    (0.0, np.pi/2.) ,             # No field: (magnitude in V/μm, angle with x-axis in radians)
    (0.2, np.pi/2.)               # Field of 0.2 V/μm in y-direction
]                               
    
# =====================
# BAND STRUCTURE PARAMETERS
# =====================

# Eigenvalue search configuration
e_search = 0.539144444444444     # Center energy for eigenvalue search in eV
                                 # Should be near the chemical potential range

number_eigenvalues = 20          # Number of eigenvalues (subbands) to compute
                                 # Should include states near Fermi level

# k-space sampling configuration
k_range = [0, 0.05]              # k-space range [initial, final] - only positive values
                                 # Negative values included automatically by symmetry
                                 # Units: relative to π/√3/lattice_constant

number_k_pts = 96                # Number of k-points for band structure calculation
                                 # Must be divisible by number of MPI processes

# =====================
# SELF-CONSISTENT CYCLE PARAMETERS
# =====================

# Initial conditions
init_pot_name = None             # Path to initial potential file (None = solve Poisson first)
                                 # If provided, loads pre-computed electrostatic potential
                                 # Format: NumPy .npy file with potential values
                                 # Useful for restart calculations or parameter continuations

# Convergence criteria
maxiter = 20                     # Maximum number of self-consistent iterations

maxchargeerror = 1e-3            # Convergence threshold for charge density (relative)
                                 # Calculation stops when charge density changes
                                 # by less than this fraction between iterations

maxchargeerror_dk = 1e-3         # Convergence threshold for charge in dk (legacy parameter)
                                 # Maintained for compatibility with older versions

# =====================
# SPURIOUS SOLUTION SUPPRESSION
# =====================

# k·p model refinement parameters
rescale = ['S=0', 'S=0']         # Rescaling method for [core, shell] materials
                                 # Suppresses spurious high-energy solutions in 8-band k·p
                                 # Reference: B. A. Foreman, Phys. Rev. B 56, R12748 (1997)
                                 # Options:
                                 # - 'S=0': Standard Ep evaluation (Eq. 6.158)
                                 # - 'S=1': Modified Ep evaluation (Eq. 6.159)  
                                 # - Numerical value: Fractional Ep reduction (e.g., 0.26)

# =====================
# ADVANCED FEATURES
# =====================

# Envelope Function Approximation (EFA) settings
modified_EFA = True              # Use Modified Envelope Function Approximation
                                 # Recommended: True for broken-gap structures
                                 # Improves treatment of mixed electron-hole states

# State classification thresholds
character_threshold = [0.8, 0.95]  # [electron, hole], used with modified_EFA = True
                                   # States with character > threshold contribute to densities
                                   # Range: 0.0 (all states) to 1.0 (pure states only)
                                   # Higher values = more selective inclusion

# =====================
# NUMERICAL METHODS
# =====================

# Finite element shape function selection
shape_function_kp = ['Hermite', 'LagrangeQuadratic']    # for kp Hamiltonian, [electrons, holes]
                                                        # Use to
                                                        # - optimize accuracy vs cost
                                                        # - ghost band avoidance strategy

shape_function_poisson = 'LagrangeQuadratic'            # For Poisson equation

# =====================
# BROYDEN MIXING PARAMETERS
# =====================

# Advanced mixing scheme for accelerating self-consistent convergence
betamix = 0.35                   # Simple mixing parameter (0 < betamix < 1)
                                 # Lower values = more conservative mixing
                                 # Higher values = faster but potentially unstable convergence
                                 # Typical range: 0.1-0.5

maxter = 6                       # Maximum number of previous iterations to store
                                 # Higher values = better convergence but more memory

w0 = 0.01                        # Weight for m=0 iteration in Broyden scheme
                                 # Controls influence of initial iteration
                                 # Typical range: 0.01-0.1

use_wm = True                    # Whether to use iteration-dependent weights for m>0
                                 # True: More sophisticated weighting scheme
                                 # False: Uniform weighting

toreset = []                     # List of iteration numbers where to reset Broyden history
                                 # Empty list = no resets
                                 # Example: [5, 10] resets mixing at iterations 5 and 10
                                 # Useful for difficult convergence cases

# =====================
# BOUNDARY CONDITIONS FOR POISSON EQUATION
# =====================

# Electrostatic boundary condition specification
dirichlet = {                    # Boundary conditions fix electron potential energy -e·φ in eV
    'ref': None,                 # Use Neumann BC as default (zero normal electric field)
    1: 0.0,                      # Fix potential to 0.0 eV on boundary 1 (ground contact)
}
                                 # Examples of different boundary conditions:
                                 # - {'ref': 0.0}: Zero potential on all boundaries
                                 # - {'ref': 0.0, 1: 0.1}: Zero on all except border 1 (0.1 eV)
                                 # - {'ref': None, 1: 0.0}: Zero on border 1, Neumann on others
                                 # - None: Pure Neumann BC (mean potential = 0)

# =====================
# PLOTTING PREFERENCES
# =====================

# Configuration for energy band dispersion plots
plotting_preferencies_bands = {
    'xlim': (0, 0.06),           # k-space range in nm⁻¹ for x-axis
    
    'ylim': (528, 545),          # Energy range in meV for y-axis
    
    'chemical_potential': 539.1444444,  # Chemical potential line in meV
                                        # Displayed as horizontal reference line
                                        # Should match one of the computed values
    
    'cmap_in': 'rainbow',        # Colormap for band character visualization
                                 # Options: 'viridis', 'plasma', 'rainbow', 'coolwarm'
                                 # Colors indicate electron/hole character
    
    'character_to_show': 'H-EL', # Character type to display in colors
                                 # 'H-EL': Heavy hole - Electron character
                                 # Other options: 'LH-EL', 'SO-EL' for different bands
    
    'loc_cbar': 1,               # Colorbar location (matplotlib LocationError codes)
                                 # 1 = upper right, 2 = upper left, etc.
    
    'spines_lw': 4,              # Plot border line width for publication quality

    'lw': 5,                     # Line width for energy band curves

    'fontsize': 20               # Font size for labels and tick marks
}

# Configuration for 2D charge density contour plots
plotting_preferencies_density = {
    'subdiv': 1,                 # Mesh subdivision level for smoother plotting
                                 # 0 = original mesh, 1 = refined once, 2 = refined twice
    
    'figsize': (5, 5),           # Figure size in inches (width, height)
                                 # Use consistently with potential plots for comparison
    
    'xlim': (-15, 15),           # x-axis range in nm for cross-sectional view
                                 # Should cover the nanowire and surrounding region
    
    'ylim': (-15, 15),           # y-axis range in nm for cross-sectional view
                                 # Should cover the nanowire and surrounding region
    
    'cmapin': 'rainbow',         # Colormap for charge density visualization
                                 # Options: 'viridis', 'plasma', 'rainbow'
    
    'levels': 21,                # Number of contour levels for density plot
                                 # Odd numbers include zero contour line
    
    'fontsize': 20,              # Font size for labels and colorbars
    
    'polygons': None             # Additional polygons to overlay (e.g., interfaces)
                                 # None = no overlays
                                 # Can mark core-shell interface or device regions
}

# Configuration for electrostatic potential contour plots
plotting_preferencies_potential = {
    'subdiv': 0,                 # Mesh subdivision level for plotting
                                 # 0 = original mesh elements (fastest)
    
    'figsize': (5, 5),           # Figure size in inches (width, height)
                                 # Use consistently with density plots for comparison
    
    'xlim': (-15, 15),           # x-axis range in nm for cross-sectional view
                                 # Should cover the nanowire and surrounding region
    
    'ylim': (-15, 15),           # y-axis range in nm for cross-sectional view
                                 # Should cover the nanowire and surrounding region
    
    'cmapin': 'rainbow',         # Colormap for potential visualization
                                 # 'rainbow': Full spectrum, good for large ranges
                                 # 'RdBu': Red-blue diverging, good for +/- potentials
    
    'levels': 21,                # Number of contour levels for potential plot
                                 # Odd numbers include zero contour line
    
    'fontsize': 20,              # Font size for axis labels and colorbar
    
    'polygons': None             # Additional geometric overlays
                                 # None = clean plot, or list of shapes to highlight
                                 # Can mark interfaces, contacts, or device regions
}