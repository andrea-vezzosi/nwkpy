# =============================================================================
# HEXAGONAL CORE-SHELL NANOWIRE MESH GENERATION - INPUT PARAMETERS
# =============================================================================

# This file contains all the parameters needed to generate a hexagonal 
# finite element mesh for core-shell nanowire structures using FreeFem++

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Directory where all generated files will be saved
directory_name = './outdata'        # relative to the script location

# Base name of the mesh file to be generated
mesh_name = "mesh"                  # .msh and .dat will be added automatically

# =============================================================================
# MATERIAL CONFIGURATION
# =============================================================================

# Materials for each region of a core-shell structure
# NOTE Material names do not matter here, only used  for bookkeeping
material = ["InAs", "GaSb"]          # Order: [core material, shell material]

# Dominant carrier types for each material region
# Options: "electron", "hole"
# NOTE Material names do not matter here, only used  for bookkeeping
carrier = ["electron", "hole"]       # Order: [core carrier, shell carrier]

# =============================================================================
# GEOMETRIC PARAMETERS
# =============================================================================

# Physical dimensions of the nanowire cross-section in nanometers
width = [7.5, 4.88]  # [nm]          # Order: [core width, shell width]

# =============================================================================
# MESH DISCRETIZATION PARAMETERS
# =============================================================================

# Number of mesh points for each segment of the irreducible wedge of an hexagonal section. 
# 
# Border numbering follows the hexagonal geometry:
#   - Borders 1-6 correspond to the six segments of a wedge (1/12 of hexagon)
#   - Each border can have different mesh densities based on geometric requirements
#   - Typical values range from 5-20 depending on desired accuracy
#  # 
#            / |
#           /  |    NOTE: The origin (0,0) of the coordinate system 
#          4   3          is bottom-left corner
#         /    |    
#        /--6--|
#       5      2
#      /       | 
#   (0,0)---1--- 
# 
edges = [10, 5, 7, 6, 5, 5]          # Order: [border_1, ... , border_6]

# =============================================================================
# USAGE NOTES
# =============================================================================

# This configuration will create a hexagonal mesh with:
# - Inner core region (Region 1): InAs material, electron-dominated, radius 7.5 nm
# - Outer shell region (Region 2): GaSb material, hole-dominated, thickness 4.88 nm
# - Total structure diameter: ~24.76 nm
# - Variable mesh density on each hexagonal border for optimization
#
# The generated mesh will be compatible with:
# - Band structure calculations using k·p method
# - Electrostatic potential calculations
# - Charge density calculations
# - Multi-physics nanowire simulations