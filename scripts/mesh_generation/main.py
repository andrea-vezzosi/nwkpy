#!/usr/bin/env python

"""
Hexagonal Core-Shell Nanowire Mesh Generator

This script generates finite element meshes for core-shell nanowire structures
using FreeFem++ through the nwkpy interface. The generated meshes are optimized 
for electronic structure calculations and electrostatic simulations.

Key Features:
- Hexagonal cross-section geometry 
- One- (core) or two-region (core + shell) structures 
- Symmetry compliant mesh design for computational efficiency
- Customizable mesh density along different borders
- Compatible with k·p band structure calculations in library nwkpy

Output:
- .msh file in FreeFem++ format
- .dat file with mesh metadata
- Detailed mesh statistics and verification
- Logged information for reproducibility
"""

# =============================================================================
# GENERAL IMPORTS AND SETUP
# =============================================================================

import sys
import os
import numpy as np
import logging

# =============================================================================
# CORE LIBRARY IMPORTS
# =============================================================================

from nwkpy.fem import Hex2regsymm, Mesh
from nwkpy import tic, toc
from nwkpy import library_header

# =============================================================================
# LOCAL IMPORTS 
# =============================================================================

from nwkpy.utilities import *
from nwkpy.config import *

# =============================================================================
# SCRIPT PARAMETERS
# =============================================================================

SCRIPT_NAME = 'Hexagonal Core-Shell Nanowire Mesh Generation'

# =============================================================================
# IMPORT INPUT PARAMETERS
# =============================================================================

from indata import *

# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================

cdir = os.getcwd()                                      # current directory
outdata_path = os.path.join(cdir, directory_name, '')   # output directory
os.makedirs(outdata_path, exist_ok=True)                # create output directory
log_file = os.path.join(outdata_path, LOG_FILE_NAME + ".log") # log file

# =============================================================================
# CONFIGURE LOGGING SYSTEM
# =============================================================================

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    filename=log_file,
    datefmt='%H:%M:%S', # log time format
    filemode='w',       # overwrite log file if existent
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print(f'\nAll log messages sent to file: {log_file}\n')

# =============================================================================
# MESH FILES
# =============================================================================

msh = mesh_name + ".msh"
dat = mesh_name + ".dat"
mesh_file = os.path.join(outdata_path, msh)
mesh_data = os.path.join(outdata_path, dat)

# =============================================================================
# INPUTS CONSISTENCY CHECKS
# =============================================================================

def consistency_checks():
    """
    Perform comprehensive consistency checks on input parameters.
    """
    # Check materials
    if len(material) != 2:
        logger.warning(f"material = {material} should contain exactly 2 materials")
        raise ValueError("Invalid material - Provide exactly 2 materials [core, shell]")

    # Check carriers
    if any(c not in CARRIER_TYPES for c in carrier):
        logger.warning(f"carrier = {carrier} not available")
        logger.warning(f"Valid carrier types: {CARRIER_TYPES}")
        raise ValueError("Invalid carrier type - Choose valid carrier types")

    # Check dimensions
    if len(width) != 2 or any(w <= 0 for w in width):
        logger.warning(f"width = {width} invalid")
        raise ValueError("Invalid width - Provide [core_radius, shell_thickness] with positive values")

    # Check edges
    if len(edges) != 6 or any(e < 1 for e in edges):
        logger.warning(f"edges = {edges} invalid")
        raise ValueError("Invalid edges - Provide exactly 6 edge counts ≥ 1")

# =============================================================================
# MAIN MESH GENERATION FUNCTION
# =============================================================================

def main():
    """
    Main mesh generation function.
    """
    
    # =============================================================================
    # HEADER AND INITIALIZATION
    # =============================================================================

    print_header(SCRIPT_NAME)   # log header with date
    library_header()            # log library info

    try:
        consistency_checks()    # perform input consistency checks
    except ValueError as e:
        execution_aborted(e)    # abort execution on input error
    else:
        logger.info("")
        logger.info(f'Input parameter consistency checks passed')

    # =============================================================================
    # PHYSICAL SYSTEM CONFIGURATION
    # =============================================================================

    reg2mat = {
        1: material[0],
        2: material[1]
    }
    mat2partic = {
        material[0]: carrier[0],
        material[1]: carrier[1]
    }

    R_c = width[0]                          # core radius
    shell_width = width[1]                  # shell thickness
    total_width = (R_c + shell_width) * 2   # total diameter

    edges_per_border = {
        'nC1': edges[0],
        'nC2': edges[1],
        'nC3': edges[2],
        'nC4': edges[3],
        'nC5': edges[4],
        'nC6': edges[5]
    }

    # =============================================================================
    # CONFIGURATION LOGGING
    # =============================================================================
    
    logger.info("")
    logger.info('Configuration Summary')
    logger.info("---------------------")
    logger.info('              Material     Carrier      Dimension [nm]')
    logger.info('-' * 66)
    logger.info(f"1 (core)  :   {material[0]:<12} {carrier[0]:<12} {R_c:<12.2f} (radius)")
    logger.info(f"2 (shell) :   {material[1]:<12} {carrier[1]:<12} {shell_width:<12.2f} (thickness)")
    logger.info('-' * 66)
    logger.info('')
    logger.info(f"Total nanowire diameter: {total_width:.2f} nm")

    logger.info("")
    logger.info('Mesh Edge Distribution:')
    logger.info(f"{DLM}Border Index : " + " ".join(f"{i+1:>3}" for i in range(len(edges))))
    logger.info(f"{DLM}Edge Count   : " + " ".join(f"{v:>3}" for v in edges))

    # =============================================================================
    # MESH GENERATION
    # =============================================================================
    
    logger.info("")
    logger.info('Starting mesh generation')

    try:
        Hex2regsymm(
            mesh_name=mesh_file,
            total_width=total_width,
            shell_width=shell_width,
            edges_per_border=edges_per_border
        )
        logger.info('Mesh generation completed successfully')
    except Exception as e:
        execution_aborted(e)

    # =============================================================================
    # MESH METADATA
    # =============================================================================

    with open(mesh_data, 'w') as f:
        f.write('Materials : ' + ' '.join(material) + '\n')
        f.write('Carriers  : ' + ' '.join(carrier) + '\n')
        f.write('Width     : ' + ' '.join(str(w) for w in width) + '\n')
        f.write('Edges     : ' + ' '.join(str(e) for e in edges) + '\n')

    # =============================================================================
    # MESH VERIFICATION
    # =============================================================================

    logger.info("")
    logger.info('Loading mesh for verification')
    
    try:
        mesh = Mesh(
            mesh_name=mesh_file,
            reg2mat=reg2mat,
            mat2partic=mat2partic
        )
        logger.info('Mesh loaded successfully')
    except Exception as e:
        execution_aborted(e)

    # Log mesh statistics
    logger.info("")
    logger.info('Mesh Statistics:')
    logger.info(f"{DLM}Total vertices (nodes): {mesh.ng_nodes}")
    logger.info(f"{DLM}Total elements: {mesh.nelem}")
    logger.info(f"{DLM}Boundary edges: {len(mesh.e_l)}")

    # Log boundary analysis
    logger.info("")
    logger.info('Boundary Analysis:')
    logger.info(f"{DLM}Boundary labels: {mesh.border_labels}")
    border_edges = [str(np.count_nonzero(mesh.e_l == (i+1))) for i in range(6)]
    logger.info(f"{DLM}Border edges: " + " ".join(f"{val:>3}" for val in border_edges))

    # Log region analysis
    logger.info("")
    logger.info('Region Analysis:')
    logger.info(f"{DLM}Available regions: {mesh.region_labels}")
    for region_id in mesh.region_labels:
        element_count = np.count_nonzero(mesh.t_l == region_id)
        region_material = reg2mat.get(region_id, 'Unknown')
        logger.info(f"{DLM}Region {region_id} ({region_material}) elements: {element_count}")

    # Quality checks
    logger.info("")
    logger.info('Mesh Quality Check:')
    total_elements = sum(np.count_nonzero(mesh.t_l == region_id) for region_id in mesh.region_labels)
    if total_elements == mesh.nelem and len(mesh.region_labels) == 2 and len(mesh.border_labels) == 6:
        logger.info(f"{DLM}✓ All quality checks passed")
    else:
        logger.error(f"{DLM}✗ Quality check failed")
        execution_aborted(ValueError("Mesh quality verification failed"))

    # =============================================================================
    # COMPLETION
    # =============================================================================

    logger.info("")
    logger.info('Mesh ready for band structure calculations')
    logger.info(FMT_STR.format('Mesh file', f'{directory_name}/{msh}'))
    logger.info(FMT_STR.format('Metadata file', f'{directory_name}/{dat}'))

    execution_successful()

# =============================================================================
# SCRIPT EXECUTION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    tic()
    main()
    toc()