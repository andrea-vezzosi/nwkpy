# print("Incondizionato: First time loading the library!")

# import sys

# if __name__ not in sys.modules or not hasattr(sys.modules[__name__], '_initialized'):
#     print("First time loading the library!")
#     sys.modules[__name__]._initialized = True
    
from ._constants import *
from ._common import *
from .interface.bandstructure import BandStructure
from .interface.poisson import PoissonProblem
from .interface.updater import Broyden
from .interface.angmom import AngularMomentum
from .hamiltonian import *
from .physics import *
from .operator_matrices import *

from .interface.bandstructure import debug_write, MPI_debug_setup

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    # If MPI is not available, assume single process
    rank = 0
    
# =============================================================================
# LOGGING SYSTEM SETUP
# =============================================================================
import logging
from datetime import datetime
from importlib import metadata
import sys

# =============================================================================
# HEADER
# =============================================================================
def library_header():    
    """Output a header for the library to standard output."""
    
    # Get metadata from installed package
    pkg_info = metadata.metadata("nwkpy")
    name = pkg_info["Name"]
    version = pkg_info["Version"]
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Prepare header lines
    header_lines = [
        " ",
        "=" * 80,
        f"{name} v{version} initialized on {current_date}",
        "A Object Oriented Python library for self-consistent k.p calculations in semiconductor nanowires", 
        "   For more information visit: https://github.com/andrea-vezzosi/nwkpy",
        "   If you use this library for research please cite Phys. Rev. B 105, 245303 (2022)",
        # "   If you use MEFA option please also cite _____.",
        "=" * 80,
    ]
    
    # Log the header to the logger and print to standard output
    logger = logging.getLogger(__name__)
    
    # Print header regardless of logging configuration
    # Check if logging is properly configured
    if logger.hasHandlers() or logging.getLogger().hasHandlers():
        # Logging is configured, use it, print to log file
        for line in header_lines:
            logger.info(line)
    else:
        # No logging configuration yet, use print to stdout 
        for line in header_lines:
            print(line)

# Call library_header only on rank 0 for stdout, but allow logging on all processes
if rank == 0:
    library_header()

