"""
Configuration Constants for scripts based on the nwkpy library.

These constants control formatting, validation, and output behavior


Constants:
    I/O and Formatting:
        DLM: Standard delimiter for text output and log spacing
        FMT_STR: Template for consistent log message formatting
        LOG_FILE_NAME: Base name for simulation log files
    
    Validation Lists:
        CARRIER_TYPES: Valid carrier type specifications
        SHAPE_FUNCTION_TYPES: Valid finite element shape function options
        PRINCIPAL_AXIS_DIRECTIONS: Valid crystallographic growth directions

Usage:
    Import this module to access consistent formatting and validation
    constants across the entire simulation codebase.
"""

# =============================================================================
# INPUT/OUTPUT FORMATTING CONSTANTS
# =============================================================================

# Text formatting and spacing controls
DLM = '   '                              # Delimiter for text files and log spacings
                                         # Creates consistent column alignment of log messages

FMT_STR = "   {:<30} : {}"               # Log message formatting template with two
                                         # placeholders, 1st 30 char, left justified
                                         # Format: "   Parameter Name           : Value"

LOG_FILE_NAME = 'logfile'                # Base name for log file (without extension)   
                                         # Full log file will be: logfile.log
                                         # Located in the output directory set by indata

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Valid input parameter options for error checking and user guidance
# used by input validation functions

CARRIER_TYPES = ['electron', 'hole']                 # Valid carrier types
                                                     # Used to specify dominant carriers
                                                     # in each material region

SHAPE_FUNCTION_TYPES = ['Lagrange', 'Hermite',       # Valid shape functions
         'LagrangeHermite', 'LagrangeQuadratic']  
                                                     # Finite element basis functions:
                                                     # - 'Lagrange': Standard polynomial basis
                                                     # - 'Hermite': Higher-order accuracy
                                                     # - 'LagrangeHermite': Mixed basis  
                                                     # - 'LagrangeQuadratic': Quadratic elements

PRINCIPAL_AXIS_DIRECTIONS = ['100', '110', '111']   # Valid principal axis directions
                                                     # Crystallographic directions for
                                                     # nanowire growth axis:
                                                     # - '100': Face-centered cubic
                                                     # - '110': Edge-centered
                                                     # - '111': Body diagonal (zinc blende)
