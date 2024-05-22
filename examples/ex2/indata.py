import numpy as np
import os

cdir = os.path.dirname(__file__)
outdata_path = cdir+'/outdata/'

############ MESH #############
mesh_name = "./mesh.msh"

############ PHYSICAL SYSTEM #############
"""
The material we consider for each region of the core-shell nanowire.
1: core material, 2: shell material
"""
reg2mat = {
    1 : 'InAs',
    2 : 'GaSb'
}

"""
The definition of the valence band edge for each material in the heterostructure.
It has to be entered in eV.
"""
valence_band_edges = {
    'InAs': 0.0,
    'GaSb': 0.56
}

"""
Nanowire growth z-axis
"""
principal_axis_direction='111'


"""
Simulation temperature. It affects the energy gaps.
"""
temperature = 4.0 #K

"""
Rescaling of the P parameter is required in order to avoid the appearance
of spurious solutions in the 8-band kp model.
see for example
B. A. Foreman, Elimination of spurious solutions from eight-bandk·ptheory,Phys. Rev. B56, R12748 (1997)

In what follows, S is the parameter you find at Eq. (6.62) of the following reference
[1] S.  Birner,  The  multi-band k·phamiltonian  for  heterostructures: Parameters  and  applications,
in Multi-Band EffectiveMass Approximations(Springer, Cham, 2014), pp. 193-244

Evaluate the Ep (or equivalently P) through Eq. 6.158 of [1]
rescaling={
    'InAs' : 'S=0'
}

Evaluate the Ep (or equivalently P) through Eq. 6.159 of [1]
rescaling={
    'InAs' : 'S=1'
}

Rescale the Ep (or equivalently P) through the following equation
Ep = Ep - 0.26 * Ep
that is reduce its value of 26%, for example.
Than the S parameter, as usual, is computed from Eq. 6.62 of [1]
rescaling={
    'InAs' : 0.26
}

Need to be specified for each material in the heterostructure

The rescaling procedure has to be set for each material in the heterostructure
"""
rescaling={
    'InAs' : 'S=0',
    'GaSb' : 'S=0'
}


'''
Used only for broken-gap nanowires
'''
mat2partic = {
    'InAs' : 'electron',
    'GaSb' : 'hole'
}

""""
The chemical potential, fixed, in eV. The zero of the energy is set in the "valence_band_edges" dicttionary.
"""
chemical_potential = 0.528 #eV

""""
The eigensolver will look for a finite set of eigenvalues
around "e_search", fixed, in eV.
"""
e_search =  0.528 #eV

""""
kz points (must be only positive values since in the charge density we
multiply by two to account for the negative kz values)
"""
# kz initial and final relative kz-values values
kzin_rel = 0.0
kzfin_rel = 0.05
# number of equally spaced kz points
numkz = 4

lattice_constant = 6.0583 #[A] this, for example, is the InAs lattice constant

# absolute kz values
kzvals = np.linspace(kzin_rel, kzfin_rel, numkz) * np.pi / np.sqrt(3.0) / lattice_constant # [A^-1]

##################### SOLVER ################
# shape function (SAFE)
shape_kind_kp = {
    'el' : 'Hermite',
    'h'  : 'LagrangeQuadratic'
}
shape_kind_poisson = 'LagrangeQuadratic'

# number of eigenvalues to search for (or number of subbands)
k = 20

############## ELECTROSTATICS ###############
init_pot_name = None

# if this parameter is true, the modified EFA approach is used (see Ch. 5 of my thesis)
modified_EFA=True

thr_el=0.8
thr_h=0.95

# don't remember what this was
particle_s_components = 'electron'
particle_p_components = 'electron'

"""
SOME EXAMPLES OF BOUNDARY CONDITIONS FOR THE POISSON EQUATION

This boundary conditions fix the electron potential energy -e \phi in units of eV.

Fix the potential to zero in all borders
dirichlet = {
    'ref' : 0.0
}

Fix the potential to zero in all borders but the
bottom border, which is set to 0.1
dirichlet = {
    'ref' : 0.0,
    1 : 0.1
}

Fix the potential to zero on the bottom border, and
use Neumann boundary conditions on the remaing borders
dirichlet = {
    'ref' : None,
    1 : 0.0
}

Fix the potential to zero on the bottom border, to 0.1 on the
top border, and use Neumann BC on the remaining borders
dirichlet = {
    'ref' : None,
    1 : 0.0
    2 : 0.1
}

Use pure Neumann BC. An additional constrain is required
in this case, which is that the mean value of the potential
is equal to zero.
dirichlet = None

"""

dirichlet = {
    'ref':None,
    1 : 0.0,
}

"""
Include an external electric field in the simulation.
tuple = (modulus of electric field in V/1e-6m, angle with the x-axis clockwise)
"""
electric_field = (0.0, np.pi/2) # V/1e-6m (Volt/micron)

############## MATERIAL PARAMETERS ###############

"""
Band strcture parameters for each material in the heterostructure
Energies in eV, lengths in A
"""

user_defined_params = {
    'InAs' :{
            'Eg' : 0.417,
            'delta' : 0.39 ,
            'Ep' : 21.5 ,
            'me' : 0.026,
            'lu1' : 20.0,
            'lu2' : 8.5,
            'lu3' : 9.2,
            'L'  : 0.0,   # these are calculated from the luttinger params
            'M'  : 0.0,
            'Nm' : 0.0,
            'Np' : 0.0,
            'N'  : 0.0,
            'P' : 0.0,
            'Ac' : 0.0,
            'B' : 0.0,
            'eps' : 15.5,
            'alc' : 6.0583, # A at T=300 K
            'alpha' : 2.76 * 1e-4, # eV ...
            'alcTpar': 2.74 * 1e-5, # A/K
            'beta' : 93 # K
            },
    'GaSb' :{
            'Eg' : 0.812,
            'delta' : 0.76,
            'Ep' : 27.0,
            'me' : 0.039,
            'lu1' : 13.4,
            'lu2' : 4.7,
            'lu3' : 6.0,
            'L'  : 0.0,
            'M'  : 0.0,
            'Nm' : 0.0,
            'Np' : 0.0,
            'N'  : 0.0,
            'P' : 0.0,
            'Ac' : 0.0,
            'B' : 0.0,
            'eps': 15.7,
            'alc': 6.0959, #A at T=300K
            'alcTpar': 4.72 * 1e-5, # A/K
            'alpha' : 4.17 * 1e-4,
            'beta' : 140
            }
}

################ Plotting preferencies ################
plotting_preferencies_bands = {
    'xlim' : (0,0.3), # nm^-1
    'ylim' : (515,560), # meV
    'cmap_in' : 'rainbow',
    'character_to_show' : 'H-EL',#-reg-threshold',
    'threshold_el' : 0.8, 
    'threshold_h' : 0.95, 
    'loc_cbar' : 1,
    'spines_lw' : 4,
    'lw' : 5,
    'fontsize' : 20
}

plotting_preferencies_density = {
    'subdiv': 1,
    'figsize': (5, 5),
    'xlim': (-20, 20), # nm
    'ylim': (-20, 20), # nm
    'cmapin': 'rainbow',
    'levels': 21,
    'fontsize': 20,
    'polygons': None
}

plotting_preferencies_potential = {
    'subdiv': 0,
    'figsize': (5, 5),
    'xlim': (-20, 20), # nm
    'ylim': (-20, 20), # nm
    'cmapin': 'rainbow',
    'levels': 21,
    'fontsize': 20,
    'polygons': None
}
