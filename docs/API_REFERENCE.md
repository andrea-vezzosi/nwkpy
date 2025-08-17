# API Reference

Complete API documentation for nwkpy - the 8-band k·p library for semiconductor nanowire calculations.

## Overview

nwkpy provides a comprehensive set of classes and functions for semiconductor nanowire calculations using the 8-band k·p method with finite element discretization. The library is organized into several key modules:

- **Core Interface Classes**: `BandStructure`, `PoissonProblem`, `Broyden`, `AngularMomentum`
- **Physics Modules**: `FreeChargeDensity`, `ElectrostaticPotential`, `WaveFunction`, `DopingChargeDensity`
- **Hamiltonian Classes**: `HamiltonianZB`, `HamiltonianWZ`
- **Finite Element Framework**: `Mesh`, `FemSpace`, `FiniteElement`, shape functions, solvers
- **Utilities**: Material database, constants, and I/O functions

## Quick Reference

### Essential Imports

```python
from nwkpy import BandStructure, PoissonProblem
from nwkpy.fem.mesh import Mesh
from nwkpy.physics import FreeChargeDensity, ElectrostaticPotential
from nwkpy.interface import Broyden
```

### Basic Workflow

```python
# 1. Load mesh
mesh = Mesh("nanowire.msh", reg2mat={1: "InAs", 2: "GaSb"})

# 2. Calculate band structure
bs = BandStructure(mesh=mesh, kzvals=k_values, ...)
bs.run()

# 3. Self-consistent coupling (optional)
poisson = PoissonProblem(mesh, rho_free={'rho_el': rho_el, 'rho_h': rho_h})
poisson.run()
```

---

## Core Interface Classes

### BandStructure

**Location**: `nwkpy.interface.bandstructure.BandStructure`

Main class for 8-band k·p electronic structure calculations with full MPI support and advanced numerical features.

#### Constructor

```python
BandStructure(
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
)
```

#### Parameters

**mesh** : `Mesh`
    Finite element mesh object containing the nanowire geometry and material assignments.

**kzvals** : `array_like`
    Array of k-vector values along the nanowire axis (1/Å). Only positive values needed due to symmetry.

**valence_band_edges** : `dict`
    Dictionary mapping material names to valence band edge energies (eV).
    Example: `{"InAs": 0.0, "GaSb": 0.56}`

**principal_axis_direction** : `str`, optional
    Crystallographic growth direction. Options: `'001'`, `'110'`, `'111'`. Default: `'001'`.

**crystal_form** : `str`, optional
    Crystal structure type. Options: `'ZB'` (zinc blende), `'WZ'` (wurtzite). Default: `'ZB'`.

**temperature** : `float`, optional
    System temperature in Kelvin. Affects band gaps and Fermi-Dirac statistics. Default: 4.0.

**k** : `int`, optional
    Number of eigenvalues/eigenvectors to compute per k-point. Default: 60.

**shape_functions** : `dict`, optional
    Shape function types for electrons and holes. Default: `{'el': 'Hermite', 'h': 'LagrangeQuadratic'}`.

**epot** : `ElectrostaticPotential`, optional
    External electrostatic potential. If None, calculation is non-self-consistent. Default: None.

**e_search** : `float`, optional
    Energy around which to search for eigenvalues (eV). Should be near chemical potential. Default: 0.0.

**rescaling** : `dict`, optional
    P-parameter rescaling factors by material to suppress spurious solutions. 
    Options: `'S=0'`, `'S=1'`, `'P=0'`, or numerical values. Default: None.

#### Attributes

**mesh** : `Mesh`
    The finite element mesh object.

**bands** : `ndarray`
    Energy eigenvalues, shape `(n_k_points, n_eigenvalues)` in eV.

**spinor_distribution** : `ndarray`
    8-component spinor character, shape `(n_k_points, 8, n_eigenvalues)`.

**psi_el** : `ndarray`
    Electron envelope functions, shape `(n_k_points, n_nodes, 2, n_eigenvalues)`.

**psi_h** : `ndarray`
    Hole envelope functions, shape `(n_k_points, n_nodes, 6, n_eigenvalues)`.

**norm_sum_region** : `ndarray`
    Regional charge localization, shape `(n_k_points, n_regions, n_eigenvalues)`.

**kzvals** : `ndarray`
    k-vector values used in calculation.

**fs_el, fs_h** : `FemSpace`
    Finite element spaces for electrons and holes.

#### Methods

##### run()

Execute the band structure calculation with MPI parallelization.

```python
bs.run()
```

**Returns**
    None. Results are stored in class attributes.

**Notes**
    This method performs the complete 8-band k·p calculation:
    1. Distributes k-points across MPI processes
    2. Assembles Hamiltonian matrices for each k-point
    3. Solves generalized eigenvalue problems using ARPACK
    4. Collects results and calculates spinor character
    5. Stores energy bands and envelope functions

##### plot_bands()

Generate comprehensive band structure visualization with character analysis.

```python
fig = bs.plot_bands(
    threshold_el=None, 
    threshold_h=None, 
    chemical_potential=None, 
    character_to_show=None, 
    figsize=(5, 5), 
    xlim=None, 
    ylim=None, 
    cmap_in='Blues',
    loc_cbar=1,
    spines_lw=4,
    fontsize=20
)
```

**Parameters**

**character_to_show** : `str`, optional
    Spinor character to visualize. Options:
    - `'EL'` - Electron character
    - `'HH'` - Heavy hole character  
    - `'LH'` - Light hole character
    - `'SO'` - Split-off character
    - `'LH-HH'` - Light hole vs heavy hole
    - `'H-EL'` - Hole vs electron character
    - `'H-EL-reg'` - Regional hole vs electron character

**Returns**

**fig** : `matplotlib.figure.Figure`
    Figure object containing the band structure plot with color-coded character.

##### plot_density()

Visualize charge density distribution on nanowire cross-section.

```python
fig = bs.plot_density(
    *density, 
    xlim, 
    ylim, 
    figsize=(5,5), 
    subdiv=1, 
    cmapin='rainbow', 
    levels=21,
    fontsize=20,
    polygons=None
)
```

#### Examples

**Basic band structure calculation:**

```python
import numpy as np
from nwkpy import BandStructure
from nwkpy.fem.mesh import Mesh

# Load mesh
mesh = Mesh("hexagonal_nanowire.msh", 
           reg2mat={1: "InAs", 2: "GaSb"})

# Set up k-space
kz_values = np.linspace(0, 0.05, 20) * np.pi / np.sqrt(3) / 6.0583

# Calculate band structure
bs = BandStructure(
    mesh=mesh,
    kzvals=kz_values,
    valence_band_edges={"InAs": 0.0, "GaSb": 0.56},
    temperature=4.0,
    k=20,
    rescaling={"InAs": "S=0", "GaSb": "S=0"}
)
bs.run()

# Visualize results
fig = bs.plot_bands(
    chemical_potential=0.528,
    character_to_show='H-EL',
    xlim=(0, 0.3),
    ylim=(520, 540)
)
```

**With electrostatic potential:**

```python
from nwkpy.physics import ElectrostaticPotential

# Include external electric field
epot = ElectrostaticPotential(
    mesh.fs_el, 
    electric_field=(0.1, np.pi/2)  # 0.1 V/μm in y-direction
)

# Band structure with electrostatic coupling
bs = BandStructure(
    mesh=mesh,
    kzvals=kz_values,
    valence_band_edges={"InAs": 0.0, "GaSb": 0.56},
    epot=epot,
    temperature=4.0
)
bs.run()
```

---

### PoissonProblem

**Location**: `nwkpy.interface.poisson.PoissonProblem`

Solver for the Poisson equation with free charge densities and doping profiles, implementing the Burt-Foreman formulation.

#### Constructor

```python
PoissonProblem(
    mesh,
    shape_class_name="LagrangeQuadratic",
    dirichlet=None,
    electric_field=(0.0, np.pi*0.5),
    user_defined_parameters=None,
    rho_doping=None,
    **rho_free
)
```

#### Parameters

**mesh** : `Mesh`
    Finite element mesh object.

**shape_class_name** : `str`, optional
    Shape function type for Poisson equation. Options: `"LagrangeQuadratic"`, `"Lagrange"`. Default: `"LagrangeQuadratic"`.

**dirichlet** : `dict`, optional
    Dirichlet boundary conditions. Format: `{'ref': default_value, boundary_label: potential_value}`.
    Example: `{'ref': None, 1: 0.0, 3: 0.1}` sets boundaries 1 and 3 to 0.0V and 0.1V.

**electric_field** : `tuple`, optional
    External electric field as `(magnitude_V_per_μm, angle_radians)`. Default: `(0.0, π/2)`.

**user_defined_parameters** : `dict`, optional
    Custom material parameters overriding database values. Default: None.

**rho_doping** : `DopingChargeDensity`, optional
    Doping charge density object. Default: None.

****rho_free** : `FreeChargeDensity`
    Free charge densities from electronic structure (rho_el, rho_h, etc.).

#### Attributes

**mesh** : `Mesh`
    The finite element mesh.

**fs** : `FemSpace`
    Finite element space for potential.

**epot** : `ElectrostaticPotential`
    Resulting electrostatic potential object.

**electric_field** : `tuple`
    Applied external electric field.

**c** : `float`
    Lagrange multiplier for pure Neumann problems.

**pure_neumann** : `bool`
    Whether the problem has only Neumann boundary conditions.

#### Methods

##### run()

Solve the Poisson equation using finite element method.

```python
poisson.run()
```

**Returns**
    None. Results stored in `self.epot`.

**Notes**
    Solves: ∇·[ε(r)∇φ(r)] = -ρ(r)/ε₀ in CGS units
    where ρ(r) includes free carriers and doping.

#### Examples

**Basic Poisson solution:**

```python
from nwkpy import PoissonProblem
from nwkpy.physics import FreeChargeDensity

# Create charge densities from band structure
rho_el = FreeChargeDensity(bs.fs_el)
rho_h = FreeChargeDensity(bs.fs_h)

# Add charge from occupied states
rho_el.add_charge(bs.psi_el, bs.bands, dk=0.001, mu=0.528, temp=4.0)
rho_h.add_charge(bs.psi_h, bs.bands, dk=0.001, mu=0.528, temp=4.0)

# Solve Poisson equation
poisson = PoissonProblem(
    mesh=mesh,
    dirichlet={'ref': None, 1: 0.0},  # Ground boundary 1
    electric_field=(0.1, np.pi/2),   # 0.1 V/μm in y-direction
    rho_el=rho_el,
    rho_h=rho_h
)
poisson.run()

# Use resulting potential
potential = poisson.epot
```

**With doping:**

```python
from nwkpy.physics import DopingChargeDensity

# Define doping region
def hexagon_region(coords):
    # Define hexagonal doping region
    width = 10  # nm
    s = width/2. / np.cos(np.pi/6.)
    x, y = np.abs(coords.T)
    mask = y < np.sqrt(3) * np.minimum(s - x, s / 2)
    return mask

# Create doping charge density
doping = DopingChargeDensity(
    doping_concentration_value=1e17,  # cm^-3
    region_fun=hexagon_region
)

# Solve with doping
poisson = PoissonProblem(
    mesh=mesh,
    rho_doping=doping,
    rho_el=rho_el,
    rho_h=rho_h
)
poisson.run()
```

---

### Broyden

**Location**: `nwkpy.interface.updater.Broyden`

Advanced convergence acceleration for self-consistent calculations using Broyden's method.

#### Constructor

```python
Broyden(N, M=6, beta=0.35, w0=0.01, use_wm=True)
```

#### Parameters

**N** : `int`
    Dimension of the problem (number of potential nodes).

**M** : `int`, optional
    Maximum number of stored iterations for Broyden acceleration. Default: 6.

**beta** : `float`, optional
    Linear mixing parameter. Default: 0.35.

**w0** : `float`, optional
    Weight for initial iteration. Default: 0.01.

**use_wm** : `bool`, optional
    Use iteration-dependent weights. Default: True.

#### Methods

##### update()

Update potential using Broyden mixing.

```python
V_new = broyden.update(xin=V_in, xout=V_out)
```

**Parameters**

**xin** : `array_like`
    Input potential from current iteration.

**xout** : `array_like` 
    Output potential from Poisson solution.

**Returns**

**V_new** : `ndarray`
    Mixed potential for next iteration.

#### Example

```python
from nwkpy.interface import Broyden

# Initialize Broyden mixer
broyden = Broyden(N=mesh.ng_nodes, M=6, beta=0.35)

# Self-consistent loop
for iteration in range(max_iterations):
    # 1. Solve Schrödinger equation
    bs = BandStructure(mesh=mesh, epot=epot_in, ...)
    bs.run()
    
    # 2. Calculate charge densities
    rho_el.add_charge(bs.psi_el, bs.bands, ...)
    
    # 3. Solve Poisson equation
    poisson = PoissonProblem(mesh, rho_el=rho_el, ...)
    poisson.run()
    V_out = poisson.epot.V
    
    # 4. Check convergence
    if converged: break
    
    # 5. Apply Broyden mixing
    V_mixed = broyden.update(xin=V_in, xout=V_out)
    epot_in = ElectrostaticPotential(mesh.fs, V=V_mixed)
    V_in = V_mixed
```

---

## Physics Classes

### FreeChargeDensity

**Location**: `nwkpy.physics.FreeChargeDensity`

Handles free carrier charge densities from electronic wavefunctions with support for Modified Envelope Function Approximation (MEFA).

#### Constructor

```python
FreeChargeDensity(fs, logger=None)
```

#### Parameters

**fs** : `FemSpace`
    Finite element space object for interpolation.

**logger** : `Logger`, optional
    Logging object for debug output. Default: None.

#### Attributes

**fs** : `FemSpace`
    Finite element space.

**n, p** : `ndarray`
    Electron and hole densities at mesh nodes (cm⁻³).

**charge_matrix_el, charge_matrix_h** : `sparse matrix`
    Interpolation matrices for electron and hole densities.

**charge_matrix_el_pure, charge_matrix_h_pure** : `sparse matrix`
    Interpolation matrices for pure (non-hybridized) states.

#### Methods

##### add_charge()

Add charge contribution from electronic states using standard or modified EFA.

```python
rho.add_charge(
    psi, 
    e, 
    dk, 
    mu, 
    temp, 
    modified_EFA=False,
    particle='electron', 
    norm_sum_region=None, 
    thr_el=0.5, 
    thr_h=0.5
)
```

**Parameters**

**psi** : `ndarray`
    Wave function array, shape `(n_k_points, n_nodes, n_components, n_eigenvalues)`.

**e** : `ndarray`
    Energy eigenvalues (eV), shape `(n_k_points, n_eigenvalues)`.

**dk** : `float`
    k-space integration step (1/Å).

**mu** : `float`
    Chemical potential (eV).

**temp** : `float`
    Temperature (K).

**modified_EFA** : `bool`, optional
    Use Modified Envelope Function Approximation for broken-gap systems. Default: False.

**particle** : `str`, optional
    Particle type: `'electron'` or `'hole'`. Default: `'electron'`.

**norm_sum_region** : `ndarray`, optional
    Regional character analysis, required for modified_EFA=True.

**thr_el, thr_h** : `float`, optional
    Character thresholds for state classification. Default: 0.5.

##### interp()

Interpolate charge density at arbitrary coordinates.

```python
density = rho.interp(coords, total=True, values=None)
```

**Parameters**

**coords** : `ndarray`
    Coordinate array, shape `(n_points, 2)`.

**total** : `bool`, optional
    Return total density (n+p) or separate (n, p). Default: True.

**Returns**

**density** : `ndarray` or `tuple`
    Total charge density or (electron_density, hole_density).

##### get_total_charge()

Calculate total integrated charge for verification.

```python
n_total, p_total = rho.get_total_charge()
```

**Returns**

**n_total, p_total** : `float`
    Total electron and hole charges (cm⁻¹).

#### Example

```python
from nwkpy.physics import FreeChargeDensity

# Create charge density object
rho_el = FreeChargeDensity(bs.fs_el)

# Add charge from all calculated states
dk = kz_values[1] - kz_values[0]  # k-space step
rho_el.add_charge(
    psi=bs.psi_el,
    e=bs.bands,
    dk=dk,
    mu=0.528,           # Chemical potential
    temp=4.0,           # Temperature
    modified_EFA=True,  # Use MEFA for broken-gap systems
    particle='electron',
    norm_sum_region=bs.norm_sum_region,
    thr_el=0.8          # 80% electron character threshold
)

# Check total charge
n_total, p_total = rho_el.get_total_charge()
print(f"Total electron charge: {n_total:.2e} cm⁻¹")

# Interpolate at specific points
coords = np.array([[0, 0], [5, 5], [10, 0]])  # nm
densities = rho_el.interp(coords)
```

---

### ElectrostaticPotential

**Location**: `nwkpy.physics.ElectrostaticPotential`

Represents electrostatic potential energy for electronic structure calculations.

#### Constructor

```python
ElectrostaticPotential(
    fs, 
    V=None, 
    electric_field=(0.0, np.pi*0.5)
)
```

#### Parameters

**fs** : `FemSpace`
    Finite element space for potential interpolation.

**V** : `array_like`, optional
    Potential values at mesh nodes (eV). Default: None.

**electric_field** : `tuple`, optional
    External uniform field `(magnitude_V_per_μm, angle_radians)`. Default: `(0.0, π/2)`.

#### Attributes

**fs** : `FemSpace`
    Finite element space.

**V** : `ndarray`
    Potential values at mesh nodes (eV).

#### Methods

##### interp()

Interpolate potential at arbitrary coordinates.

```python
potential = epot.interp(coords)
```

##### plot()

Visualize electrostatic potential distribution.

```python
fig = epot.plot(xlim, ylim, figsize=(5,5), subdiv=1, levels=21)
```

#### Example

```python
from nwkpy.physics import ElectrostaticPotential

# Create potential with external field
epot = ElectrostaticPotential(
    fs=mesh.fs,
    electric_field=(0.2, np.pi/4)  # 0.2 V/μm at 45°
)

# Or with solved Poisson potential
epot = ElectrostaticPotential(
    fs=mesh.fs,
    V=poisson_solution,
    electric_field=(0.1, 0.0)
)

# Plot potential
fig = epot.plot(
    xlim=(-10, 10),
    ylim=(-10, 10),
    levels=25
)
```

---

### WaveFunction

**Location**: `nwkpy.physics.WaveFunction`

Advanced wavefunction analysis including symmetry operations and angular momentum calculations.

#### Constructor

```python
WaveFunction(fs, psi)
```

#### Parameters

**fs** : `FemSpace`
    Finite element space object.

**psi** : `ndarray`
    Solution vector, shape `(n_k_points, n_nodes, n_components, n_eigenvalues)`.

#### Methods

##### get_dominant_total_angular_momentum()

Calculate total angular momentum analysis.

```python
eigvals, eigvecs, coeff, mj_proj = wf.get_dominant_total_angular_momentum(
    mj_values, neig=100
)
```

##### apply_symmetry_operator()

Apply crystallographic symmetry operations.

```python
psi_transformed = wf.apply_symmetry_operator(symop)
```

---

### DopingChargeDensity

**Location**: `nwkpy.physics.DopingChargeDensity`

Represents spatially-varying doping charge densities.

#### Constructor

```python
DopingChargeDensity(
    doping_concentration_value=None, 
    region_fun=None
)
```

#### Example

```python
def core_doping(coords):
    """Define doping in core region"""
    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    return r < 5.0  # 5 nm radius

doping = DopingChargeDensity(
    doping_concentration_value=1e18,  # cm⁻³
    region_fun=core_doping
)
```

---

## Hamiltonian Classes

### HamiltonianZB

**Location**: `nwkpy.hamiltonian.HamiltonianZB`

8-band k·p Hamiltonian for zinc blende crystal structures with spurious solution suppression.

#### Constructor

```python
HamiltonianZB(
    material,
    valence_band_edge=0.0,
    principal_axis_direction='001',
    rescaling='S=0',
    temperature=0.0,
    rembands=True,
    spherical_approx=False,
    decouple_split_off=None,
    decouple_conduction=None,
    user_defined_params=None
)
```

#### Parameters

**material** : `str`
    Material name (must exist in material database).

**valence_band_edge** : `float`, optional
    Valence band edge energy (eV). Default: 0.0.

**rescaling** : `str` or `float`, optional
    P-parameter rescaling scheme. Options:
    - `'S=0'` - Standard Ep evaluation (Foreman Eq. 6.158)  
    - `'S=1'` - Modified Ep evaluation (Foreman Eq. 6.159)
    - `'P=0'` - Set P=0 (no interband coupling)
    - `float` - Fractional Ep reduction. Default: `'S=0'`.

**temperature** : `float`, optional
    Temperature for band gap calculation (K). Default: 0.0.

**spherical_approx** : `bool`, optional
    Use spherical approximation for Luttinger parameters. Default: False.

#### Attributes

**p** : `dict`
    Material parameters in atomic units.

**material** : `str`
    Material name.

**theta, phi** : `float`
    Rotation angles for crystal orientation.

**C2, C1L, C1R, C0** : `ndarray`
    Coefficient matrices for k·p Hamiltonian.

**Hxx, Hxy, ...** : `ndarray`
    Individual Hamiltonian component matrices.

#### Methods

##### get()

Get bulk Hamiltonian for specific k-vector.

```python
H = hamiltonian.get(kx, ky, kz)
```

##### solve()

Solve for eigenvalues at specific k-point.

```python
eigenvalues, eigenvectors = hamiltonian.solve(kx, ky, kz)
```

##### quantize1D()

Prepare Hamiltonian for finite element calculation.

```python
hamiltonian.quantize1D(kz)
```

#### Example

```python
from nwkpy.hamiltonian import HamiltonianZB

# Create Hamiltonian for InAs
ham = HamiltonianZB(
    material="InAs",
    valence_band_edge=0.0,
    principal_axis_direction='111',  # [111] growth
    rescaling='S=0',                 # Spurious solution suppression
    temperature=4.0,
    rembands=True                    # Include remote bands
)

# Get Hamiltonian at Γ-point
H_gamma = ham.get(0, 0, 0)
print(f"8x8 Hamiltonian shape: {H_gamma.shape}")

# Solve for bulk band structure
k_values = np.linspace(0, 0.1, 50)
band_structure = []
for k in k_values:
    eigenvals, _ = ham.solve(0, 0, k)
    band_structure.append(eigenvals)
```

---

### HamiltonianWZ

**Location**: `nwkpy.hamiltonian_wz.HamiltonianWZ`

8-band k·p Hamiltonian for wurtzite crystal structures.

Similar interface to `HamiltonianZB` but adapted for wurtzite symmetry with anisotropic parameters.

#### Additional Parameters

**me_par, me_perp** : `float`
    Parallel and perpendicular electron effective masses.

**Ep1, Ep2** : `float`
    Anisotropic Kane parameters.

**delta1, delta2, delta3** : `float`
    Wurtzite spin-orbit parameters.

---

## Finite Element Framework

### Mesh

**Location**: `nwkpy.fem.mesh.mesh.Mesh`

Main mesh class for finite element discretization of nanowire geometries.

#### Constructor

```python
Mesh(
    mesh_name,
    reg2mat=None, 
    mat2partic=None,
    restrict_to=None, 
    bandwidth_reduction=True
)
```

#### Parameters

**mesh_name** : `str`
    Path to mesh file (without extension). Requires both `.msh` and `.dat` files.

**reg2mat** : `dict`
    Region to material mapping.

**material** : `ndarray`
    Material name for each element.

**nelem** : `int`
    Number of finite elements.

**ng_nodes** : `int`
    Number of mesh nodes.

**triangulation** : `matplotlib.tri.Triangulation`
    Triangulation object for interpolation.

**trifinder** : `matplotlib.tri.TrapezoidMapTriFinder`
    Fast point location for interpolation.

#### Methods

##### size_print()

Print mesh size information.

```python
mesh.size_print()
```

**Output example:**
```
MESH SIZE DATA

Number of vertices = 1247
Number of boundary edges = 84
Number of triangles = 2388
```

##### ChangeP1toP2Mesh()

Convert linear mesh to quadratic elements.

```python
mesh.ChangeP1toP2Mesh()
```

#### Examples

```python
from nwkpy.fem.mesh import Mesh

# Load hexagonal nanowire mesh
mesh = Mesh(
    mesh_name="nanowire_mesh",
    reg2mat={1: "InAs", 2: "GaSb"},
    mat2partic={"InAs": "electron", "GaSb": "hole"}
)

# Print mesh statistics
mesh.size_print()

# Access mesh data
print(f"Core material: {mesh.reg2mat[1]}")
print(f"Shell material: {mesh.reg2mat[2]}")
print(f"Total elements: {mesh.nelem}")

# Convert to quadratic elements
mesh.ChangeP1toP2Mesh()
```

---

### FemSpace

**Location**: `nwkpy.fem.element.FemSpace`

Finite element space containing all elements and shape function information.

#### Constructor

```python
FemSpace(mesh, shape_class_name)
```

#### Parameters

**mesh** : `Mesh`
    Finite element mesh object.

**shape_class_name** : `str`
    Shape function type. Options:
    - `'Lagrange'` - Linear Lagrange elements
    - `'LagrangeQuadratic'` - Quadratic Lagrange elements  
    - `'Hermite'` - Cubic Hermite elements
    - `'LagrangeHermite'` - Mixed Lagrange-Hermite elements

#### Attributes

**mesh** : `Mesh`
    Associated mesh object.

**felems** : `list`
    List of `FiniteElement` objects.

**dlnc** : `ndarray`
    Degrees of freedom per node.

**dlnc_cumul** : `ndarray`
    Cumulative DOF indexing.

**total_area** : `float`
    Total mesh area in rescaled units.

#### Methods

##### get_total_area()

Calculate total mesh area.

```python
area = fs.get_total_area()
```

##### get_region_area()

Calculate area of specific regions.

```python
area = fs.get_region_area([1, 2])  # Core and shell regions
```

---

### FiniteElement

**Location**: `nwkpy.fem.element.FiniteElement`

Individual finite element with shape functions and integration capabilities.

#### Constructor

```python
FiniteElement(iel, mesh, shape)
```

#### Attributes

**iel** : `int`
    Element index.

**nods** : `ndarray`
    Node indices for this element.

**material** : `str`
    Material type for this element.

**detJ** : `float`
    Jacobian determinant.

**gauss_coords** : `ndarray`
    Gauss integration point coordinates.

#### Methods

##### int_f()

Integrate arbitrary function over element.

```python
integral = fel.int_f(function_values_at_gauss_points)
```

##### interp_sol()

Interpolate solution at arbitrary point.

```python
value = fel.interp_sol(x, y, nodal_solution)
```

---

### Shape Functions

**Location**: `nwkpy.fem.shape`

Shape function classes for different interpolation orders.

#### Available Shape Functions

- **ShapeFunctionLagrange** - Linear triangular elements (3 nodes, 3 DOF)
- **ShapeFunctionLagrangeQuadratic** - Quadratic triangular elements (6 nodes, 6 DOF)
- **ShapeFunctionHermite** - Cubic Hermite elements (3 nodes, 9 DOF)
- **ShapeFunctionLH6** - Mixed Lagrange-Hermite (4 nodes, 6 DOF)
- **ShapeFunctionLH7** - Mixed Lagrange-Hermite (3 nodes, 7 DOF)

#### Example

```python
from nwkpy.fem.shape import ShapeFunctionHermite

# Create Hermite shape function
shape = ShapeFunctionHermite(ngauss=12)

# Evaluate at reference coordinates
csi, eta = 0.5, 0.3
N = shape.fun(csi, eta)        # Shape function values
dN_dcsi = shape.der_csi(csi, eta)  # Derivatives
```

---

### Solvers

**Location**: `nwkpy.fem.solver`

Advanced numerical solvers for eigenvalue and linear systems.

#### GenEigenProblem

Generalized eigenvalue problem solver for band structure calculations.

```python
from nwkpy.fem.solver import GenEigenProblem
from nwkpy.fem.problem import Schrodinger

# Create solver
solver = GenEigenProblem()

# Assemble matrices
problem = Schrodinger(kz=0.01, hamiltonian_dict=hamiltonians)
solver.assembly(fem_space_product, problem)

# Solve eigenvalue problem
eigenvals, eigvecs_el, eigvecs_h, spinor_dist, norm_region = solver.solve(
    k=20,           # Number of eigenvalues
    sigma=0.528,    # Search around this energy (eV)
    which='LM'      # Largest magnitude
)
```

#### LinearSystem

Linear system solver for Poisson equation.

```python
from nwkpy.fem.solver import LinearSystem
from nwkpy.fem.problem import Poisson

# Create solver
solver = LinearSystem()

# Assemble system
problem = Poisson(rho_el=rho_el, rho_h=rho_h)
solver.assembly(fem_space, problem, dirichlet_borval=boundary_conditions)

# Solve linear system
solution = solver.solve()
```

---

### Problems

**Location**: `nwkpy.fem.problem`

Variational formulations for physical problems.

#### Schrodinger

8-band k·p Schrödinger equation formulation.

```python
from nwkpy.fem.problem import Schrodinger

problem = Schrodinger(
    kz=0.02,                    # Wave vector (1/Å)
    hamiltonian_dict=ham_dict,  # Material Hamiltonians
    epot=electrostatic_potential # Optional potential
)
```

#### Poisson

Poisson equation for electrostatic potential.

```python
from nwkpy.fem.problem import Poisson

problem = Poisson(
    rho_dop=doping_density,     # Optional doping
    rho_el=electron_density,    # Free electron density
    rho_h=hole_density         # Free hole density
)
```

---

## Utility Functions and Data

### Material Database

**Location**: `nwkpy._database.params`

Comprehensive database of semiconductor material parameters.

```python
from nwkpy._database import params

# Get all available materials
materials = list(params.keys())
print(f"Available materials: {materials}")

# Access material parameters
inas_params = params['InAs']
print(f"InAs band gap: {inas_params['Eg']} eV")
print(f"InAs electron mass: {inas_params['me']} m₀")
print(f"InAs Luttinger parameters: γ₁={inas_params['lu1']}, γ₂={inas_params['lu2']}")
```

#### Available Materials

- **III-V Semiconductors**: InAs, GaAs, GaSb, InSb, InP, AlSb, AlGaAs
- **Material Variants**: InAsP0, GaSbP0 (modified parameter sets)

#### Parameter Categories

**Band Structure Parameters**:
- `Eg` - Band gap (eV)
- `delta` - Spin-orbit coupling (eV)  
- `Ep` - Kane parameter (eV)
- `me` - Electron effective mass (m₀)
- `lu1`, `lu2`, `lu3` - Luttinger parameters

**Temperature Dependence**:
- `alpha`, `beta` - Band gap temperature coefficients
- `alcTpar` - Lattice constant temperature dependence

**Physical Properties**:
- `eps` - Dielectric constant
- `alc` - Lattice constant (Å)

### Physical Constants

**Location**: `nwkpy._constants`

Physical constants in atomic units.

```python
from nwkpy import _constants

# Length scale conversion
bohr_radius = _constants.length_scale  # 0.529 Å
print(f"Bohr radius: {bohr_radius} Å")

# Energy scale conversion  
hartree = _constants.energy_scale      # 27.21 eV
print(f"Hartree: {hartree} eV")

# Usage in unit conversion
length_nm = 10.0  # nm
length_au = length_nm / (bohr_radius / 10)  # Convert to atomic units
```

### Mesh Generation

**Location**: `nwkpy.fem.mesh`

Mesh generation utilities using FreeFem++ integration.

#### Hex2regsymm

Generate hexagonal core-shell nanowire mesh.

```python
from nwkpy.fem.mesh import Hex2regsymm

# Generate mesh
Hex2regsymm(
    mesh_name="hexagonal_nanowire.msh",
    total_width=24.76,          # Total diameter (nm)
    shell_width=4.88,           # Shell thickness (nm)  
    edges_per_border={
        'nC1': 10, 'nC2': 5, 'nC3': 7,
        'nC4': 6,  'nC5': 5, 'nC6': 5
    }
)
```

#### Hex3regsymm

Generate three-region hexagonal mesh (core/shell/barrier).

```python
from nwkpy.fem.mesh import Hex3regsymm

mesh = Hex3regsymm(
    mesh_name="three_region.msh",
    reg2mat={1: "InAs", 2: "GaSb", 3: "AlSb"},
    mat2partic={"InAs": "electron", "GaSb": "hole", "AlSb": "hole"},
    tw=15.0,    # Core width (nm)
    sw=5.0,     # Shell width (nm)
    bw=3.0,     # Barrier width (nm)
    np={'nC1': 8, 'nC2': 6, ...}  # Mesh density
)
```

### I/O Functions

**Location**: `nwkpy.fem.mesh.ffem_io`

FreeFem++ mesh file operations.

```python
from nwkpy.fem.mesh.ffem_io import ffmsh_2d_size_read, ffmsh_2d_data_read

# Read mesh file dimensions
v_num, e_num, t_num = ffmsh_2d_size_read("mesh.msh")
print(f"Mesh has {v_num} vertices, {e_num} edges, {t_num} triangles")

# Read complete mesh data
v_xy, v_l, e_v, e_l, t_v, t_l = ffmsh_2d_data_read("mesh.msh", v_num, e_num, t_num)
```

---

## Advanced Workflows

### Self-Consistent Schrödinger-Poisson

Complete workflow for self-consistent calculations.

```python
import numpy as np
from nwkpy import BandStructure, PoissonProblem
from nwkpy.physics import FreeChargeDensity, ElectrostaticPotential
from nwkpy.interface import Broyden
from nwkpy.fem.mesh import Mesh

# 1. Load mesh
mesh = Mesh("nanowire.msh", reg2mat={1: "InAs", 2: "GaSb"})

# 2. Set up calculation parameters
kz_values = np.linspace(0, 0.05, 20) * np.pi / np.sqrt(3) / 6.0583
dk = kz_values[1] - kz_values[0]
chemical_potential = 0.528  # eV
temperature = 4.0           # K
max_iterations = 25
convergence_threshold = 1e-3

# 3. Initialize with external field only
epot_init = ElectrostaticPotential(
    mesh.fs_el,
    electric_field=(0.1, np.pi/2)  # 0.1 V/μm in y-direction
)

# 4. Initialize Broyden mixer
broyden = Broyden(N=mesh.ng_nodes, M=6, beta=0.35)

# 5. Self-consistent loop
print("Starting self-consistent calculation...")
for iteration in range(max_iterations):
    print(f"Iteration {iteration + 1}/{max_iterations}")
    
    # a) Solve Schrödinger equation
    bs = BandStructure(
        mesh=mesh,
        kzvals=kz_values,
        valence_band_edges={"InAs": 0.0, "GaSb": 0.56},
        epot=epot_init,
        temperature=temperature,
        k=20
    )
    bs.run()
    
    # b) Calculate charge densities
    rho_el = FreeChargeDensity(bs.fs_el)
    rho_h = FreeChargeDensity(bs.fs_h)
    
    rho_el.add_charge(
        bs.psi_el, bs.bands, dk, chemical_potential, temperature,
        modified_EFA=True, particle='electron',
        norm_sum_region=bs.norm_sum_region, thr_el=0.8
    )
    rho_h.add_charge(
        bs.psi_h, bs.bands, dk, chemical_potential, temperature,
        modified_EFA=True, particle='hole', 
        norm_sum_region=bs.norm_sum_region, thr_h=0.95
    )
    
    # c) Solve Poisson equation
    poisson = PoissonProblem(
        mesh=mesh,
        dirichlet={'ref': None, 1: 0.0},
        electric_field=(0.1, np.pi/2),
        rho_el=rho_el,
        rho_h=rho_h
    )
    poisson.run()
    V_out = poisson.epot.V
    
    # d) Check convergence
    n_total, p_total = rho_el.get_total_charge()
    total_charge = abs(n_total + p_total)
    
    if iteration > 0:
        charge_change = abs(total_charge - prev_total_charge) / abs(prev_total_charge)
        print(f"  Total charge: {total_charge:.2e} cm⁻¹")
        print(f"  Relative change: {charge_change:.2e}")
        
        if charge_change < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations!")
            break
    
    # e) Apply Broyden mixing
    if iteration == 0:
        V_mixed = V_out
    else:
        V_mixed = broyden.update(xin=V_in, xout=V_out)
    
    # f) Prepare for next iteration
    epot_init = ElectrostaticPotential(mesh.fs_el, V=V_mixed)
    V_in = V_mixed
    prev_total_charge = total_charge

# 6. Final results
print(f"Final band structure calculated with {len(bs.bands)} k-points")
print(f"Energy range: {bs.bands.min():.3f} to {bs.bands.max():.3f} eV")

# 7. Visualization
fig_bands = bs.plot_bands(
    chemical_potential=chemical_potential,
    character_to_show='H-EL',
    xlim=(0, 0.3),
    ylim=(520, 540)
)

fig_potential = poisson.epot.plot(
    xlim=(-15, 15),
    ylim=(-15, 15),
    levels=25
)
```

### Multi-Parameter Study

Systematic exploration of parameter space.

```python
# Parameter ranges
field_values = [0.0, 0.1, 0.2, 0.3]  # V/μm
mu_values = np.linspace(0.520, 0.560, 9)  # eV
temperature_values = [4.0, 77.0, 300.0]  # K

results = {}

for T in temperature_values:
    for E_field in field_values:
        for mu in mu_values:
            print(f"Calculating: T={T}K, E={E_field}V/μm, μ={mu:.3f}eV")
            
            # Set up calculation
            epot = ElectrostaticPotential(
                mesh.fs_el, 
                electric_field=(E_field, np.pi/2)
            )
            
            bs = BandStructure(
                mesh=mesh,
                kzvals=kz_values,
                valence_band_edges={"InAs": 0.0, "GaSb": 0.56},
                epot=epot,
                temperature=T,
                k=15  # Reduced for speed in parameter sweep
            )
            bs.run()
            
            # Calculate observables
            band_gap = bs.bands.max() - bs.bands.min()
            electron_states = np.sum(bs.spinor_distribution[:, [0, 1], :], axis=1)
            hole_states = np.sum(bs.spinor_distribution[:, [2, 3, 4, 5, 6, 7], :], axis=1)
            
            # Store results
            results[(T, E_field, mu)] = {
                'bands': bs.bands.copy(),
                'band_gap': band_gap,
                'electron_character': electron_states.mean(),
                'hole_character': hole_states.mean(),
                'spinor_distribution': bs.spinor_distribution.copy()
            }

# Analyze results
print(f"Completed {len(results)} calculations")

# Create phase diagrams
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, T in enumerate(temperature_values):
    # Extract data for this temperature
    E_grid = []
    mu_grid = []
    gap_grid = []
    
    for (temp, E, mu), data in results.items():
        if temp == T:
            E_grid.append(E)
            mu_grid.append(mu)
            gap_grid.append(data['band_gap'] * 1000)  # meV
    
    # Create 2D plot
    scatter = axes[i].scatter(mu_grid, E_grid, c=gap_grid, cmap='viridis')
    axes[i].set_xlabel('Chemical Potential (eV)')
    axes[i].set_ylabel('Electric Field (V/μm)')
    axes[i].set_title(f'Band Gap at T={T}K')
    plt.colorbar(scatter, ax=axes[i], label='Band Gap (meV)')

plt.tight_layout()
plt.show()
```

### Advanced Charge Analysis

Detailed analysis of charge distribution and localization.

```python
# Regional charge analysis
def analyze_charge_distribution(bs, mesh, chemical_potential, temperature):
    """Analyze charge distribution in different regions"""
    
    # Create charge densities
    rho_el = FreeChargeDensity(bs.fs_el)
    rho_h = FreeChargeDensity(bs.fs_h)
    
    dk = bs.kzvals[1] - bs.kzvals[0] if len(bs.kzvals) > 1 else 0.001
    
    # Add charge with regional analysis
    rho_el.add_charge(
        bs.psi_el, bs.bands, dk, chemical_potential, temperature,
        modified_EFA=True, norm_sum_region=bs.norm_sum_region
    )
    rho_h.add_charge(
        bs.psi_h, bs.bands, dk, chemical_potential, temperature,
        modified_EFA=True, norm_sum_region=bs.norm_sum_region
    )
    
    # Calculate regional charges
    core_area = mesh.get_region_area([1])
    shell_area = mesh.get_region_area([2])
    
    # Sample charge densities
    n_points = 1000
    coords = np.random.uniform(-10, 10, (n_points, 2))  # Random sampling
    
    # Find which points are in which region
    core_mask = []
    shell_mask = []
    
    for coord in coords:
        try:
            # Use mesh triangulation to find element
            tri_idx = mesh.trifinder(coord[0], coord[1])
            if tri_idx >= 0:
                region = mesh.t_l[tri_idx]
                core_mask.append(region == 1)
                shell_mask.append(region == 2)
            else:
                core_mask.append(False)
                shell_mask.append(False)
        except:
            core_mask.append(False)
            shell_mask.append(False)
    
    core_mask = np.array(core_mask)
    shell_mask = np.array(shell_mask)
    
    # Get charge densities at sample points
    n_density, p_density = rho_el.interp(coords, total=False)
    
    # Calculate average densities per region
    core_n_avg = np.mean(n_density[core_mask]) if np.any(core_mask) else 0
    core_p_avg = np.mean(p_density[core_mask]) if np.any(core_mask) else 0
    shell_n_avg = np.mean(n_density[shell_mask]) if np.any(shell_mask) else 0
    shell_p_avg = np.mean(p_density[shell_mask]) if np.any(shell_mask) else 0
    
    return {
        'core_electron_density': core_n_avg,
        'core_hole_density': core_p_avg,
        'shell_electron_density': shell_n_avg,
        'shell_hole_density': shell_p_avg,
        'core_area': core_area,
        'shell_area': shell_area,
        'charge_separation': abs(core_n_avg - shell_p_avg)
    }

# Use the analysis
charge_analysis = analyze_charge_distribution(bs, mesh, 0.528, 4.0)

print("Charge Distribution Analysis:")
print(f"Core electron density: {charge_analysis['core_electron_density']:.2e} cm⁻³")
print(f"Core hole density: {charge_analysis['core_hole_density']:.2e} cm⁻³") 
print(f"Shell electron density: {charge_analysis['shell_electron_density']:.2e} cm⁻³")
print(f"Shell hole density: {charge_analysis['shell_hole_density']:.2e} cm⁻³")
print(f"Charge separation index: {charge_analysis['charge_separation']:.2e}")
```

---

## Error Handling and Debugging

### Common Exceptions

**FileNotFoundError**
```python
try:
    mesh = Mesh("nonexistent.msh")
except FileNotFoundError:
    print("Mesh file not found. Generate mesh first.")
```

**ValueError** 
```python
try:
    bs = BandStructure(mesh, kzvals=[], ...)  # Empty k-values
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

**MemoryError**
```python
import psutil

# Check available memory before large calculations
memory_gb = psutil.virtual_memory().available / (1024**3)
if memory_gb < 4.0:
    print(f"Warning: Only {memory_gb:.1f} GB RAM available")
```

**Convergence Issues**
```python
# Handle non-convergent eigenvalue problems
try:
    eigvals, eigvecs, ... = solver.solve(k=20, sigma=0.528, tol=1e-12)
except Exception as e:
    print(f"Convergence failed: {e}")
    # Try with relaxed tolerance
    eigvals, eigvecs, ... = solver.solve(k=15, sigma=0.530, tol=1e-8)
```

### Debugging Tools

```python
from nwkpy.interface.bandstructure import debug_write, MPI_debug_setup

# Enable MPI debugging
MPI_debug_setup("./debug_output")
debug_write("Starting calculation with parameters: ...")

# Validation functions
def validate_calculation(bs, mesh):
    """Validate calculation results"""
    
    # Check energy range
    if bs.bands.min() < -2.0 or bs.bands.max() > 3.0:
        print("Warning: Unusual energy range")
    
    # Check spinor normalization
    spinor_sum = np.sum(bs.spinor_distribution, axis=1)
    if not np.allclose(spinor_sum, 1.0, atol=1e-2):
        print("Warning: Spinor normalization issues")
    
    # Check mesh quality
    if mesh.nelem < 100:
        print("Warning: Very coarse mesh")
    
    return True

# Use validation
validate_calculation(bs, mesh)
```

---

## Performance Optimization

### Memory Management

```python
# Optimize for large calculations
def optimize_memory_usage(mesh, kz_values):
    """Optimize memory usage for large calculations"""
    
    # Estimate memory requirements
    n_nodes = mesh.ng_nodes
    n_k = len(kz_values)
    n_eigenvals = 20
    
    # Memory for wavefunctions (complex128)
    memory_mb = n_nodes * n_k * 8 * n_eigenvals * 16 / (1024**2)
    
    print(f"Estimated memory usage: {memory_mb:.1f} MB")
    
    if memory_mb > 1000:  # > 1 GB
        print("Large calculation detected. Consider:")
        print("- Reducing number of k-points")
        print("- Reducing number of eigenvalues")
        print("- Using coarser mesh")
    
    return memory_mb

# Check before calculation
optimize_memory_usage(mesh, kz_values)
```

### Parallel Scaling

```python
from mpi4py import MPI

def optimize_mpi_distribution(kz_values):
    """Optimize MPI process distribution"""
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    n_k = len(kz_values)
    
    if size > n_k:
        print(f"Warning: More processes ({size}) than k-points ({n_k})")
        print("Consider reducing number of processes")
    
    k_per_process = n_k // size
    remainder = n_k % size
    
    print(f"K-point distribution:")
    print(f"  {size} processes")
    print(f"  {k_per_process} k-points per process")
    print(f"  {remainder} processes get 1 extra k-point")
    
    return k_per_process

# Check MPI efficiency
if 'mpi4py' in globals():
    optimize_mpi_distribution(kz_values)
```

---

## Version Information and Compatibility

**API Version**: 0.1  
**Compatible Python**: 3.7+  
**Required Dependencies**: NumPy, SciPy, matplotlib  
**Optional Dependencies**: mpi4py (MPI), FreeFem++ (mesh generation)

### Import Structure

```python
# Main interface
from nwkpy import BandStructure, PoissonProblem

# Physics classes  
from nwkpy.physics import (
    FreeChargeDensity, 
    ElectrostaticPotential,
    WaveFunction,
    DopingChargeDensity
)

# Finite element framework
from nwkpy.fem.mesh import Mesh
from nwkpy.fem.element import FemSpace, FiniteElement
from nwkpy.fem.solver import GenEigenProblem, LinearSystem
from nwkpy.fem.problem import Schrodinger, Poisson

# Hamiltonians
from nwkpy.hamiltonian import HamiltonianZB
from nwkpy.hamiltonian_wz import HamiltonianWZ

# Utilities
from nwkpy.interface import Broyden, AngularMomentum
from nwkpy import _constants, _database
```

---

## Further Reading

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Installation Guide](INSTALLATION.md)** - Detailed setup including FreeFem++
- **[Physics Background](PHYSICS_BACKGROUND.md)** - k·p theory and FEM methods
- **[Script Documentation](SCRIPTS/)** - Detailed guides for mesh generation, band structure, and self-consistent calculations
- **[Tutorial Collection](TUTORIALS/)** - Step-by-step examples from basic to advanced

---

*This API reference documents nwkpy v0.1. For the most current information, please refer to the docstrings in the source code and the latest documentation.*, optional
    Mapping from region labels to material names. 
    Example: `{1: "InAs", 2: "GaSb"}`. Default: None.

**mat2partic** : `dict`, optional
    Mapping from materials to particle types.
    Example: `{"InAs": "electron", "GaSb": "hole"}`. Default: None.

**bandwidth_reduction** : `bool`, optional
    Apply Cuthill-McKee node reordering for efficiency. Default: True.

#### Attributes

**vertices** : `ndarray`
    Vertex coordinates, shape `(n_vertices, 2)` in nm.

**triangles** : `ndarray`
    Element connectivity, shape `(n_elements, 3)` with vertex indices.

**reg2mat** : `dict`
    