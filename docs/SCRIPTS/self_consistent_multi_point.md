# Self-Consistent Script Documentation

The self-consistent script performs advanced Schrödinger-Poisson calculations with multi-parameter sweeps over chemical potentials and electric fields. This is the most sophisticated calculation mode in nwkpy, featuring iterative coupling between electronic structure and electrostatics with accelerated convergence algorithms.

## Overview

**Purpose:** Self-consistent Schrödinger-Poisson calculations with parameter sweeps  
**Method:** Iterative coupling with Broyden mixing for rapid convergence  
**Input:** Pre-generated mesh files + parameter ranges  
**Output:** Converged results for all parameter combinations with convergence monitoring  
**Parallelization:** MPI over k-points with systematic parameter exploration  

## Key Features

- **Multi-parameter sweeps** - Systematic exploration of chemical potential and electric field space
- **Self-consistent coupling** - Electronic structure ↔ electrostatic potential feedback
- **Broyden mixing** - Advanced convergence acceleration beyond simple mixing
- **Convergence monitoring** - Real-time tracking of charge density residuals
- **Hierarchical output** - Organized directory structure for parameter combinations
- **Restart capability** - Continue from previous calculations or converged potentials

## Quick Usage

```bash
cd scripts/self_consistent/
python main.py
```

**For parallel execution:**
```bash
mpirun -np 4 python main.py  # MPI parallelization over k-points
```

**Expected runtime:** Hours to days depending on parameter ranges and convergence criteria

## File Structure

```
scripts/self_consistent/
├── main.py                    # Main execution script
├── indata.py                  # Input parameters (EDIT THIS)
├── mesh.msh                   # Input: mesh geometry
├── mesh.dat                   # Input: mesh metadata
├── outdata/                   # Hierarchical output structure
│   ├── OUT_EF_0/             # Electric field 0 results
│   │   ├── OUT_CP_0/         # Chemical potential 0 results
│   │   │   ├── bands.npy
│   │   │   ├── envelope_*.npy
│   │   │   ├── epot_init.npy     # Initial potential
│   │   │   ├── epot_conv.npy     # Converged potential
│   │   │   ├── n_resid.npy       # Convergence history
│   │   │   └── *.png             # Plots for this parameter set
│   │   ├── OUT_CP_1/         # Chemical potential 1 results
│   │   └── ...
│   ├── OUT_EF_1/             # Electric field 1 results
│   └── self_consistent.log   # Master execution log
```

## Input Parameters (`indata.py`)

### File Configuration

```python
directory_name = "outdata"       # Root output directory
mesh_name = "mesh"               # Mesh file base name
```

**Output organization:**
- Hierarchical structure: `outdata/OUT_EF_i/OUT_CP_j/`
- Each parameter combination gets separate directory
- Enables systematic parameter studies

### Execution Control

```python
generate_txt_files = False       # Human-readable text output
generate_png_graphs = False      # Generate plots (can be slow for many params)
MPI_debug = False               # MPI debugging output
```

**Performance considerations:**
- `generate_png_graphs = False` recommended for large parameter sweeps
- Can generate plots afterwards from saved data
- `generate_txt_files = True` useful for external analysis tools

### Material Properties

```python
material = ["InAs", "GaSb"]      # Core and shell materials
valence_band = [0.0, 0.56]       # Band alignment [core, shell] (eV)
carrier = ["electron", "hole"]   # Dominant carriers [core, shell]
```

**Must match mesh generation parameters exactly**

### Crystal Structure and Physical Conditions

```python
principal_axis_direction = '111' # Growth direction
lattice_constant = 6.0583        # Lattice parameter (Å)
temperature = 4.0                # Temperature (K)
```

### Multi-Parameter Sweeps

The key feature of this script is systematic parameter exploration:

#### Chemical Potential Sweep

```python
chemical_potential_set = [0.539144444444444, 0.55]  # List of μ values (eV)
```

**Examples:**
```python
# Fine sweep around band gap
chemical_potential_set = [0.520, 0.525, 0.530, 0.535, 0.540, 0.545, 0.550]

# Coarse sweep for overview
chemical_potential_set = [0.510, 0.530, 0.550, 0.570]

# Single value (no sweep)
chemical_potential_set = [0.528]

# Temperature-dependent gap
chemical_potential_set = np.linspace(0.515, 0.555, 9).tolist()
```

#### Electric Field Sweep

```python
electric_field_set = [
    (0.0, np.pi/2.),             # No field
    (0.2, np.pi/2.)              # 0.2 V/μm in y-direction
]
```

**Field specification:** `(magnitude_in_V_per_μm, angle_in_radians)`

**Examples:**
```python
# Field magnitude sweep
electric_field_set = [
    (0.0, 0.0),      # No field
    (0.1, 0.0),      # 0.1 V/μm in x
    (0.2, 0.0),      # 0.2 V/μm in x  
    (0.5, 0.0),      # 0.5 V/μm in x
]

# Field direction sweep  
electric_field_set = [
    (0.2, 0.0),          # x-direction
    (0.2, np.pi/4),      # 45° 
    (0.2, np.pi/2),      # y-direction
    (0.2, 3*np.pi/4),    # 135°
]

# 2D field map
magnitudes = [0.0, 0.1, 0.2]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
electric_field_set = [(E, θ) for E in magnitudes for θ in angles]
```

**Total calculations:** `len(electric_field_set) × len(chemical_potential_set)`

### Band Structure Parameters

```python
e_search = 0.539144444444444     # Eigenvalue search center (eV)
number_eigenvalues = 20          # Subbands per k-point
k_range = [0, 0.05]             # k-space range  
number_k_pts = 96               # Number of k-points
```

**Important for self-consistent:**
- `number_k_pts` should be divisible by number of MPI processes
- More k-points give better charge density resolution
- `e_search` should span all chemical potentials in sweep

### Self-Consistent Cycle Parameters

#### Convergence Criteria

```python
maxiter = 20                     # Maximum SCF iterations
maxchargeerror = 1e-3           # Charge density convergence threshold (relative)
maxchargeerror_dk = 1e-3        # Additional convergence parameter (legacy)
```

**Convergence logic:**
- Iteration stops when charge density changes < `maxchargeerror`
- Relative error: `|ρ_new - ρ_old| / |ρ_old| < maxchargeerror`
- Tighter criteria (1e-4, 1e-5) give better accuracy but slower convergence

**Typical values:**
```python
# Quick convergence (testing)
maxiter = 10
maxchargeerror = 1e-2

# Standard convergence
maxiter = 20  
maxchargeerror = 1e-3

# High precision (publication)
maxiter = 50
maxchargeerror = 1e-4
```

#### Initial Conditions

```python
init_pot_name = None             # Initial potential file path
```

**Options:**
- `None` - Start from Poisson solution (external field only)
- Path string - Load converged potential from previous calculation
- Enables continuation/restart of calculations

**Examples:**
```python
# Start from scratch
init_pot_name = None

# Continue from previous calculation  
init_pot_name = "outdata/OUT_EF_0/OUT_CP_0/epot_conv.npy"

# Use external potential
init_pot_name = "external_potential.npy"
```

### Spurious Solution Suppression

```python
rescale = ['S=0', 'S=0']         # P-parameter rescaling [core, shell]
modified_EFA = True              # Modified Envelope Function Approximation
character_threshold = [0.8, 0.95] # State classification [electron, hole]
```

**Same as band structure script** - see that documentation for details.

### Numerical Methods

```python
shape_function_kp = ['Hermite', 'LagrangeQuadratic']
shape_function_poisson = 'LagrangeQuadratic'
```

**Shape function selection affects:**
- Convergence stability
- Accuracy of self-consistent solution
- Computational cost per iteration

### Broyden Mixing Parameters

**Advanced convergence acceleration beyond simple linear mixing:**

```python
betamix = 0.35                   # Simple mixing parameter
maxter = 6                       # Max stored iterations for Broyden
w0 = 0.01                       # Weight for initial iteration
use_wm = True                   # Use iteration-dependent weights
toreset = []                    # Iterations to reset Broyden history
```

**Broyden mixing theory:**
- Combines multiple previous iterations to predict optimal potential
- Much faster convergence than simple mixing
- Particularly effective for charge transfer problems

**Parameter tuning:**
```python
# Conservative (stable, slower)
betamix = 0.1
maxter = 3

# Standard (balanced)  
betamix = 0.35
maxter = 6

# Aggressive (faster, may oscillate)
betamix = 0.7
maxter = 10

# Reset if oscillations occur
toreset = [5, 10, 15]  # Reset Broyden at these iterations
```

### Boundary Conditions

```python
dirichlet = {
    'ref': None,    # Default boundary condition
    1: 0.0,         # Fix boundary 1 to 0.0 eV (ground)
}
```

**Examples for different device geometries:**
```python
# Floating potential (intrinsic)
dirichlet = {'ref': None}

# Single contact
dirichlet = {'ref': None, 1: 0.0}

# Bias voltage
dirichlet = {'ref': None, 1: 0.0, 3: 0.1}  # 0.1 V bias

# All boundaries grounded
dirichlet = {'ref': 0.0}
```

### Plotting Configuration

```python
plotting_preferencies_bands = {
    'xlim': (0, 0.06),           # k-space range
    'ylim': (528, 545),          # Energy range (meV)
    'chemical_potential': 539.1444444,  # Fermi level marker
    'cmap_in': 'rainbow',
    'character_to_show': 'H-EL',
    # ... other parameters
}
```

**Self-consistent specific:**
- `chemical_potential` value should match one of the calculated values
- Consider wide energy range to show field effects
- Plot generation can be CPU-intensive for many parameter combinations

## Script Execution Flow

### 1. Multi-Parameter Loop Structure

```python
# Outer loop: Electric fields
for ii, ef in enumerate(electric_field_set):
    directory_ef = f"OUT_EF_{ii}"
    
    # Inner loop: Chemical potentials
    for i, mu in enumerate(chemical_potential_set):
        directory_mu = f"OUT_CP_{i}"
        path_mu = os.path.join(outdata_path, directory_ef, directory_mu)
        
        # Self-consistent calculation for this (E-field, μ) combination
        scf_calculation(ef, mu, path_mu)
```

### 2. Self-Consistent Cycle (for each parameter combination)

```python
# Initialize Broyden mixing
up = Broyden(N=Vin.shape[0], M=maxter, beta=betamix, w0=w0, use_wm=use_wm)

# SCF iteration loop
for j in range(maxiter):
    # 1. Solve Schrödinger equation with current potential
    bs = BandStructure(..., epot=ElectrostaticPotential(Vin))
    bs.run()
    
    # 2. Calculate charge densities from wavefunctions
    rho_el.add_charge(bs.psi_el, bs.bands, mu=mu, ...)
    rho_h.add_charge(bs.psi_h, bs.bands, mu=mu, ...)
    
    # 3. Solve Poisson equation with new charge densities
    p = PoissonProblem(..., rho_el=rho_el, rho_h=rho_h)
    p.run()
    Vout = p.epot.V
    
    # 4. Check convergence
    n_resid, p_resid = get_density_resid(rho_el, rho_h, rho_el_prev, rho_h_prev)
    if converged: break
    
    # 5. Apply Broyden mixing  
    Vout = up.update(xin=Vin, xout=Vout)
    Vin = Vout
```

### 3. MPI Parallelization Within SCF

Each SCF iteration uses MPI for k-point parallelization:
```python
# Distribute k-points across processes
kzslice = np.s_[rank*kmaxlocal:(rank+1)*kmaxlocal]

# Each process calculates its k-points
bs = BandStructure(..., kzvals=kzvals[kzslice])
bs.run()

# Collect results from all processes
comm.Allgather(bs.bands, complete_bands)
```

### 4. Convergence Monitoring

Real-time convergence tracking with detailed logging:
```python
logger.info(f'{iteration:>3} {total_charge:>12.4e} {max_potential_change:>12.4e} '
           f'{charge_residual:>12.4e} {relative_residual:>12.4e}')
```

**Log output example:**
```
(1)    (2)          (3)          (4)          (5)          (6)        (7)          (8)
 1  -2.1234e-06  1.4567e-02  8.9012e-03  3.4567e-02  1.2345e-02  4.5678e-03  2.3456e-03
 2  -1.8901e-06  7.8901e-03  4.5678e-03  1.7890e-02  9.8765e-03  2.3456e-03  1.4567e-03
...
```

**Column interpretation:**
1. Iteration number
2. Total charge balance
3. Maximum potential change
4. Mean absolute potential error
5. Negative charge residual
6. Relative negative residual (%)
7. Positive charge residual  
8. Relative positive residual (%)

## Understanding the Output

### Directory Structure

```
outdata/
├── OUT_EF_0/                    # Electric field 0: (0.0, π/2)
│   ├── OUT_CP_0/               # Chemical potential 0: 0.539 eV
│   │   ├── bands.npy           # Final converged band structure
│   │   ├── spinor_dist.npy     # Spinor character
│   │   ├── envelope_el.npy     # Electron envelopes
│   │   ├── envelope_h.npy      # Hole envelopes
│   │   ├── epot_init.npy       # Initial potential (before SCF)
│   │   ├── epot_conv.npy       # Final converged potential
│   │   ├── n_resid.npy         # Convergence history (negative charges)
│   │   ├── p_resid.npy         # Convergence history (positive charges)
│   │   ├── total_charge_init.npy # Initial charge balance
│   │   ├── total_charge_conv.npy # Final charge balance
│   │   ├── energy_bands.png    # Band structure plot
│   │   ├── carrier_density.png # Charge density plot
│   │   └── potential.png       # Electrostatic potential plot
│   ├── OUT_CP_1/               # Chemical potential 1: 0.55 eV
│   └── ...
├── OUT_EF_1/                   # Electric field 1: (0.2, π/2)
├── electric_field.npy          # Array of all electric field values
├── kzvals.npy                  # k-point array
└── self_consistent.log         # Master log file
```

### Convergence Analysis

#### Charge Residual Tracking

```python
# Load convergence history for specific parameter combination
n_resid = np.load('outdata/OUT_EF_0/OUT_CP_0/n_resid.npy')
p_resid = np.load('outdata/OUT_EF_0/OUT_CP_0/p_resid.npy')

# Plot convergence
import matplotlib.pyplot as plt
plt.semilogy(n_resid, label='Negative charge residual')
plt.semilogy(p_resid, label='Positive charge residual')
plt.xlabel('SCF Iteration')
plt.ylabel('Charge Residual (cm⁻¹)')
plt.legend()
```

#### Potential Evolution

```python
# Compare initial vs converged potential
V_init = np.load('outdata/OUT_EF_0/OUT_CP_0/epot_init.npy')
V_conv = np.load('outdata/OUT_EF_0/OUT_CP_0/epot_conv.npy')

potential_change = np.abs(V_conv - V_init)
print(f"Max potential change: {potential_change.max():.4f} eV")
print(f"RMS potential change: {np.sqrt(np.mean(potential_change**2)):.4f} eV")
```

### Multi-Parameter Analysis

#### Parameter Sweep Results

```python
# Analyze results across parameter space
electric_fields = np.load('outdata/electric_field.npy')
chemical_potentials = [0.539144444444444, 0.55]

results = {}
for i, ef in enumerate(electric_fields):
    for j, mu in enumerate(chemical_potentials):
        path = f'outdata/OUT_EF_{i}/OUT_CP_{j}/'
        
        # Load key results
        bands = np.load(path + 'bands.npy')
        total_charge = np.load(path + 'total_charge_conv.npy')
        
        results[(ef[0], mu)] = {
            'band_gap': calculate_band_gap(bands, mu),
            'total_charge': total_charge,
            'converged': True  # Check convergence criteria
        }
```

#### Phase Diagrams and Maps

```python
# Create parameter space maps
E_fields = [ef[0] for ef in electric_fields]
mu_values = chemical_potentials

# Extract quantity of interest (e.g., band gap)
band_gaps = np.zeros((len(E_fields), len(mu_values)))
for i, E in enumerate(E_fields):
    for j, mu in enumerate(mu_values):
        band_gaps[i, j] = results[(E, mu)]['band_gap']

# Plot 2D parameter map
plt.imshow(band_gaps, extent=[min(mu_values), max(mu_values), 
                             min(E_fields), max(E_fields)],
           aspect='auto', origin='lower')
plt.xlabel('Chemical Potential (eV)')
plt.ylabel('Electric Field (V/μm)')
plt.colorbar(label='Band Gap (eV)')
```

## Common Use Cases

### Example 1: Chemical Potential Sweep at Zero Field

```python
# Study intrinsic carrier density vs Fermi level
electric_field_set = [(0.0, 0.0)]  # No external field
chemical_potential_set = np.linspace(0.520, 0.560, 9).tolist()

# Results: carrier density vs chemical potential
```

### Example 2: Electric Field Response

```python
# Study field-induced band bending
chemical_potential_set = [0.528]  # Fixed Fermi level
electric_field_set = [
    (0.0, np.pi/2),   # No field
    (0.1, np.pi/2),   # 0.1 V/μm
    (0.2, np.pi/2),   # 0.2 V/μm  
    (0.5, np.pi/2),   # 0.5 V/μm
]

# Results: band structure evolution with field
```

### Example 3: 2D Parameter Map

```python
# Comprehensive parameter exploration
E_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # V/μm
mu_values = np.linspace(0.520, 0.560, 5)    # eV

electric_field_set = [(E, np.pi/2) for E in E_values]
chemical_potential_set = mu_values.tolist()

# Results: 6×5 = 30 calculations
# Creates full (E-field, μ) phase diagram
```

### Example 4: Convergence Study

```python
# Compare different mixing parameters
parameter_sets = [
    {'betamix': 0.1, 'maxter': 3},   # Conservative
    {'betamix': 0.35, 'maxter': 6},  # Standard
    {'betamix': 0.7, 'maxter': 10},  # Aggressive
]

# Run same physical system with different numerical parameters
# Compare convergence rates and stability
```

### Example 5: Temperature Series

```python
# Study temperature effects (requires multiple runs)
temperatures = [4.0, 77.0, 300.0]  # K

for T in temperatures:
    # Update temperature in indata.py
    # Run self-consistent calculation
    # Compare band structures and charge distributions
```

## Troubleshooting

### Convergence Issues

#### Problem: Oscillating Convergence

**Symptoms:**
```
SCF Iteration 15: Residual = 1.2e-2
SCF Iteration 16: Residual = 1.8e-2  
SCF Iteration 17: Residual = 1.1e-2
...
```

**Solutions:**
1. **Reduce mixing parameter:**
   ```python
   betamix = 0.1  # More conservative
   ```

2. **Reset Broyden periodically:**
   ```python
   toreset = [5, 10, 15]  # Reset every 5 iterations
   ```

3. **Increase Broyden memory:**
   ```python
   maxter = 3  # Store fewer iterations (more stable)
   ```

#### Problem: Slow Convergence

**Symptoms:**
```
SCF reaches maxiter = 20 without convergence
Final residual > maxchargeerror
```

**Solutions:**
1. **Relax convergence criteria:**
   ```python
   maxchargeerror = 1e-2  # Less strict
   maxiter = 50           # More iterations
   ```

2. **Better initial guess:**
   ```python
   init_pot_name = "converged_potential_nearby_parameters.npy"
   ```

3. **Improve physical parameters:**
   ```python
   # Ensure chemical potential is reasonable
   chemical_potential = 0.528  # Near band gap
   
   # Check mesh quality
   # Use results from mesh generation diagnostics
   ```

#### Problem: Charge Imbalance

**Symptoms:**
```
Total charge = 1.5e-2 e/cm³  # Should be ~1e-10 for intrinsic
```

**Solutions:**
1. **Check parameter consistency:**
   ```python
   # Verify materials match mesh generation
   material = ["InAs", "GaSb"]  # Must be identical
   ```

2. **Improve k-point sampling:**
   ```python
   number_k_pts = 96  # More k-points for better integration
   ```

3. **Check boundary conditions:**
   ```python
   # Ensure appropriate boundary conditions
   dirichlet = {'ref': None, 1: 0.0}  # Proper grounding
   ```

### Performance Issues

#### Problem: Memory Errors

**Error:**
```
MemoryError: Unable to allocate array of shape (10000, 40, 8, 20)
```

**Solutions:**
1. **Reduce calculation size:**
   ```python
   number_k_pts = 48         # Fewer k-points
   number_eigenvalues = 15   # Fewer states
   ```

2. **Use more MPI processes:**
   ```bash
   mpirun -np 8 python main.py  # Distribute memory load
   ```

3. **Process parameter combinations separately:**
   ```python
   # Split large parameter sweeps into smaller chunks
   chemical_potential_set = [0.520, 0.525]  # First batch
   # Run, then continue with [0.530, 0.535], etc.
   ```

#### Problem: Very Slow Execution

**Symptoms:**
- Hours per SCF iteration
- Single parameter combination takes days

**Solutions:**
1. **Optimize mesh:**
   ```python
   # Use coarser mesh from mesh generation
   edges = [8, 5, 6, 5, 4, 5]  # Reduce mesh density
   ```

2. **Reduce precision:**
   ```python
   maxchargeerror = 1e-2     # Faster convergence
   maxiter = 15              # Fewer max iterations
   ```

3. **Parallel execution:**
   ```bash
   # Use all available cores
   mpirun -np $(nproc) python main.py
   ```

### Parameter-Specific Issues

#### Problem: Unphysical Results

**Symptoms:**
- Negative band gaps
- Extremely large charge densities
- Potential values >> 1 eV

**Solutions:**
1. **Check material parameters:**
   ```python
   # Verify band alignment is reasonable
   valence_band = [0.0, 0.56]  # Typical InAs/GaSb
   ```

2. **Validate chemical potential range:**
   ```python
   # Ensure μ values are near band gap
   chemical_potential_set = [0.520, 0.530, 0.540]  # Reasonable range
   ```

3. **Check electric field values:**
   ```python
   # Avoid extremely large fields
   electric_field_set = [(0.1, 0.0)]  # Moderate field, not (10.0, 0.0)
   ```

### Restart and Continuation

#### Restarting Failed Calculations

```python
# Use converged potential from previous run
init_pot_name = "outdata/OUT_EF_0/OUT_CP_0/epot_conv.npy"

# Or partially converged potential
init_pot_name = "outdata/OUT_EF_0/OUT_CP_0/epot_init.npy"
```

#### Continuing Parameter Sweeps

```python
# Remove completed parameter combinations from sets
electric_field_set = [(0.3, np.pi/2), (0.4, np.pi/2)]  # Continue from where stopped
chemical_potential_set = [0.545, 0.550, 0.555]         # Remaining values
```

## Performance Guidelines

### Computational Scaling

**Per parameter combination:**

| K-points | Eigenvalues | SCF Iterations | Time (4 cores) | Memory |
|----------|-------------|----------------|-----------------|--------|
| 20       | 20          | 10             | ~30 min         | ~2 GB  |
| 48       | 20          | 15             | ~2 hours        | ~4 GB  |
| 96       | 30          | 20             | ~8 hours        | ~12 GB |

**Total time = single_time × n_E_fields × n_chemical_potentials**

### Recommended Strategies

#### Development/Testing Phase
```python
# Quick parameter exploration
electric_field_set = [(0.0, 0.0), (0.2, np.pi/2)]
chemical_potential_set = [0.528, 0.540]
maxiter = 10
maxchargeerror = 1e-2
# Total: 4 calculations, ~2 hours
```

#### Production Calculations
```python
# Systematic study
electric_field_set = [(E, np.pi/2) for E in np.linspace(0, 0.5, 6)]
chemical_potential_set = np.linspace(0.520, 0.560, 9).tolist()
maxiter = 25
maxchargeerror = 1e-3
# Total: 54 calculations, ~1 week on 4 cores
```

#### High-Precision Studies
```python
# Publication-quality results
electric_field_set = [(E, np.pi/2) for E in np.linspace(0, 0.5, 11)]
chemical_potential_set = np.linspace(0.515, 0.565, 21).tolist()
number_k_pts = 96
maxiter = 50
maxchargeerror = 1e-4
# Total: 231 calculations, several weeks on cluster
```

## Integration and Workflow

### Relationship to Other Scripts

#### Sequential Workflow
```bash
# 1. Generate mesh
cd mesh_generation/
python main.py

# 2. Basic band structure (optional - for validation)
cd ../band_structure/
cp ../mesh_generation/outdata/mesh.* ./
python main.py

# 3. Self-consistent calculations
cd ../self_consistent/
cp ../mesh_generation/outdata/mesh.* ./
python main.py
```

#### Parameter Consistency

**All scripts must have matching:**
```python
material = ["InAs", "GaSb"]
carrier = ["electron", "hole"]
valence_band = [0.0, 0.56]
temperature = 4.0
principal_axis_direction = '111'
lattice_constant = 6.0583
```

### Post-Processing Workflow

#### Analysis Scripts

```python
# analyze_parameter_sweep.py
import numpy as np
import matplotlib.pyplot as plt

def analyze_sweep(outdata_path):
    """Analyze results from parameter sweep"""
    
    # Load parameter arrays
    E_fields = np.load(f'{outdata_path}/electric_field.npy')
    mu_values = [0.539144444444444, 0.55]  # From indata.py
    
    # Extract results
    results = {}
    for i, ef in enumerate(E_fields):
        for j, mu in enumerate(mu_values):
            path = f'{outdata_path}/OUT_EF_{i}/OUT_CP_{j}/'
            
            try:
                bands = np.load(path + 'bands.npy')
                charge = np.load(path + 'total_charge_conv.npy')
                n_resid = np.load(path + 'n_resid.npy')
                
                results[(ef[0], mu)] = {
                    'bands': bands,
                    'charge': charge,
                    'convergence_iterations': len(n_resid),
                    'final_residual': n_resid[-1] if len(n_resid) > 0 else np.inf,
                    'converged': n_resid[-1] < 1e-3 if len(n_resid) > 0 else False
                }
            except FileNotFoundError:
                results[(ef[0], mu)] = {'converged': False}
    
    return results

# Create summary plots
def plot_parameter_map(results, quantity='band_gap'):
    """Create 2D parameter space map"""
    
    E_values = sorted(set(key[0] for key in results.keys()))
    mu_values = sorted(set(key[1] for key in results.keys()))
    
    data = np.zeros((len(E_values), len(mu_values)))
    
    for i, E in enumerate(E_values):
        for j, mu in enumerate(mu_values):
            if (E, mu) in results and results[(E, mu)]['converged']:
                if quantity == 'band_gap':
                    bands = results[(E, mu)]['bands']
                    # Calculate band gap (simplified)
                    data[i, j] = bands.max() - bands.min()
                elif quantity == 'convergence_iterations':
                    data[i, j] = results[(E, mu)]['convergence_iterations']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(data, extent=[min(mu_values), max(mu_values),
                            min(E_values), max(E_values)],
               aspect='auto', origin='lower')
    plt.xlabel('Chemical Potential (eV)')
    plt.ylabel('Electric Field (V/μm)')
    plt.colorbar(label=quantity.replace('_', ' ').title())
    plt.title(f'{quantity.replace("_", " ").title()} Map')
    
if __name__ == "__main__":
    results = analyze_sweep('outdata')
    plot_parameter_map(results, 'band_gap')
    plot_parameter_map(results, 'convergence_iterations')
    plt.show()
```

### Data Export and Sharing

#### HDF5 Export for Large Datasets

```python
# export_hdf5.py
import h5py
import numpy as np

def export_to_hdf5(outdata_path, output_file='results.h5'):
    """Export all results to HDF5 for efficient storage and sharing"""
    
    with h5py.File(output_file, 'w') as f:
        # Create groups for organization
        params_grp = f.create_group('parameters')
        results_grp = f.create_group('results')
        
        # Store parameter arrays
        E_fields = np.load(f'{outdata_path}/electric_field.npy')
        params_grp.create_dataset('electric_fields', data=E_fields)
        
        # Store results for each parameter combination
        for i in range(len(E_fields)):
            for j in range(len(chemical_potential_set)):
                try:
                    path = f'{outdata_path}/OUT_EF_{i}/OUT_CP_{j}/'
                    
                    # Create group for this parameter combination
                    param_grp = results_grp.create_group(f'EF_{i}_CP_{j}')
                    
                    # Store all arrays
                    param_grp.create_dataset('bands', data=np.load(path + 'bands.npy'))
                    param_grp.create_dataset('spinor_dist', data=np.load(path + 'spinor_dist.npy'))
                    param_grp.create_dataset('potential_final', data=np.load(path + 'epot_conv.npy'))
                    param_grp.create_dataset('convergence_history', data=np.load(path + 'n_resid.npy'))
                    
                    # Store metadata
                    param_grp.attrs['electric_field'] = E_fields[i]
                    param_grp.attrs['chemical_potential'] = chemical_potential_set[j]
                    param_grp.attrs['converged'] = True
                    
                except FileNotFoundError:
                    # Mark failed calculations
                    param_grp = results_grp.create_group(f'EF_{i}_CP_{j}')
                    param_grp.attrs['converged'] = False

# Usage
export_to_hdf5('outdata', 'nwkpy_results.h5')
```

## Advanced Usage and Extensions

### Custom Convergence Criteria

```python
# Advanced convergence monitoring
def custom_convergence_check(rho_el, rho_h, rho_el_prev, rho_h_prev, 
                           iteration, max_potential_change):
    """Custom convergence logic"""
    
    # Standard charge residual
    n_resid, p_resid = get_density_resid(rho_el, rho_h, rho_el_prev, rho_h_prev)
    
    # Additional criteria
    charge_converged = (n_resid < 1e-3) and (p_resid < 1e-3)
    potential_converged = max_potential_change < 1e-3
    min_iterations = iteration > 5  # Minimum iterations
    
    # Combined logic
    converged = charge_converged and potential_converged and min_iterations
    
    return converged, {'n_resid': n_resid, 'p_resid': p_resid}
```

### Adaptive Parameter Sweeps

```python
# Adaptive parameter selection based on previous results
def adaptive_sweep(initial_params, convergence_map):
    """Refine parameter grid in regions of interest"""
    
    # Identify regions with poor convergence
    problem_regions = [(E, mu) for (E, mu), data in convergence_map.items() 
                      if not data['converged'] or data['final_residual'] > 1e-2]
    
    # Add intermediate parameter points
    new_params = []
    for E, mu in problem_regions:
        # Add points around problematic parameters
        for dE in [-0.05, 0.05]:
            for dmu in [-0.005, 0.005]:
                new_params.append((E + dE, mu + dmu))
    
    return new_params
```

### Multi-Scale Analysis

```python
# Analyze results at different length scales
def multiscale_analysis(envelope_functions, mesh):
    """Analyze wavefunction localization at different scales"""
    
    # Core vs shell localization
    core_region = mesh.t_l == 1  # Core elements
    shell_region = mesh.t_l == 2  # Shell elements
    
    core_weight = np.sum(np.abs(envelope_functions)**2 * core_region)
    shell_weight = np.sum(np.abs(envelope_functions)**2 * shell_region)
    
    localization_ratio = core_weight / shell_weight
    
    # Interface analysis (elements near core-shell boundary)
    interface_elements = find_interface_elements(mesh)
    interface_weight = np.sum(np.abs(envelope_functions)**2 * interface_elements)
    
    return {
        'core_localization': core_weight,
        'shell_localization': shell_weight,
        'interface_localization': interface_weight,
        'core_shell_ratio': localization_ratio
    }
```

## Validation and Benchmarking

### Convergence Validation

```python
# Validate convergence by comparing with different numerical settings
validation_sets = [
    {'maxchargeerror': 1e-2, 'betamix': 0.3},   # Standard
    {'maxchargeerror': 1e-3, 'betamix': 0.3},   # Tighter convergence
    {'maxchargeerror': 1e-3, 'betamix': 0.1},   # More conservative mixing
]

# Run same physical system with different numerical parameters
# Compare final band structures and charge densities
```

### Mesh Convergence Study

```python
# Study convergence with respect to mesh density
mesh_densities = [
    [5, 3, 4, 3, 3, 3],     # Coarse
    [10, 5, 7, 6, 5, 5],    # Standard  
    [15, 10, 12, 10, 8, 10], # Fine
]

# Generate different meshes and compare self-consistent results
# Look for convergence of:
# - Band gap values
# - Charge densities
# - Electrostatic potentials
```

### Comparison with Non-Self-Consistent

```python
# Compare self-consistent vs non-self-consistent results
def compare_scf_vs_non_scf():
    """Compare self-consistent with band structure script results"""
    
    # Load non-self-consistent results (from band structure script)
    bands_non_scf = np.load('../band_structure/outdata/bands.npy')
    potential_non_scf = np.load('../band_structure/outdata/potential.npy')
    
    # Load self-consistent results
    bands_scf = np.load('outdata/OUT_EF_0/OUT_CP_0/bands.npy')
    potential_scf = np.load('outdata/OUT_EF_0/OUT_CP_0/epot_conv.npy')
    
    # Compare differences
    band_shift = np.mean(bands_scf - bands_non_scf)
    potential_difference = np.abs(potential_scf - potential_non_scf)
    
    print(f"Average band shift due to self-consistency: {band_shift*1000:.1f} meV")
    print(f"RMS potential difference: {np.sqrt(np.mean(potential_difference**2)):.4f} eV")
    
    return {
        'band_shift': band_shift,
        'potential_rms_diff': np.sqrt(np.mean(potential_difference**2))
    }
```

## Computational Considerations

### Memory Management

```python
# Monitor and optimize memory usage
import psutil
import gc

def memory_checkpoint(label):
    """Log memory usage at checkpoints"""
    memory_percent = psutil.virtual_memory().percent
    memory_gb = psutil.virtual_memory().used / (1024**3)
    logger.info(f"Memory at {label}: {memory_gb:.1f} GB ({memory_percent:.1f}%)")

# Use throughout calculation
memory_checkpoint("Start of SCF iteration")
# ... calculation steps ...
gc.collect()  # Force garbage collection
memory_checkpoint("End of SCF iteration")
```

### Parallel Efficiency

```python
# Analyze MPI scaling efficiency
def analyze_parallel_efficiency():
    """Study parallel scaling with different process counts"""
    
    process_counts = [1, 2, 4, 8, 16]
    timings = {}
    
    for n_proc in process_counts:
        # Run same calculation with different process counts
        # Measure time per SCF iteration
        start_time = time.time()
        # ... run calculation ...
        elapsed = time.time() - start_time
        
        timings[n_proc] = elapsed
    
    # Calculate parallel efficiency
    serial_time = timings[1]
    for n_proc in process_counts:
        efficiency = serial_time / (n_proc * timings[n_proc])
        print(f"{n_proc} processes: {efficiency:.2f} efficiency")
```

### Cluster Integration

```bash
#!/bin/bash
# slurm_submit.sh - Example SLURM submission script

#SBATCH --job-name=nwkpy_scf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00
#SBATCH --mem=64GB

module load python/3.9
module load openmpi/4.0

# Set up environment
source activate nwkpy_env

# Run calculation
mpirun -np $SLURM_NTASKS python main.py

# Submit with: sbatch slurm_submit.sh
```

## Related Documentation

- **[Mesh Generation Script](mesh_generation.md)** - Required prerequisite
- **[Band Structure Script](band_structure.md)** - Simpler non-self-consistent version
- **[Physics Background](../PHYSICS_BACKGROUND.md)** - Schrödinger-Poisson theory
- **[Installation Guide](../INSTALLATION.md)** - MPI and parallel computing setup

## Next Steps

✅ **Self-consistent calculation completed successfully?**

🎯 **Advanced analysis:**
1. **Parameter space exploration** - Create phase diagrams and property maps
2. **Comparison studies** - Validate against experimental data or literature
3. **Multi-physics extensions** - Include strain, magnetic fields, or phonon coupling
4. **High-throughput screening** - Automated exploration of material combinations

📊 **Research applications:**
- **Device modeling** - Gate response, contact effects, transport properties
- **Material discovery** - Systematic exploration of heterostructure combinations  
- **Temperature studies** - Thermal effects on electronic structure
- **Field engineering** - Tuning band structure with external fields

💡 **Code development:**
- **Custom material parameters** - Extend material database
- **Alternative geometries** - Beyond hexagonal nanowires
- **Advanced convergence** - Machine learning accelerated SCF
- **Post-processing tools** - Automated analysis and visualization

---

**Congratulations!** You now have complete documentation for all three core nwkpy scripts. This provides a comprehensive foundation for users to perform sophisticated nanowire calculations with full understanding of the underlying physics and numerical methods.