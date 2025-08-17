# Quick Start Guide

Get up and running with nwkpy in 5 minutes! This guide walks you through your first nanowire band structure calculation.

## Prerequisites

- Python 3.7+ with NumPy, SciPy, matplotlib
- MPI implementation (OpenMPI or MPICH) 
- FreeFem++ (optional, for advanced mesh generation)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nwkpy.git
cd nwkpy

# Install in development mode
pip install -e .

# Install MPI support
pip install mpi4py
```

## Your First Calculation: InAs/GaSb Core-Shell Nanowire

nwkpy uses a three-step workflow for nanowire calculations:

### Step 1: Generate Mesh

Create a finite element mesh for your nanowire geometry:

```bash
cd scripts/mesh_generation
python main.py
```

**What this does:** Creates a hexagonal cross-section mesh with core (InAs) and shell (GaSb) regions. The output files `mesh.msh` and `mesh.dat` contain the finite element grid and material assignments.

**Key parameters** (edit `indata.py`):
```python
material = ["InAs", "GaSb"]         # Core and shell materials
width = [7.5, 4.88]                 # Core radius, shell thickness (nm)
edges = [10, 5, 7, 6, 5, 5]         # Mesh density on each border
```

### Step 2: Calculate Band Structure

Compute the electronic band structure using the 8-band k·p method:

```bash
cd ../band_structure
python main.py
```

**What this does:** Solves the 8-band k·p Hamiltonian for a range of k-points along the nanowire axis. Calculates electron and hole states, charge densities, and electrostatic potential.

**Key parameters** (edit `indata.py`):
```python
chemical_potential = 0.528          # Fermi level (eV)
temperature = 4.0                   # Temperature (K)
k_range = [0, 0.05]                 # k-space range
number_k_pts = 20                   # Number of k-points
number_eigenvalues = 20             # Number of subbands
```

**Expected output:**
- Band structure data (`bands.npy`)
- Visualization plots (`energy_bands.png`, `carrier_density.png`, `potential.png`)
- Charge density results

### Step 3: Self-Consistent Calculation (Optional)

For more accurate results with charge redistribution effects:

```bash
cd ../self_consistent
python main.py
```

**What this does:** Solves the coupled Schrödinger-Poisson equations self-consistently. Accounts for electrostatic potential changes due to charge redistribution.

## Understanding Your Results

### Band Structure Plot (`energy_bands.png`)
- **X-axis**: k-vector along nanowire (nm⁻¹)
- **Y-axis**: Energy (meV)
- **Colors**: Electron (blue) vs hole (red) character
- **Horizontal line**: Chemical potential (Fermi level)

### Charge Density Plot (`carrier_density.png`)
- **2D contour map** of electron and hole densities
- **Core-shell interface** visible as material boundary
- **Charge localization** shows where carriers prefer to reside

### Key Physics to Look For

1. **Band Alignment**: InAs/GaSb forms a broken-gap (Type III) heterostructure
2. **Charge Separation**: Electrons prefer InAs core, holes prefer GaSb shell
3. **Quantum Confinement**: Discrete energy levels due to nanowire geometry
4. **Interface Effects**: Band bending and charge accumulation at core-shell boundary

## Common Parameter Adjustments

### Change Materials
```python
material = ["GaAs", "AlGaAs"]       # Different material system
valence_band = [0.0, 0.3]           # Adjust band offsets (eV)
```

### Temperature Effects
```python
temperature = 77.0                  # Liquid nitrogen temperature
temperature = 300.0                 # Room temperature
```

### External Electric Field
```python
electric_field = (0.1, np.pi/2)     # 0.1 V/μm in y-direction
```

### Mesh Resolution
```python
edges = [20, 10, 14, 12, 10, 10]    # Higher resolution (slower)
edges = [5, 3, 4, 3, 3, 3]          # Lower resolution (faster)
```

## Troubleshooting

### Common Issues

**"Mesh file not found"**
- Run mesh generation script first
- Check that `mesh.msh` and `mesh.dat` exist in the same directory

**"Material not in database"**
- Use materials from the supported list: InAs, GaAs, GaSb, InP, AlAs, etc.
- Check spelling and capitalization

**"MPI Error"**
- For single-core calculations: `python main.py`
- For parallel calculations: `mpirun -np 4 python main.py`

**Convergence Problems (Self-Consistent)**
- Reduce `betamix` parameter (try 0.1-0.3)
- Increase `maxiter` (try 30-50)
- Check chemical potential is reasonable (near band gap)

### Performance Tips

- **Start small**: Use coarse mesh and few k-points for testing
- **Parallel runs**: Use MPI for k-point calculations: `mpirun -np 4 python main.py`
- **Memory**: Large meshes require significant RAM (~1-8 GB)

## Next Steps

✅ **Completed your first calculation?** Great!

📖 **Learn more**:
- [Installation Guide](INSTALLATION.md) - Detailed setup including FreeFem++
- [Physics Background](PHYSICS_BACKGROUND.md) - k·p theory and FEM methods
- [API Reference](API_REFERENCE.md) - Complete class documentation

🎯 **Try more examples**:
- [InAs/GaSb Tutorial](TUTORIALS/01_basic_mesh.md) - Detailed walkthrough
- [Electric Field Effects](TUTORIALS/02_band_structure.md) - External field calculations
- [Parameter Sweeps](TUTORIALS/03_self_consistent.md) - Multi-parameter studies

💡 **Advanced topics**:
- Custom geometries with FreeFem++
- Multi-material heterostructures  
- Temperature-dependent calculations
- Comparison with experimental data

---

**Questions?** Open an issue on GitHub or check our [documentation](../README.md) for more detailed guides.