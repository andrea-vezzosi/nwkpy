# nwkp<sub>y</sub> 

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](docs/)

**nwkp<sub>y</sub>** is a numerical library for calculating electronic band structures of semiconductor nanowires using the 8-band k·p Hamiltonian with real-space discretization using FEM. It supports radial material modulation and doping within the self-consistent coupled Schrödinger-Poisson problem. Inhomogenous grids and advanced numerical techniques ensure computational optimization and accuracy.

**nwkp<sub>y</sub>** is a Object Oriented library written in Python. Its classes and functions allow to write flexible, numerically efficient scrpits to calculate the band structure of nanowires with different composition, dopint, gemotry, growth axes. The library is equipped with ready-to-use, fully documented scripts for homogeneous and core-single-shell exagonal nanowires covering most common nanowire classes.

## Key Features

- **8-band k·p Hamiltonian**: Complete treatment of conduction and valence bands including spin-orbit coupling
- **Finite Element Method (FEM)**: Flexible spatial discretization with FreeFem++ integration
- **Core-Shell Nanowires**: Specialized support for hexagonal cross-section heterostructures  
- **Self-Consistent Coupling**: Schrödinger-Poisson equations with Broyden mixing for rapid convergence
- **Advanced Numerics**: Spurious solution suppression
- **Modified Envelope Function Approximation (MEFA)**: Self-consistent approach for broken-gap alignements
- **High Performance**: MPI parallelization, optimized sparse matrix solvers, inhomogeneous mesh support
- **Comprehensive Analysis**: Band structure visualization, charge density plots, electrostatic potential mapping

## Physics Background

**nwkp<sub>y</sub>** implements state-of-the-art methods for semiconductor nanostructure calculations:

- **k·p Theory**: 8-band Kane model with P-parameter rescaling to eliminate spurious solutions
- **Heterostructure Physics**: Type I/II band alignments, broken-gap structures, carrier localization
- **Electrostatics**: Self-consistent treatment of built-in fields, external electric fields, charge redistribution
- **Material Systems**: Comprehensive database of III-V semiconductors (InAs, GaAs, GaSb, InP, etc.)

## Quick Start

### Installation

**Option 1: Using Anaconda/Miniconda (Recommended)**

If you don't have Anaconda/Miniconda installed:
```bash
# Download and install Miniconda (lighter than full Anaconda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the installer prompts, then restart your terminal
```

Then install **nwkp<sub>y</sub>**:
```bash
# Create a new environment with Python 3.8+
conda create -n nwkpy python=3.8 numpy scipy matplotlib mpi4py
conda activate nwkpy

# Clone and install nwkp<sub>y</sub>
git clone https://github.com/andrea-vezzosi/nwkpy
cd nwkpy
pip install -e .
```

**Option 2: Using Python Virtual Environment (Unix)**

```bash
# Make sure you have Python 3.7+ installed
python3 --version

# Create virtual environment
python3 -m venv nwkpy-env
source nwkpy-env/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install numpy scipy matplotlib mpi4py

# Clone and install nwkp<sub>y</sub>
git clone https://github.com/andrea-vezzosi/nwkpy
cd nwkpy
pip install -e .
```

### Use scripts for homogeneous or core-shell hexagonal nanowires

**nwkp<sub>y</sub>** comes with ready-to-use scripts to performe band-structure calculations for the very common case of hexagonal section a core-shell nanowire. 

There are three scripts with common behaviours:
- **input file:** indata.py
- **script name:** main.py
- **output data directory:** chosen as an input
- **mesh files:** must be in the script directory

Workflow:

#### 1. Generate finite element mesh

```bash
cd scripts/mesh_generation/

# Choose material, width and grid densities in the core and shell regions
edit indata.py

# Generate mesh files
python main.py
```
#### 2. Calculate band structure (single field, non-self-consistent) 
```bash
cd scritps/band_structure

# Copy the generated mesh (the exact location is set in scripts/mesh_generation/indata.py)
cp ../mesh_generation/outdata/mesh.* .

# Choose carrier type, electric field, growth direction, k-range and a large set of available parameters
edit indata.py

# Perform band structure calculation
python main.py
```
#### 2. (alternatively) Self-consistent calculation with electric field and chemical potential sweeps

```bash
cd scripts/self_consistent_multi_point

# Copy the generated mesh (the exact location is set in scripts/mesh_generation/indata.py)
cp ../mesh_generation/outdata/mesh.* .

# Choose material, physics and convergence parameters and sweep selected electric fields and chemical potentials
edit indata.py

# Perform self-consistent band structure calculation
python main.py
```

#### 3. (optional) Generate publicaiton-quality plots
```bash
# Use scripts/band_structure/main.py in plot_only_mode
cd scritps/band_structure

# Copy .npy files from a previous calculation (either from the band_structure or the self_consistten_single_point scripts)
cp <directoyry>/*.npy ./oudata/

# edit indata.py and tailor plotting preferences at your needs
plot_only_mode = True
plotting_preferencies_bands = { <parameters> }
plotting_preferencies_potential = { <parameters> }
plotting_preferencies_density = { <parameters> }

# produce *.png graphs
python main.py

```



<!--
### Simple Example

Classes and functions in **nwkp<sub>y</sub>** allow to write tailoered scripts, extend calculations to more layers or different geometries. Here is a simple example of a script:

(not working yet, to be revise)

```python
import numpy as np
from nwkp<sub>y</sub>.fem import Mesh
from nwkp<sub>y</sub> import BandStructure

# Load pre-generated mesh
mesh = Mesh(mesh_name="hexagonal_nanowire.msh", 
           reg2mat={1: "InAs", 2: "GaSb"})

# Set up k-space sampling
kz_values = np.linspace(0, 0.05, 20) * np.pi / np.sqrt(3) / 6.0583

# Calculate band structure
bs = BandStructure(
    mesh=mesh,
    kzvals=kz_values,
    valence_band_edges={"InAs": 0.0, "GaSb": 0.56},
    temperature=4.0,
    number_eigenvalues=20
)
bs.run()

# Visualize results
bs.plot_bands()
```
-->

## Core Components

### 1. Mesh Generation
- **Purpose**: Create optimized finite element grids for nanowire cross-sections
- **Geometry**: Hexagonal symmetry with core-shell regions
- **Output**: FreeFem++ compatible mesh files with material region definitions

### 2. Band Structure Solver  
- **Method**: 8-band k·p Hamiltonian with FEM spatial discretization
- **Features**: MPI parallelization over k-points, spurious solution suppression
- **Analysis**: Carrier character classification, regional charge distribution

### 3. Self-Consistent Solver
- **Coupling**: Schrödinger-Poisson equations with iterative solution
- **Convergence**: Broyden mixing for accelerated convergence
- **Applications**: Multi-parameter sweeps, external field effects

## Documentation (to be revised)

- **[Quick Start Guide](docs/QUICKSTART.md)**: Get up and running in minutes
- **[Scripts](docs/SCRIPTS/)**: Use available ready-to-use scripts
- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup including FreeFem++ integration  
- **[Tutorial Collection](docs/TUTORIALS/)**: Step-by-step examples from basic to advanced (to be done)
- **[API Reference](docs/API_REFERENCE.md)**: Complete class and method documentation
- **[Physics Background](docs/PHYSICS_BACKGROUND.md)**: Theoretical foundations and numerical methods

## Example Applications

- **External Electric Fields**: Band structure engineering and carrier localization control by doping  
- **Temperature-Dependent Properties**: Thermal effects on electronic structure
- **Multi-Material Systems**: Complex heterostructure with multiple interfaces
- **InAs/GaSb Core-Shell Nanowires**: Broken-gap heterostructures for topological applications

## Requirements

- **Python**: 3.7+
- **Core Dependencies**: NumPy, SciPy, matplotlib
- **MPI Support**: mpi4py for parallel calculations
- **External Tools**: FreeFem++ for advanced mesh generation

## Contributing

We welcome contributions to test/extend the library and the scripts. Get in touch with one of the authors to collaborate.

<!-- Please see our [Contributing Guidelines](docs/DEVELOPMENT/contributing.md) for:

- Code style and standards
- Testing procedures  
- Documentation requirements
- Issue reporting and feature requests
-->

## Citation

If you use nwkp<sub>y</sub> in your research, please cite:

```bibtex
@software{nwkp<sub>y</sub>2025,
  title={nwkp<sub>y</sub>: 8-band kÂ·p calculations for semiconductor nanowires},
  author={Vezzosi, Andrea and Goldoni, Guido and Bertoni, Andrea},
  year={2025},
  url={https://github.com/andrea-vezzosi/nwkp<sub>y</sub>}
}

@article{Vezzosi2022,
title = {{Phys. Rev. B 105, 245303 (2022) Band structure of n - and p -doped core-shell nanowires}},
author = {Vezzosi, Andrea and Bertoni, Andrea and Goldoni, Guido},
pages = {245303},
volume = {105},
year = {2022},
doi = {10.1103/PhysRevB.105.245303}
}
```

## License

**nwkp<sub>y</sub>** is released under the MIT License. See [LICENSE](LICENSE) for details.

<!-- 
## Support

- **Documentation**: [https://nwkp<sub>y</sub>.readthedocs.io](https://nwkp<sub>y</sub>.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/nwkp<sub>y</sub>/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/nwkp<sub>y</sub>/discussions)
-->

## Developed by  
*A. Vezzosi (EPFL, Lausanne), G. Goldoni (UNIMORE, Modena), A. Bertoni (CNR-NANO, Modena)*