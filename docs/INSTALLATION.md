# Installation Guide

Complete installation instructions for nwkpy on different operating systems. This guide covers all dependencies and common installation issues.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), Windows 10
- **Python**: 3.7 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space

### Recommended for Performance
- **RAM**: 16 GB or more for large calculations
- **CPU**: Multi-core processor (4+ cores)
- **MPI**: For parallel k-point calculations

## Quick Installation (Recommended)

If you just want to get started quickly:

```bash
# 1. Clone repository
git clone https://github.com/your-username/nwkpy.git
cd nwkpy

# 2. Install with pip
pip install -e .

# 3. Install MPI support
pip install mpi4py

# 4. Test installation
python -c "import nwkpy; print('nwkpy installed successfully!')"
```

For advanced features (mesh generation), continue to the detailed installation below.

## Detailed Installation

### Step 1: Python Environment Setup

#### Option A: Using Conda (Recommended)

```bash
# Create new environment
conda create -n nwkpy python=3.9 numpy scipy matplotlib

# Activate environment
conda activate nwkpy

# Install additional scientific packages
conda install -c conda-forge mpi4py spyder jupyter
```

#### Option B: Using pip with venv

```bash
# Create virtual environment
python -m venv nwkpy_env

# Activate environment
# Linux/macOS:
source nwkpy_env/bin/activate
# Windows:
nwkpy_env\Scripts\activate

# Install core dependencies
pip install numpy scipy matplotlib jupyter
```

### Step 2: MPI Installation

MPI is required for parallel k-point calculations and significantly speeds up computations.

#### Linux (Ubuntu/Debian)

```bash
# Install MPI implementation
sudo apt-get update
sudo apt-get install mpich mpich-dev

# Or alternatively OpenMPI
sudo apt-get install openmpi-bin openmpi-dev

# Install Python MPI bindings
pip install mpi4py

# Test MPI installation
mpirun -np 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank} of {MPI.COMM_WORLD.size}')"
```

#### macOS

```bash
# Using Homebrew
brew install mpich

# Or OpenMPI
brew install open-mpi

# Install Python bindings
pip install mpi4py

# Test installation
mpirun -np 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank} of {MPI.COMM_WORLD.size}')"
```

#### Windows

**Option 1: Using conda (easiest)**
```bash
conda install -c conda-forge mpi4py
```

**Option 2: Microsoft MPI**
1. Download and install [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
2. Set environment variables
3. Install mpi4py: `pip install mpi4py`

### Step 3: nwkpy Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nwkpy.git
cd nwkpy

# Install in development mode (recommended)
pip install -e .

# Or install from PyPI (when available)
# pip install nwkpy
```

#### Verify Core Installation

```python
# Test basic functionality
python -c "
import nwkpy
from nwkpy.fem import Mesh
from nwkpy import BandStructure
print('✓ nwkpy core installation successful')
"
```

### Step 4: FreeFem++ Installation (Optional)

FreeFem++ is required for advanced mesh generation and custom geometries.

#### Linux (Ubuntu/Debian)

```bash
# Install FreeFem++
sudo apt-get install freefem++

# Or build from source for latest version
wget https://github.com/FreeFem/FreeFem-sources/releases/download/v4.13/freefem++-4.13.tar.gz
tar -xzf freefem++-4.13.tar.gz
cd freefem++-4.13
./configure
make
sudo make install
```

#### macOS

```bash
# Using Homebrew
brew install freefem

# Or download installer from FreeFem++ website
# https://freefem.org/download.html
```

#### Windows

1. Download installer from [FreeFem++ website](https://freefem.org/download.html)
2. Run installer with administrator privileges
3. Add FreeFem++ to PATH environment variable

#### Test FreeFem++ Integration

```bash
# Test FreeFem++ command line
FreeFem++ -h

# Test nwkpy integration
python -c "
from nwkpy.fem import Hex2regsymm
print('✓ FreeFem++ integration available')
"
```

## Installation Verification

### Basic Functionality Test

Create a test script `test_installation.py`:

```python
#!/usr/bin/env python
"""Test nwkpy installation"""

import sys
import numpy as np

def test_imports():
    """Test all core imports"""
    try:
        import nwkpy
        from nwkpy.fem import Mesh
        from nwkpy import BandStructure, PoissonProblem
        from nwkpy import FreeChargeDensity, ElectrostaticPotential
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_mpi():
    """Test MPI functionality"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        print(f"✓ MPI available - rank {comm.rank} of {comm.size}")
        return True
    except ImportError:
        print("✗ MPI not available (optional for single-core calculations)")
        return False

def test_database():
    """Test material parameter database"""
    try:
        from nwkpy._database import params
        materials = list(params.keys())
        print(f"✓ Material database loaded - {len(materials)} materials available")
        print(f"  Sample materials: {materials[:5]}")
        return True
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False

def test_freefem():
    """Test FreeFem++ integration"""
    try:
        from nwkpy.fem import Hex2regsymm
        print("✓ FreeFem++ integration available")
        return True
    except ImportError:
        print("✗ FreeFem++ not available (optional for mesh generation)")
        return False

if __name__ == "__main__":
    print("nwkpy Installation Verification")
    print("=" * 40)
    
    tests = [
        ("Core imports", test_imports),
        ("MPI support", test_mpi),
        ("Material database", test_database),
        ("FreeFem++ integration", test_freefem)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        results.append(test_func())
    
    print(f"\nSummary: {sum(results)}/{len(tests)} tests passed")
    
    if results[0]:  # Core imports successful
        print("\n✓ nwkpy is ready for basic calculations!")
        if not results[1]:
            print("  Note: Install MPI for parallel calculations")
        if not results[3]:
            print("  Note: Install FreeFem++ for mesh generation")
    else:
        print("\n✗ Installation incomplete - check error messages above")
        sys.exit(1)
```

Run the test:
```bash
python test_installation.py
```

### Performance Test

Test with a small calculation:

```python
# Create minimal test
python -c "
import numpy as np
from nwkpy.fem import Mesh

# Test mesh loading (requires pre-generated mesh)
try:
    # This will fail if no mesh exists - that's expected
    mesh = Mesh('test.msh', reg2mat={1: 'InAs', 2: 'GaSb'})
    print('✓ Mesh functionality working')
except FileNotFoundError:
    print('✓ Mesh class loads correctly (no test mesh file)')
except Exception as e:
    print(f'✗ Mesh error: {e}')
"
```

## Common Installation Issues

### Issue 1: ImportError for nwkpy modules

**Symptoms:**
```
ImportError: No module named 'nwkpy'
```

**Solutions:**
```bash
# Ensure you installed in the correct environment
pip list | grep nwkpy

# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue 2: MPI compilation errors

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 is required
```

**Solutions:**

**Windows:**
```bash
# Use conda instead
conda install -c conda-forge mpi4py

# Or install Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Linux:**
```bash
# Install development headers
sudo apt-get install python3-dev

# Try different MPI implementation
sudo apt-get install libopenmpi-dev
pip install mpi4py
```

### Issue 3: FreeFem++ not found

**Symptoms:**
```
FreeFem++: command not found
```

**Solutions:**
```bash
# Add to PATH (Linux/macOS)
export PATH=$PATH:/usr/local/bin

# Windows - add FreeFem++ installation directory to PATH
# Usually: C:\Program Files\FreeFem++
```

### Issue 4: Memory errors during calculations

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
- Reduce mesh resolution (fewer `edges` in mesh generation)
- Use fewer k-points (`number_k_pts`)
- Reduce number of eigenvalues (`number_eigenvalues`)
- Increase system RAM or use HPC cluster

### Issue 5: Numerical convergence problems

**Symptoms:**
```
Warning: Convergence not achieved
```

**Solutions:**
- Check material parameters are physical
- Verify mesh quality
- Adjust convergence criteria (`maxchargeerror`)
- Try different shape functions

## Performance Optimization

### MPI Configuration

```bash
# Optimal number of processes (usually = number of k-points)
mpirun -np 4 python main.py  # For 4 k-points

# Check MPI performance
mpirun -np 2 python -c "
from mpi4py import MPI
import time
comm = MPI.COMM_WORLD
start = time.time()
comm.Barrier()
print(f'Rank {comm.rank}: MPI latency {time.time()-start:.3f}s')
"
```

### Memory Management

```python
# In your calculations, monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# For large calculations, use sparse matrices
from scipy.sparse import save_npz, load_npz
```

## Development Installation

For contributing to nwkpy:

```bash
# Clone with development tools
git clone --recursive https://github.com/your-username/nwkpy.git
cd nwkpy

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Container Installation (Advanced)

### Docker

```dockerfile
# Dockerfile for nwkpy
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    mpich mpich-dev \
    freefem++ \
    && rm -rf /var/lib/apt/lists/*

COPY . /nwkpy
WORKDIR /nwkpy
RUN pip3 install -e .

CMD ["python3"]
```

### Singularity

```bash
# Build container
singularity build nwkpy.sif nwkpy.def

# Run calculation
singularity exec nwkpy.sif python main.py
```

## Next Steps

✅ **Installation complete?** 

📖 **Continue with:**
- [Quick Start Guide](QUICKSTART.md) - Your first calculation
- [Physics Background](PHYSICS_BACKGROUND.md) - Theory and methods
- [API Reference](API_REFERENCE.md) - Detailed class documentation

🔧 **For developers:**
- [Contributing Guide](DEVELOPMENT/contributing.md) - How to contribute code
- [Testing Guide](DEVELOPMENT/testing.md) - Running and writing tests

💬 **Need help?**
- Open an issue on [GitHub Issues](https://github.com/your-username/nwkpy/issues)
- Check [Troubleshooting section](TROUBLESHOOTING.md)
- Join our [discussions](https://github.com/your-username/nwkpy/discussions)