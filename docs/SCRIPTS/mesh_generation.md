# Mesh Generation Script Documentation

The mesh generation script creates finite element meshes for hexagonal core-shell nanowire structures using FreeFem++ integration. This is the first step in any nwkpy calculation workflow.

## Overview

**Purpose:** Generate optimized finite element grids for nanowire cross-sections  
**Output:** `.msh` (FreeFem++ mesh) and `.dat` (metadata) files  
**Geometry:** Hexagonal symmetry with core-shell material regions  
**Method:** Uses FreeFem++ through nwkpy's `Hex2regsymm` function  

## Quick Usage

```bash
cd scripts/mesh_generation/
python main.py
```

**Default output:** Creates `./outdata/mesh.msh` and `./outdata/mesh.dat`

## File Structure

```
scripts/mesh_generation/
├── main.py          # Main execution script
├── indata.py        # Input parameters (EDIT THIS)
├── outdata/         # Output directory (created automatically)
│   ├── mesh.msh     # FreeFem++ mesh file
│   ├── mesh.dat     # Metadata file
│   └── mesh_generation.log  # Execution log
```

## Input Parameters (`indata.py`)

All mesh parameters are configured in `indata.py`. Here's a complete breakdown:

### File Configuration

```python
directory_name = './outdata'    # Output directory
mesh_name = "mesh"              # Base filename (adds .msh/.dat)
```

**Tips:**
- Use relative paths for portability
- Directory created automatically if doesn't exist
- Avoid spaces in filenames

### Material Configuration

```python
material = ["InAs", "GaSb"]     # [core_material, shell_material]
carrier = ["electron", "hole"]  # [core_carrier, shell_carrier]
```

**Available materials:** InAs, GaAs, GaSb, InP, AlAs, AlGaAs, InGaAs, AlInAs, etc.

**Carrier types:**
- `"electron"` - n-type doping, electron-dominated
- `"hole"` - p-type doping, hole-dominated

**Common material combinations:**
```python
# Type II (staggered) alignment
material = ["InAs", "GaSb"]     # Broken-gap heterostructure
material = ["GaAs", "AlGaAs"]   # Standard quantum well

# Type I (nested) alignment  
material = ["InGaAs", "InP"]    # Telecom applications
material = ["GaAs", "AlAs"]     # High contrast
```

### Geometric Parameters

```python
width = [7.5, 4.88]  # [core_radius, shell_thickness] in nanometers
```

**Physical interpretation:**
- `width[0]` = Core radius (nm)
- `width[1]` = Shell thickness (nm)  
- Total radius = `width[0] + width[1]`
- Total diameter = `2 * (width[0] + width[1])`

**Typical ranges:**
- **Core radius:** 2-20 nm (quantum confinement regime)
- **Shell thickness:** 1-10 nm (sufficient for interface effects)
- **Total diameter:** 10-50 nm (experimental nanowire range)

**Examples:**
```python
width = [5.0, 2.0]    # Small nanowire (14 nm diameter)
width = [10.0, 5.0]   # Medium nanowire (30 nm diameter)  
width = [15.0, 10.0]  # Large nanowire (50 nm diameter)
```

### Mesh Discretization Parameters

```python
edges = [10, 5, 7, 6, 5, 5]  # Mesh density on each hexagonal border
```

**Understanding the hexagonal mesh:**

The mesh uses 1/12 symmetry of a hexagon (30° wedge). The `edges` parameter controls mesh density on each of the 6 borders of this wedge:

```
        4|    3
        /|   /
       / |  /
      /  | /
   5 /---+---\ 2
    /    |    \
   /     |     \
(0,0)----1------
```

**Border mapping:**
- `edges[0]` = Border 1 (bottom horizontal)
- `edges[1]` = Border 2 (bottom-right diagonal)  
- `edges[2]` = Border 3 (top-right diagonal)
- `edges[3]` = Border 4 (top vertical)
- `edges[4]` = Border 5 (left diagonal)
- `edges[5]` = Border 6 (core-shell interface)

**Mesh density guidelines:**
```python
# Coarse mesh (fast, lower accuracy)
edges = [5, 3, 4, 3, 3, 3]

# Standard mesh (good balance)
edges = [10, 5, 7, 6, 5, 5]

# Fine mesh (slow, high accuracy)
edges = [20, 10, 14, 12, 10, 10]

# Ultra-fine mesh (very slow, research quality)
edges = [40, 20, 28, 24, 20, 20]
```

**Border-specific optimization:**
```python
# Emphasize core-shell interface (border 6)
edges = [10, 5, 7, 6, 5, 15]

# Uniform density
edges = [8, 8, 8, 8, 8, 8]

# Emphasize boundaries (borders 1-5)
edges = [15, 10, 12, 10, 8, 5]
```

## Script Execution Flow

### 1. Initialization and Validation

```python
# Header display and library info
print_header('Hexagonal Core-Shell Nanowire Mesh Generation')
library_header()

# Input parameter consistency checks
consistency_checks()
```

**Validation checks:**
- Material names exist in nwkpy database
- Carrier types are valid (`"electron"` or `"hole"`)
- Dimensions are positive
- Edge counts are positive integers
- Exactly 2 materials and carriers specified

### 2. System Configuration

```python
# Create material mappings
reg2mat = {1: material[0], 2: material[1]}  # Region → Material
mat2partic = {material[0]: carrier[0], material[1]: carrier[1]}  # Material → Carrier

# Calculate dimensions
R_c = width[0]                    # Core radius
shell_width = width[1]            # Shell thickness  
total_width = (R_c + shell_width) * 2  # Total diameter
```

### 3. Mesh Generation

```python
# Call FreeFem++ through nwkpy
Hex2regsymm(
    mesh_name=mesh_file,
    total_width=total_width,
    shell_width=shell_width, 
    edges_per_border=edges_per_border
)
```

**What happens internally:**
1. FreeFem++ script is generated automatically
2. Hexagonal geometry is constructed with proper symmetry
3. Core and shell regions are defined
4. Mesh is refined according to `edges` parameters
5. Boundary conditions are applied
6. Output files are written

### 4. Mesh Verification

```python
# Load and verify the generated mesh
mesh = Mesh(
    mesh_name=mesh_file,
    reg2mat=reg2mat,
    mat2partic=mat2partic
)

# Quality checks
total_elements = mesh.nelem
boundary_edges = len(mesh.e_l)
regions = mesh.region_labels
```

## Output Files

### `mesh.msh` - FreeFem++ Mesh File

Binary/text file containing:
- **Vertex coordinates** (x, y) in nm
- **Element connectivity** (triangular elements)
- **Boundary edge definitions**
- **Region labels** (1=core, 2=shell)

**File size:** Typically 100 KB - 10 MB depending on mesh density

### `mesh.dat` - Metadata File

Human-readable text file containing:
```
Materials : InAs GaSb
Carriers  : electron hole  
Width     : 7.5 4.88
Edges     : 10 5 7 6 5 5
```

**Purpose:** Parameter tracking and verification for later calculations

### `mesh_generation.log` - Execution Log

Detailed execution log containing:
- Input parameter validation
- System configuration summary  
- Mesh generation progress
- Quality verification results
- Timing information

## Mesh Quality Analysis

The script automatically performs quality checks:

### Element Count Analysis
```
Total vertices (nodes): 1247
Total elements: 2388
Boundary edges: 84
```

### Regional Analysis
```
Region 1 (InAs) elements: 856
Region 2 (GaSb) elements: 1532
```

### Boundary Analysis
```
Border 1 edges: 10
Border 2 edges: 5  
Border 3 edges: 7
Border 4 edges: 6
Border 5 edges: 5
Border 6 edges: 5
```

## Common Use Cases

### Example 1: Standard InAs/GaSb Nanowire

```python
# indata.py configuration
material = ["InAs", "GaSb"]
carrier = ["electron", "hole"]
width = [7.5, 4.88]           # Typical experimental dimensions
edges = [10, 5, 7, 6, 5, 5]   # Balanced mesh
```

**Expected output:** ~2000-3000 elements, good for standard calculations

### Example 2: High-Resolution Mesh

```python
# For publication-quality results
edges = [20, 10, 14, 12, 10, 10]
```

**Expected output:** ~8000-12000 elements, slower but more accurate

### Example 3: Core-Shell Interface Focus

```python
# Emphasize interface physics
edges = [8, 5, 6, 5, 4, 15]  # High density at interface (border 6)
```

### Example 4: Large Nanowire

```python
width = [15.0, 8.0]          # 46 nm total diameter
edges = [15, 8, 12, 10, 8, 8] # Increased density for larger structure
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "FreeFem++ not found"

**Error message:**
```
FileNotFoundError: FreeFem++ executable not found
```

**Solutions:**
1. Install FreeFem++: See [Installation Guide](../INSTALLATION.md)
2. Add FreeFem++ to system PATH
3. Check installation: `FreeFem++ -h`

#### Issue 2: "Invalid material"

**Error message:**
```
ValueError: Invalid material - Choose available materials
```

**Solutions:**
```python
# Check available materials
from nwkpy._database import params
print(list(params.keys()))

# Fix material names (case-sensitive)
material = ["InAs", "GaSb"]  # Correct
material = ["inas", "gasb"]  # Wrong - case sensitive
```

#### Issue 3: Mesh generation fails

**Error message:**
```
Error in FreeFem++ mesh generation
```

**Solutions:**
1. **Check dimensions:**
   ```python
   width = [7.5, 4.88]  # Good
   width = [0.1, 0.01]  # Too small - numerical issues
   width = [100, 50]    # Too large - memory issues
   ```

2. **Reduce mesh density:**
   ```python
   edges = [5, 3, 4, 3, 3, 3]  # Start with coarse mesh
   ```

3. **Check output directory permissions:**
   ```bash
   mkdir -p outdata
   chmod 755 outdata
   ```

#### Issue 4: Memory errors

**Error message:**
```
MemoryError: Unable to create mesh
```

**Solutions:**
1. **Reduce mesh density:**
   ```python
   edges = [5, 3, 4, 3, 3, 3]  # Much coarser
   ```

2. **Smaller geometry:**
   ```python
   width = [5.0, 3.0]  # Smaller nanowire
   ```

3. **Increase system RAM or use smaller test cases**

#### Issue 5: Inconsistent mesh quality

**Symptoms:** Irregular elements, poor aspect ratios

**Solutions:**
1. **Balance edge densities:**
   ```python
   edges = [10, 10, 10, 10, 10, 10]  # Uniform
   ```

2. **Avoid extreme ratios:**
   ```python
   edges = [50, 2, 5, 3, 4, 2]   # Bad - large variations
   edges = [12, 8, 10, 8, 6, 8]  # Good - gradual variation
   ```

## Performance Guidelines

### Mesh Size vs Computation Time

| Total Elements | Generation Time | Memory Usage | Calculation Time |
|---------------|----------------|--------------|------------------|
| ~1000         | <1 second      | <100 MB      | Minutes          |
| ~5000         | ~5 seconds     | ~500 MB      | ~1 hour          |
| ~20000        | ~30 seconds    | ~2 GB        | ~4 hours         |
| ~50000        | ~2 minutes     | ~8 GB        | ~12 hours        |

### Recommended Settings by Use Case

#### Quick Testing
```python
width = [5.0, 3.0]
edges = [5, 3, 4, 3, 3, 3]
# ~500 elements, fast generation and calculation
```

#### Standard Research
```python  
width = [7.5, 4.88]
edges = [10, 5, 7, 6, 5, 5]
# ~2500 elements, good balance of speed and accuracy
```

#### Publication Quality
```python
width = [7.5, 4.88] 
edges = [15, 10, 12, 10, 8, 10]
# ~6000 elements, high accuracy, longer computation
```

#### Convergence Studies
```python
# Start coarse, increase systematically
edges_coarse = [5, 3, 4, 3, 3, 3]
edges_medium = [10, 5, 7, 6, 5, 5]  
edges_fine = [15, 10, 12, 10, 8, 10]
edges_superfine = [20, 15, 18, 15, 12, 15]
```

## Integration with Band Structure Calculations

### File Transfer

The generated mesh files must be accessible to band structure scripts:

```bash
# Option 1: Copy files to band structure directory
cp outdata/mesh.msh ../band_structure/
cp outdata/mesh.dat ../band_structure/

# Option 2: Update mesh_name in band structure indata.py
mesh_name = "../mesh_generation/outdata/mesh"
```

### Parameter Consistency

Ensure consistent parameters between mesh generation and band structure:

```python
# mesh_generation/indata.py
material = ["InAs", "GaSb"]
carrier = ["electron", "hole"]

# band_structure/indata.py  
material = ["InAs", "GaSb"]     # Must match
carrier = ["electron", "hole"]  # Must match
```

## Advanced Usage

### Custom Material Systems

```python
# Three-layer heterostructure (requires code modification)
material = ["InAs", "InGaAs", "InP"]  # Core, intermediate, shell

# Graded interfaces (future feature)
material = ["InAs", "InAs0.8Ga0.2As"]  # Compositional grading
```

### Non-Standard Geometries

For non-hexagonal geometries, you'll need to:
1. Create custom FreeFem++ scripts
2. Modify the `Hex2regsymm` function
3. Use FreeFem++ directly with nwkpy mesh loading

### Mesh Refinement Studies

```python
# Systematic mesh refinement
base_edges = [5, 3, 4, 3, 3, 3]
for factor in [1, 1.5, 2, 2.5, 3]:
    edges = [int(e * factor) for e in base_edges]
    # Generate mesh with current edges
    # Compare results for convergence
```

## Related Documentation

- **[Quick Start Guide](../QUICKSTART.md)** - Basic workflow including mesh generation
- **[Installation Guide](../INSTALLATION.md)** - FreeFem++ installation details
- **[Band Structure Script](band_structure.md)** - Next step after mesh generation
- **[Physics Background](../PHYSICS_BACKGROUND.md)** - FEM theory and mesh requirements

## Next Steps

✅ **Mesh generated successfully?**

🎯 **Continue with:**
1. **[Band Structure Calculation](band_structure.md)** - Use your mesh for k·p calculations
2. **[Parameter Studies](../TUTORIALS/01_basic_mesh.md)** - Systematic mesh optimization
3. **[Custom Geometries](../TUTORIALS/04_advanced_topics.md)** - Beyond hexagonal nanowires

📊 **Validation:**
- Check mesh quality metrics
- Verify material assignments
- Compare with experimental dimensions
- Test with simple band structure calculation