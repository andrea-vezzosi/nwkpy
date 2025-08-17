# Physics Background

This document provides the theoretical foundation and implementation details for nwkpy's semiconductor nanowire calculations, focusing on the specific formulation of the 8-band k·p method and its finite element implementation.

## 8-Band k·p Hamiltonian Implementation

### Basis States and Matrix Structure

nwkpy uses the 8-component spinor basis including both conduction and valence bands at the Γ-point:

**Basis ordering:**
```
|S↑⟩, |S↓⟩, |X↑⟩, |Y↑⟩, |Z↑⟩, |X↓⟩, |Y↓⟩, |Z↓⟩
```

The total Hamiltonian is constructed as:
```
HBF = H0 + HSO + Ve(r)δμν
```

where H0 contains the kinetic and band structure terms, HSO includes spin-orbit coupling, and Ve(r) is the external electrostatic potential.

### Band Structure Hamiltonian H0

The band structure component has the block structure:

```
H0 = [ Hc     Hcv   ]
     [ Hvc    Hv    ]
```

**Conduction block Hc (2×2):**
```
Hc = [Ec + ℏ²k²/2me    0                ]
     [0                Ec + ℏ²k²/2me  ]
```

where Ec(r) is the position-dependent conduction band edge and me is the electron effective mass.

**Valence block Hv (6×6):**

The valence Hamiltonian includes the Luttinger-Kohn description with proper operator ordering for heterostructures:

```
Hv = (Ev,av + ℏ²k²/2m0)I3 + HLK
```

where Ev,av = Ev - Δso/3 and HLK contains the Luttinger terms:

```
HLK = [k̂xLk̂x + k̂yMk̂y + k̂zMk̂z    k̂xN⁺k̂y + k̂yN⁻k̂x      k̂xN⁺k̂z + k̂zN⁻k̂x    ]
      [k̂yN⁺k̂x + k̂xN⁻k̂y                k̂xMk̂x + k̂yLk̂y + k̂zMk̂z  k̂yN⁺k̂z + k̂zN⁻k̂y    ]
      [k̂zN⁺k̂x + k̂xN⁻k̂z                k̂zN⁺k̂y + k̂yN⁻k̂z          k̂xMk̂x + k̂yMk̂y + k̂zLk̂z]
```

**Interband coupling Hcv and Hvc:**

The conduction-valence coupling is governed by the Kane parameter P:

```
Hcv = [iP k̂x  iP k̂y  iP k̂z  0       0       0      ]
      [0       0       0       iP k̂x  iP k̂y  iP k̂z ]

Hvc = Hcv†
```

### Spin-Orbit Coupling HSO

The spin-orbit Hamiltonian has the explicit 8×8 form:

```
HSO = (Δso/3) × [
  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  
  0  0  0 -i  0  0  0  1
  0  0  i  0  0  0  0 -i
  0  0  0  0  0 -1  i  0
  0  0  0  0 -1  0  i  0
  0  0  0  0 -i -i  0  0
  0  0  1  i  0  0  0  0
]
```

This matrix couples the p-like valence states and creates the heavy-hole, light-hole, and split-off band structure.

### Parameter Rescaling for Spurious Solution Suppression

nwkpy implements the Foreman rescaling scheme to eliminate unphysical high-energy states. The Kane energy EP is modified according to:

```
EPrsc = Eg(Eg + Δso) / (Eg + (2/3)Δso) × (m0/me)
```

This rescaling affects both the interband matrix elements and the modified Luttinger parameters:

```
γ̃1 = γ1 - EP/(3Eg)
γ̃2 = γ2 - EP/(6Eg)  
γ̃3 = γ3 - EP/(6Eg)
```

## Envelope Function Equations and Operator Ordering

### Burt-Foreman Formulation

For heterostructures with position-dependent parameters, the envelope function equations include proper operator ordering. The Hamiltonian operator takes the form:

```
ĤBF = Σα,β ∂α D̄αβ ∂β + Σα [F̄Lα ∂α + ∂α F̄Rα] + Ḡ
```

where:
- D̄αβ: Second-order differential operator matrices
- F̄Lα, F̄Rα: First-order operator matrices (left and right action)
- Ḡ: Zero-order (potential) matrix

### Matrix Element Construction

For each spatial component α,β ∈ {x,y}, the coefficient matrices are constructed from the k·p parameters:

**Second-order terms (D̄αβ):**
```
D̄xx = -diag[Ac, Ac, L, M, M, L, M, M]
D̄yy = -diag[Ac, Ac, M, L, M, M, L, M]  
D̄xy = D̄yx = -diag[0, 0, N⁺, N⁻, 0, N⁻, N⁺, 0]
```

**First-order terms (F̄Lα, F̄Rα):**

These matrices encode the interband coupling and arise from the k·p perturbation:

```
F̄Lx = F̄Rx = -i[0  0  0  0  0  0  0  0]
                [0  0  0  0  0  0  0  0]
                [P  0  0  0  0  0  0  0]
                [0  0  0  0  0  0  0  0]
                [0  0  0  0  0  0  0  0]
                [0  P  0  0  0  0  0  0]
                [0  0  0  0  0  0  0  0]
                [0  0  0  0  0  0  0  0]
```

### Coordinate System Rotation

For nanowires grown along [hkl] directions, the Hamiltonian must be rotated. The transformation is applied as:

```
D̄αβ → Σα'β' Rαα' D̄α'β' R⁻¹β'β
F̄L/Rα → Σα' F̄L/Rα' R⁻¹α'α
```

where R is the rotation matrix. For [111] growth direction:

```
R = [cos θ cos φ   sin φ    cos θ]
    [-sin φ       cos φ    0    ]  
    [cos φ sin θ  sin φ sin θ  cos θ]
```

with θ = arccos(1/√3) and φ = π/4.

## Finite Element Implementation Details

### Variational Formulation

The envelope function equations are solved using the finite element method through a variational approach. The action integral is:

```
A = Σμν ∫ dr⊥ ψ*μ Lμν ψν
```

where the Lagrangian density is:

```
Lμν = Σαβ (-∂̃α D̄αβμν ∂̃β) + Σα [F̄αL,μν ∂̃α - ∂̃α F̄αR,μν] + Ḡμν - E δμν
```

### Shape Function Implementation

nwkpy supports multiple interpolation schemes:

**Lagrange linear elements (3 nodes per triangle):**
```
N1 = x̃     N2 = ỹ     N3 = 1 - x̃ - ỹ
```

**Lagrange quadratic elements (6 nodes per triangle):**
```
N1 = -x̃ + 2x̃²
N2 = -ỹ + 2ỹ²
N3 = 1 - 3x̃ - 3ỹ + 2x̃² + 4x̃ỹ + 2ỹ²
N4 = 4ỹ - 4x̃ỹ - 4ỹ²
N5 = 4x̃ - 4x̃ỹ - 4x̃²
N6 = 4x̃ỹ
```

**Hermite cubic elements (9 DOF per triangle):**

These include function values and first derivatives at each vertex, providing C¹ continuity and eliminating spurious solutions for electrons.

### Element Matrix Assembly

For each triangular element, the element matrices are computed as:

```
H(iel)μν,ij = ∫Ωel [Σαβ (∂α Ni) D̄αβμν (∂β Nj) 
                   + Σα Ni F̄αL,μν (∂α Nj) 
                   - Σα (∂α Ni) F̄αR,μν Nj 
                   + Ni Ḡμν Nj] dr⊥
```

### Mixed Shape Function Basis

nwkpy uses a mixed formulation to suppress spurious solutions:
- **Electrons (s-like states)**: Lagrange quadratic or linear elements
- **Holes (p-like states)**: Hermite cubic elements with derivative continuity

This combination eliminates ghost bands while maintaining computational efficiency.

## Self-Consistent Schrödinger-Poisson Implementation

### Coupled System Formulation

The self-consistent loop couples:
1. **Electronic structure**: HBF ψ = E ψ with potential Ve(r)
2. **Charge densities**: ρ(r) from occupied envelope functions
3. **Electrostatics**: ∇·[ε(r)∇φ(r)] = -ρ(r)/ε0

### Charge Density Calculation

Free carrier densities are computed by integrating over the Brillouin zone:

```
ne(r⊥) = Σn∈c.s. Σν=1⁸ ∫ dk/(2π) f(En(k), μ, T) |ψνn(r⊥,k)|²
nh(r⊥) = Σn∈v.s. Σν=1⁸ ∫ dk/(2π) [1-f(En(k), μ, T)] |ψνn(r⊥,k)|²
```

The character classification uses spinor analysis:
- **Electron character**: |⟨S↑|ψ⟩|² + |⟨S↓|ψ⟩|²
- **Heavy-hole character**: |⟨X↑|ψ⟩|² + |⟨Y↑|ψ⟩|² + |⟨X↓|ψ⟩|² + |⟨Y↓|ψ⟩|²
- **Light-hole character**: |⟨Z↑|ψ⟩|² + |⟨Z↓|ψ⟩|²

### Broyden Mixing Algorithm

The self-consistent convergence is accelerated using Broyden's method:

```
V(n+1) = V(n) + β[Vout(n) - V(n)] + Σi=0M-1 αi [ΔV(n-i) - βΔF(n-i)]
```

where:
- ΔV(i) = V(i+1) - V(i): potential differences
- ΔF(i) = F(i+1) - F(i): residual differences  
- αi: Broyden coefficients determined by minimizing residual norm
- β: linear mixing parameter
- M: number of stored iterations

### Modified Envelope Function Approximation

For broken-gap heterostructures with electron-hole hybridization, nwkpy implements a modified approach that avoids double-counting in charge density calculations. States are classified based on their dominant character rather than energy position relative to the band gap.

## Material Parameter Implementation

### Database Structure

nwkpy includes material parameters organized by:

**Band structure parameters:**
- Band gaps Eg(T) with temperature dependence
- Electron effective masses me
- Luttinger parameters γ1, γ2, γ3
- Spin-orbit coupling Δso
- Kane parameter P (or equivalently EP)

**Rescaling parameters:**
- Conduction band parameter Ac
- Modified Luttinger parameters for 8-band model
- Spurious solution suppression factors

### Heterostructure Band Alignment

Band offsets are implemented through position-dependent band edges:

```
Ec(r) = Ecref + ΔEc(material(r))
Ev(r) = Evref + ΔEv(material(r))
```

The valence band offsets ΔEv are typically specified as input parameters, while conduction band offsets follow from the band gaps.

### Strain Effects

Hydrostatic strain modifies band gaps through deformation potentials:

```
Egstrained = Egunstrained + ac Tr(ε) + b(εxx - εyy)²
```

where ac and b are deformation potential parameters and ε is the strain tensor.

## Numerical Implementation Considerations

### Matrix Storage and Solution

The finite element discretization leads to large sparse eigenvalue problems. nwkpy uses:

- **Sparse matrix storage**: CSR format for memory efficiency
- **Iterative eigensolvers**: ARPACK for interior eigenvalue problems
- **Shift-invert spectral transformation**: Finding states near Esearch

### Parallel Implementation Strategy

MPI parallelization is implemented over k-points:

```
k_local = k_total[rank::size]  # Distribute k-points across processes
```

Each process:
1. Solves eigenvalue problem for assigned k-points
2. Computes local contributions to charge density
3. Communicates results via `MPI.Allgather` operations

### Convergence Monitoring

Self-consistent convergence is monitored through:

```
Rn = ||ρn(i+1) - ρn(i)|| / ||ρn(i)||
Rp = ||ρp(i+1) - ρp(i)|| / ||ρp(i)||
```

Convergence is achieved when both Rn < tolerance and Rp < tolerance.

### Boundary Conditions Implementation

**Dirichlet conditions** (fixed potential):
```
φ(boundary_nodes) = φspecified
```

**Neumann conditions** (specified normal derivative):
```
n̂ · ∇φ = σspecified / ε
```

Mixed boundary conditions are implemented by modifying the global stiffness matrix appropriately.

## Advanced Features

### External Electric Field Treatment

Uniform external fields Eext are implemented as:

```
Ve(r) = VPoisson(r) - Eext · r
```

The field components are transformed to the device coordinate system and added as a linear potential term.

### Interface Physics

Core-shell interfaces are treated with:
- **Abrupt approximation**: Material parameters change discontinuously
- **Current conservation**: Proper boundary conditions from variational formulation
- **Band offset implementation**: Position-dependent diagonal terms

### Temperature Dependencies

Temperature effects are included through:
- **Band gap variation**: Eg(T) using Varshni or Bose-Einstein fits
- **Fermi-Dirac statistics**: Temperature-dependent occupation
- **Lattice constant thermal expansion**: Affecting strain and band offsets

This implementation provides a comprehensive and accurate treatment of semiconductor nanowire physics while maintaining computational efficiency through careful algorithm design and numerical optimization.

## Theoretical Limitations and Extensions

### Current Approximations

nwkpy makes several approximations that users should understand:

- **Envelope function validity**: Requires smooth variation on atomic scale
- **Hard wall boundaries**: Neglects surface reconstruction and states
- **Parabolic bands**: Local quadratic approximation to true band structure
- **Bulk material parameters**: Interface-specific modifications not included
- **Static fields**: Dynamic and high-frequency effects neglected

### Future Extensions

Potential theoretical enhancements include:
- **Strain field calculations**: Full elastic continuum mechanics
- **Many-body effects**: Exchange-correlation beyond Hartree approximation  
- **Surface state treatment**: Explicit surface Hamiltonians
- **Phonon coupling**: Electron-phonon interaction and deformation potentials
- **Magnetic field effects**: Landau quantization and Zeeman splitting

## Comparison with Other Methods

### Tight-Binding Methods

**k·p advantages:**
- Much faster for large systems
- Smooth parameters across interfaces
- Natural inclusion of strain effects

**k·p limitations:**
- Limited to near-zone-center physics
- Requires experimental parameter input
- Less accurate for highly confined systems

### Density Functional Theory

**k·p advantages:**
- Handles thousands of atoms efficiently  
- Includes electrostatic self-consistency
- Direct experimental parameter connection

**k·p limitations:**
- Cannot predict new materials properties
- Less accurate exchange-correlation treatment
- Requires parameter fitting to experiment

### Empirical Pseudopotential Method

**k·p advantages:**
- Faster convergence with system size
- Natural heterostructure treatment
- Simpler implementation and debugging

**k·p limitations:**
- Less transferable between material systems
- Coarser real-space resolution
- Requires smooth variation assumption

## Conclusion

The theoretical framework implemented in nwkpy represents a mature and validated approach to semiconductor nanowire calculations. The combination of 8-band k·p theory, finite element spatial discretization, and self-consistent electrostatics provides an optimal balance between accuracy and computational efficiency for the length scales and material systems of current experimental interest.

The method's strength lies in its ability to handle complex heterostructures with full inclusion of quantum confinement, interface effects, and electrostatic feedback. While certain approximations limit its applicability to some extreme regimes, the approach captures the essential physics needed for understanding and designing nanowire-based devices and materials.

Users should understand both the capabilities and limitations of the theoretical framework to apply nwkpy effectively to their specific research problems. The extensive validation against experiment and comparison with other theoretical methods provides confidence in the results for the intended application domains.

---

**Further Reading:**
- Burt, M. G. "The justification for applying the effective-mass approximation to microstructures" *J. Phys.: Condens. Matter* **4**, 6651 (1992)
- Foreman, B. A. "Effective-mass Hamiltonian and boundary conditions for the valence bands of semiconductor microstructures" *Phys. Rev. B* **48**, 4964 (1993)
- Vurgaftman, I. & Meyer, J. R. "Band parameters for III–V compound semiconductors and their alloys" *J. Appl. Phys.* **89**, 5815 (2001)