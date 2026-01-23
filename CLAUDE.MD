# CLAUDE.MD - AI Assistant Context for Neuroblox.jl

## Project Overview

**Neuroblox.jl** is a Julia-based computational neuroscience and psychiatry framework designed for brain circuit simulations and modeling. It provides modular computational building blocks ("blox") that can be combined to describe large-scale brain dynamics across scales from spiking neurons to fMRI-derived circuits.

- **Version**: 0.6.0
- **Language**: Julia 1.10+
- **License**: Free for non-commercial/academic use (see NeurobloxEULA)
- **Repository**: https://github.com/Neuroblox/Neuroblox.jl
- **Documentation**: https://neuroblox.github.io/NeurobloxDocsHost/

## Core Architecture

### Key Technologies
- **ModelingToolkit.jl**: Symbolic (stochastic/delay) differential equations
- **SciMLBase**: Scientific machine learning and differential equation solving
- **Graphs.jl/MetaGraphs**: Network structure representation
- **GraphDynamics**: Graph-based dynamical systems
- **Turing.jl**: Probabilistic modeling and parameter fitting
- **StochasticDiffEq**: Stochastic differential equation solvers

### Project Structure

```
Neuroblox.jl/
├── src/
│   ├── Neuroblox.jl              # Main module
│   ├── Neurographs.jl            # Graph representations
│   ├── blox/                     # Computational building blocks
│   │   ├── neuron_models.jl      # Neuron models (HH, IF, QIF, LIF)
│   │   ├── subcortical_blox.jl   # Subcortical structures
│   │   └── blox_utilities.jl     # Utility functions
│   ├── measurementmodels/        # Observation models
│   │   ├── lfp.jl               # Local field potential
│   │   └── fmri.jl              # fMRI BOLD signal
│   ├── control/                  # Control circuit elements
│   ├── datafitting/              # Parameter fitting (spDCM_VL)
│   ├── utilities/                # General utilities
│   │   ├── spectral_tools.jl
│   │   ├── learning_tools.jl
│   │   └── bold_methods.jl
│   ├── gui/                      # Graphical user interface
│   └── GraphDynamicsInterop/     # GraphDynamics integration
├── examples/                     # Example scripts and notebooks
├── test/                         # Test suite
└── docs/                         # Documentation source
```

## Key Concepts

### Blox (Building Blocks)
Modular computational units representing:
- **Individual Neurons**: Hodgkin-Huxley, Integrate-and-Fire (IF), Quadratic IF (QIF), Leaky IF (LIF)
- **Neural Mass Models**: Jansen-Rit, Wilson-Cowan, Lauter-Breakspear, Next Generation, microcanonical circuits
- **Control Circuit Elements**: Biomimetically-constrained control elements
- **Measurement Models**: LFP (local field potential), fMRI BOLD signal

### Neurographs
Graph-based representations of brain circuits where:
- Nodes represent blox (neurons or neural populations)
- Edges represent connections/synapses
- MetaGraphs store metadata for each component

### Simulation Workflow
1. Define individual blox with parameters
2. Connect blox to form a circuit (neurograph)
3. Generate symbolic differential equations
4. Compile to numerical solver
5. Run simulations with parameter fitting to experimental data

## Development Guidelines

### Code Style
- Follow Julia conventions (PascalCase for types, snake_case for functions)
- Use descriptive variable names for clarity
- Document public functions with docstrings
- Type annotations for performance-critical code

### Testing
- Tests located in `test/` directory
- Use `@testset` for organization
- Run tests with: `julia --project=. -e "using Pkg; Pkg.test()"`

### Dependencies
Key dependencies (see Project.toml for full list):
- ModelingToolkit.jl (v9.55-9.60)
- GraphDynamics (v0.2.5+)
- StochasticDiffEq
- Distributions
- DataFrames
- ForwardDiff (automatic differentiation)

### Performance Considerations
- Leverages symbolic compilation for 100x+ speedup vs Python implementations
- Uses sparse arrays for large-scale networks
- Supports parallel computation via OhMyThreads
- Optimized differential equation solvers from SciML ecosystem

## Common Operations

### Adding New Blox
1. Define structure in `src/blox/`
2. Implement symbolic equations using ModelingToolkit
3. Add connection interfaces
4. Write tests in `test/`
5. Update documentation

### Extending Measurement Models
- Implement in `src/measurementmodels/`
- Follow existing LFP/fMRI patterns
- Connect to blox outputs

### Parameter Fitting
- Use spDCM (spectral Dynamic Causal Modeling) framework
- Turing.jl for Bayesian inference
- Support for HMC sampling and ADVI

## Installation for Development

```julia
using Pkg
pkg"registry add https://github.com/Neuroblox/NeurobloxRegistry"
pkg"registry add General"
pkg"add Neuroblox"
```

For development:
```julia
pkg"dev /path/to/Neuroblox.jl"
```

## Important Notes

### License Restrictions
- Free for non-commercial and academic use
- Commercial use requires license (contact info@neuroblox.ai)
- See NeurobloxEULA for full terms

### Citation
If using in research, cite:
- DOI: 10.5281/zenodo.14192732
- Repository: https://github.com/Neuroblox/Neuroblox.jl

### Known Constraints
- Julia 1.10+ required
- ModelingToolkit compatibility locked to v9.55-9.60
- SymbolicUtils v3.11.0 required for stability

## Contact & Resources

- **Issues**: GitHub issue tracker
- **Commercial inquiries**: info@neuroblox.ai
- **Documentation**: https://neuroblox.github.io/NeurobloxDocsHost/
- **Examples**: See `examples/` directory for usage patterns

## AI Assistant Guidelines

When working with this codebase:
1. **Respect symbolic computation**: ModelingToolkit generates code; don't optimize prematurely
2. **Maintain modularity**: Blox should be composable and independent
3. **Preserve performance**: This library emphasizes speed; profile before adding complexity
4. **Follow neuroscience conventions**: Use standard terminology from computational neuroscience
5. **Test thoroughly**: Brain simulations are complex; validate against known results
6. **Document changes**: Neural models require clear explanation of assumptions

## Recent Changes (v0.6.0)
Check git history for recent updates to LSM.jl and core functionality.
