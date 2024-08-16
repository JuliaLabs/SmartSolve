# SmartSolve. Proof of concept.

Getting started

```bash
cd SmartSolve
julia --project=.
pkg> instantiate
```

Use SmartSolve to automatically generate a smart version of a linear algebra algorithm based on enhanced algorithm choices. Case study: SmartLU.

```julia
julia> include("generate_smartlu.jl")
```
