# SmartSolve. Proof of concept.

SmartSolve aims to significantly accelerate various linear algebra algorithms based on providing better algorithmic and architectural choices.

Getting started.

```bash
cd SmartSolve
julia --project=.
pkg> instantiate
```

In the following example, SmartSolve is used to automatically generate an optimized version of the LU decomposition: SmartLU.

```julia
julia> include("generate_smartlu.jl")
```
