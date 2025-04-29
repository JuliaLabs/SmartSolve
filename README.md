# SmartSolve

SmartSolve aims to significantly accelerate various linear algebra algorithms based on providing better algorithmic and architectural choices. 


## How to start

In the following example SmartSolve is used to automatically generate SmartLU, an optimized version of the LU decomposition.

```bash
cd SmartSolve/examples/smartlu
julia --project=.
```

```julia
pkg> dev ../..
pkg> instantiate
julia> include("generate_smartlu.jl")
```

## References

Rushil Shah, Emmanuel Lujan, and Rabab Alomairy. _"Automated Algorithm Selection Discovery via LLMs,"_ JuliaCon 2025, Lightning Talk [(link)](https://pretalx.com/juliacon-2025/talk/review/FXWAYZEZ9XEPYPHL3JJNAS7NBACU3GXE).


## Acknowledgements

We thank [DARPA](https://www.darpa.mil/research/programs/mathematics-for-the-discovery-of-algorithms-and-architectures) for supporting this work at MIT.




