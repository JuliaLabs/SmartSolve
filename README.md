

<img src="smartsolve.png" alt="SmartSolve.jl">



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

- Rushil Shah, Emmanuel Lujan, and Rabab Alomairy. _"Automated Algorithm Selection Discovery via LLMs,"_ JuliaCon 2025, Lightning Talk. [(Link)](https://pretalx.com/juliacon-2025/talk/review/FXWAYZEZ9XEPYPHL3JJNAS7NBACU3GXE). 
- Alan Edelman et al. _"Julia, Portable Numerical Linear Algebra and Beyond."_ Presentation at Householder Symposium, 2025. [(Link)](https://householder-symposium.github.io/presenters.html). Accessed June 20, 2025.
- Alan Edelman, _"Improving the HPC Experience: Did Julia Get It Right, or Will AI Hide the Problem (or Both)?"_ Keynote at the Workshop on Asynchronous Many-Task Systems and Applications (WAMTA), 2025. [(Link)](https://wamta25.github.io/keynote). Accessed June 20, 2025.
- Rabab Alomairy, Felipe Tome, Julian Samaroo, Alan Edelman. _"Dynamic Task Scheduling with Data Dependency Awareness Using Julia"_, 2024 IEEE High Performance Extreme Computing Conference (HPEC) [(link)](https://ieeexplore.ieee.org/document/10938467).
- SmartSolve article (in preparation)

## How to Cite

```bibtex
@software{SmartSolve2025,
  author       = {Lujan, Emmanuel and Shah, Rushil N. and Alomairy, Rabab and Edelman, Alan},
  title        = {SmartSolve.jl: AI for Algorithmic Discovery},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.1.0-alpha},
  doi          = {10.5281/zenodo.15784217},
  url          = {https://doi.org/10.5281/zenodo.15784217},
}
```

## Acknowledgements

We thank [DARPA](https://www.darpa.mil/research/programs/mathematics-for-the-discovery-of-algorithms-and-architectures) for supporting this work at MIT.




