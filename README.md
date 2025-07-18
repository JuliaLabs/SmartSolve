
<img src="smartsolve.png" alt="SmartSolve.jl">


SmartSolve.jl is a Julia-based toolbox for AI-guided algorithmic discovery, designed to accelerate computations by generating enhanced algorithmic and architectural selection strategies. Envisioned as a general-purpose tool for scientific computing, current efforts focus on challenges in computational linear algebra. The toolbox addresses the growing complexity of selecting efficient solvers, data formats, precision strategies, and hardware resources for structurally diverse matrices—where conventional approaches offer substantial opportunities for improvement. SmartSolve.jl constructs a performance database through systematic benchmarking and applies automated Pareto analysis to identify optimal trade-offs between accuracy and speed. This database underpins a data-driven model that synthesizes dispatch strategies tailored to high-performance linear algebra software.

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

## Publications

- Emmanuel Lujan and Alan Edelman. "When Structure is Silent: Opportunities for Algorithmic Dispatch in Linear Algebra," 2025 IEEE High Performance Extreme Computing Conference (HPEC). Submitted.
- Rushil Shah, Emmanuel Lujan, and Rabab Alomairy, and Alan Edelman. "Data-Driven Dynamic Algorithm Dispatch with Large Language Models," 2025 IEEE High Performance Extreme Computing Conference (HPEC). Submitted.
- Emmanuel Lujan, Rushil Shah, Rabab Alomairy, and Alan Edelman. "SmartSolve.jl: AI for Algorithmic Discovery," 0.1.0-alpha. Zenodo [(link)](https://doi.org/10.5281/zenodo.15784217).
- Rabab Alomairy, Felipe Tome, Julian Samaroo, Alan Edelman. _"Dynamic Task Scheduling with Data Dependency Awareness Using Julia"_, 2024 IEEE High Performance Extreme Computing Conference (HPEC) [(link)](https://ieeexplore.ieee.org/document/10938467).

## Talks

- Rushil Shah, Emmanuel Lujan, and Rabab Alomairy. _"Automated Algorithm Selection Discovery via LLMs,"_ JuliaCon 2025, Lightning Talk. [(Link)](https://pretalx.com/juliacon-2025/talk/review/FXWAYZEZ9XEPYPHL3JJNAS7NBACU3GXE). 
- Alan Edelman et al. _"Julia, Portable Numerical Linear Algebra and Beyond."_ Presentation at Householder Symposium, 2025. [(Link)](https://householder-symposium.github.io/presenters.html). Accessed June 20, 2025.
- Alan Edelman, _"Improving the HPC Experience: Did Julia Get It Right, or Will AI Hide the Problem (or Both)?"_ Keynote at the Workshop on Asynchronous Many-Task Systems and Applications (WAMTA), 2025. [(Link)](https://wamta25.github.io/keynote). Accessed June 20, 2025.
  
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




