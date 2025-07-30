# Chern-Simons Vortex Scattering
 We provide a code for simulating the scattering between a Chern-Simons Vortex and an impurity
 
 We integrate the equations of motion governing the Abelian Chern-Simons-Higgs model in the presence of magnetic impurity. The equations of motion can be found in the article JHEP 12 (2024) 108, entitled "Abelian Chern-Simons vortices in the presence of magnetic impurities". If you use this code in a publication, please cite the article mentioned above.
 
 First, the static BPS equations are solved using the SciPy solve_bvp method. Then the solutions are boosted using Lorentz symmetry to be used as initial condition for a vortex scattering.
 
 The solutions is time evolved utilizing the SciPy solve_ivp method. A third-order runge-kutta method with error control and adaptive timestep was chosen. The spatial derivatives were obtained used a five-point stencil approximation.
 
 The boundary conditions are Neumann's for both vector and scalar fields. However we consider the covariant derivative to be zero at the boundary for the scalar field.
 
 Vortex position, charge conservation and Energy conservation are evaluated during the whole evolution.
 
 Rooms for improvement: 
 
 1. We now believe that periodic boundary conditions would be superior to Neumann's. It is easier to be implemented and more stable.
 
 2. We haven't find an absolutely convergent time integrator for our equations of motion yet.
 
 The software was developed with aid of the supercomputer SDumont from the Brazilian institution LNCC (Laboratório Nacional de Computação Científica).
