# Halide-Perovskite-Simulation-Via-DDR-Method
Simulates halide segregation in perovskite materials using the drift-diffusion-recombination method.

Steps:
- User inputs experimental potential function into simulation (define var datpath)
- Calculates electric field in x and y direction
- Solves for n and p sequentially using Drift-Diffusion-Recombination method
- Repeats previous part for # of desired timesteps
- Outputs time-series data with (x,y,n) & (x,y,p) distributions
- Visualizes potential, electric field, & carrier distributions at final step
