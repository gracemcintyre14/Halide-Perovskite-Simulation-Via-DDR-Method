
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import RectBivariateSpline
#Adding actual data (load), cutting size: PL map, bandgap map, Nx and Ny from bandgapmap.shape, Nxlen, Nylen by takinng those values and multiplying by real distance (objective lens on measurement)

#PHYSICAL PARAMETERS (from paper supplemental info)
k2 = 8.1e-11     # bimolecular recombination [cm³/s]
tau_n = 500e-9    # electron SRH lifetime [s]
tau_p = 500e-9    # hole SRH lifetime [s]
n_i = 1e6        # intrinsic carrier density [cm⁻³]
mu_n = 10       # electron mobility [cm²/V/s] 
mu_p = 10      # hole mobility [cm²/V/s]
G = 1e17  #photons/cm³/s 

#SIMULATION PARAMETERS
T_final = 1e-6   # final time (seconds)
dt_value = 1e-8  #time step (seconds)
datpath = #INPUT DATA HERE
cm_per_pix =  (0.3225e-6)*1e2 
imin, imax, jmin, jmax = (0, 200, 0, 200)
#################
data = np.load(datpath)[imin:imax, jmin:jmax]
# Diffusion coefficients (Einstein relation: D = μ * kT/q, kT/q ≈ 0.0259 V at 300K)
D_n = mu_n * 0.0259  
D_p = mu_p * 0.0259  
# CREATE MESH
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (data.shape[0]*cm_per_pix, data.shape[1]*cm_per_pix)),  # [cm]
    n=(data.shape[0], data.shape[1]),
    cell_type=mesh.CellType.triangle,
)
# FUNCTION SPACES
V_n = fem.functionspace(msh, ("Lagrange", 1))  #electrons
V_p = fem.functionspace(msh, ("Lagrange", 1))  #holes
V_field = fem.functionspace(msh, ("Lagrange", 1))  #for E-field components
# ELECTRIC FIELD FROM EXTERNAL POTENTIAL
# # Define potential function
potential = fem.Function(V_field, name="potential")
xbase = np.linspace(0, data.shape[1]*cm_per_pix, data.shape[1])
ybase = np.linspace(0, data.shape[0]*cm_per_pix, data.shape[0])
spline = RectBivariateSpline(xbase, ybase, data) #interpolating at points (data we are inputing is nxn array, but this mesh is triangular cells, so we need to interpolate real data onto eal mesh)
def load_external_potential(x):
    x_coords = x[0, :]   # shape (Npoints,)
    y_coords = x[1, :]   # shape (Npoints,)
    # evaluate spline at pairs (use .ev for pairwise evaluation)
    vals = spline.ev(x_coords, y_coords)   # returns 1-D array length Npoints
    # ensure dtype float64 (dolfinx expects float64)
    return vals.astype(np.float64)
def test_potential(x):
    x0_min, x0_max = 0.0, data.shape[1]*cm_per_pix  #domain bounds in x [cm]
    y0_min, y0_max = 0.0, data.shape[0]*cm_per_pix  #domain bounds in y [cm]
    
    phi_min = 1.5  #potential at (0, 0)
    phi_max = 2.0  #potential at (1mm, 1mm)
    
    # Normalized coordinates [0, 1]
    x_norm = (x[0] - x0_min) / (x0_max - x0_min)
    y_norm = (x[1] - y0_min) / (y0_max - y0_min)
    
    # Linear interpolation: φ(x,y) = φ_min + (φ_max - φ_min) * (x_norm + y_norm) / 2
    phi = phi_min + (phi_max - phi_min) * (x_norm + y_norm) / 2.0
    return phi

#Interpolate potential onto mesh
potential.interpolate(load_external_potential)
# INITIAL CONDITIONS
n_old = fem.Function(V_n, name="n_old") 
p_old = fem.Function(V_p, name="p_old") 
def initial_condition_n(x):
    """Gaussian excitation in center"""
    return n_i + 0*1e14 * np.exp(-((x[0] - data.shape[1]*cm_per_pix/2)**2 + (x[1] - data.shape[1]*cm_per_pix/2)**2) / (2.0e-4**2))
def initial_condition_p(x):
    """Same for holes (charge neutrality initially)"""
    return n_i + 0*1e14 * np.exp(-((x[0] - data.shape[1]*cm_per_pix/2)**2 + (x[1] - data.shape[1]*cm_per_pix/2)**2) / (2.0e-4**2))
n_old.interpolate(initial_condition_n)
p_old.interpolate(initial_condition_p)
# TIME STEPPING LOOP
t = 0.0
step = 0
while t < T_final:
    t += dt_value
    step += 1
    
    # Get current carrier densities

    n_array = n_old.x.array
    p_array = p_old.x.array
    
    # Compute recombination rates
    SRH_term = (n_array * p_array) / (tau_p * p_array + tau_n * n_array + 1e-20)
    bimol_term = k2 * n_array * p_array
    recomb_n = SRH_term + bimol_term
    recomb_p = SRH_term + bimol_term
    
    recomb_func_n = fem.Function(V_n)
    recomb_func_p = fem.Function(V_p)
    recomb_func_n.x.array[:] = recomb_n
    recomb_func_p.x.array[:] = recomb_p
    
    # SOLVE ELECTRON EQUATION (NO DRIFT - paper assumption)
    # ∂n/∂t = D_n ∇²n + G - R(n,p)
    # Note: No drift term for electrons (paper assumes bandgap modulation 
    # primarily affects valence band)
    
    n_trial = ufl.TrialFunction(V_n)
    v_n = ufl.TestFunction(V_n)
    
    a_n = (n_trial/dt_value) * v_n * ufl.dx
    a_n += D_n * ufl.inner(ufl.grad(n_trial), ufl.grad(v_n)) * ufl.dx
    
    L_n = (n_old/dt_value + G - recomb_func_n) * v_n * ufl.dx
    
    problem_n = LinearProblem(a_n, L_n,  petsc_options_prefix='n',
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    n_new = problem_n.solve()
    
    # Update recombination with new electron density
    n_array_new = n_new.x.array
    SRH_term_new = (n_array_new * p_array) / (tau_p * p_array + tau_n * n_array_new + 1e-20)
    bimol_term_new = k2 * n_array_new * p_array
    recomb_p_new = SRH_term_new + bimol_term_new
    recomb_func_p.x.array[:] = recomb_p_new
    
    # SOLVE HOLE EQUATION (WITH DRIFT FROM POTENTIAL)
    # ∂p/∂t = D_p ∇²p + μ_p ∇·[p E⃗] + G - R(n,p)
    # where E⃗ = -∇φ (electric field from potential)
    
    # Weak form: Note the drift term uses the GRADIENT of potential directly
    # μ_p ∇·[p E⃗] = μ_p ∇·[p (-∇φ)] = -μ_p ∇·[p ∇φ]
    
    p_trial = ufl.TrialFunction(V_p)
    v_p = ufl.TestFunction(V_p)
    
    a_p = (p_trial/dt_value) * v_p * ufl.dx
    a_p += D_p * ufl.inner(ufl.grad(p_trial), ufl.grad(v_p)) * ufl.dx
    
    # CRITICAL: Drift term using potential gradient
    # Integration by parts: -∫ μ_p ∇·[p ∇φ] v dx = ∫ μ_p p ∇φ · ∇v dx
    # Since E⃗ = -∇φ, this becomes: -∫ μ_p p E⃗ · ∇v dx
    a_p += mu_p * p_trial * ufl.inner(ufl.grad(potential), ufl.grad(v_p)) * ufl.dx
    L_p = (p_old/dt_value + G - recomb_func_p) * v_p * ufl.dx
    
    problem_p = LinearProblem(a_p, L_p, petsc_options_prefix='p',
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p_new = problem_p.solve()
    
    # Prevent negative concentrations
    n_new.x.array[:] = np.maximum(n_new.x.array, n_i * 0.01)
    p_new.x.array[:] = np.maximum(p_new.x.array, n_i * 0.01)
    
    # Update for next step
    n_old.x.array[:] = n_new.x.array[:]
    p_old.x.array[:] = p_new.x.array[:]
    
    # # Save data
    # if step % 10 == 0:
    #     coords = msh.geometry.x
    #     electron_data = np.column_stack([coords[:, 0], coords[:, 1], n_new.x.array])
    #     hole_data = np.column_stack([coords[:, 0], coords[:, 1], p_new.x.array])
        
        # np.savetxt(f"{wsl_output_dir}/electrons_{step:04d}.csv", electron_data, 
        #            delimiter=',', header='x,y,electron_density', comments='')
        # np.savetxt(f"{wsl_output_dir}/holes_{step:04d}.csv", hole_data, 
        #            delimiter=',', header='x,y,hole_density', comments='')
print("\nSimulation completed!")
# VISUALIZATION
coords = msh.geometry.x
x_um = coords[:, 0] * 1e4
y_um = coords[:, 1] * 1e4
triangulation = tri.Triangulation(x_um, y_um)
# Plot 1: Potential and E-field
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Potential landscape
contour_phi = axes[0].tricontourf(triangulation, potential.x.array, levels=20, cmap='RdYlBu_r')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('y (μm)')
axes[0].set_title('Potential φ (Bandgap Landscape)')
axes[0].set_aspect('equal')
plt.colorbar(contour_phi, ax=axes[0], label='φ (V or eV)')
X,Y = np.meshgrid(xbase, ybase)
test_data = np.flip(test_potential([X,Y]), axis=0)
Ex, Ey = np.gradient(data, ybase, xbase)
min_E = min(np.min(Ex), np.min(Ey))
max_E = max(np.max(Ex), np.max(Ey))
# E-field x-component
contour_Ex = axes[1].imshow(Ex, cmap='RdBu_r', vmin=min_E, vmax=max_E)
axes[1].set_xlabel('x (μm)')
axes[1].set_title('E-field X: Ex = -∂φ/∂x')
axes[1].set_aspect('equal')
plt.colorbar(contour_Ex, ax=axes[1], label='Ex (V/cm)')
# E-field y-component
contour_Ey = axes[2].imshow(Ey, cmap='RdBu_r', vmin=min_E, vmax=max_E)
axes[2].set_xlabel('x (μm)')
axes[2].set_title('E-field Y: Ey = -∂φ/∂y')
axes[2].set_aspect('equal')
plt.colorbar(contour_Ey, ax=axes[2], label='Ey (V/cm)')
plt.tight_layout()
# plt.savefig(f'{wsl_output_dir}/potential_and_efield.png', dpi=150)
# print("Potential and E-field plot saved")
plt.show()
# Plot 2: Final carrier distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
contour_n = axes[0].tricontourf(triangulation, n_old.x.array, levels=20, cmap='viridis')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('y (μm)')
axes[0].set_title(f'Electron Density at t={T_final*1e6:.1f} μs')
axes[0].set_aspect('equal')
plt.colorbar(contour_n, ax=axes[0], label='n (cm⁻³)')
contour_p = axes[1].tricontourf(triangulation, p_old.x.array, levels=20, cmap='plasma')
axes[1].set_xlabel('x (μm)')
axes[1].set_title(f'Hole Density at t={T_final*1e6:.1f} μs')
axes[1].set_aspect('equal')
plt.colorbar(contour_p, ax=axes[1], label='p (cm⁻³)')
plt.tight_layout()
# plt.savefig(f'{wsl_output_dir}/final_carriers.png', dpi=150)
# print("Carrier density plots saved")
plt.show()
