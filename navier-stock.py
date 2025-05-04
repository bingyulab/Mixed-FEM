#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# FEniCS/DOLFINx imports
import dolfinx
from dolfinx import mesh, fem
import basix.ufl
import ufl
from mpi4py import MPI
import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc

# Optional visualization
try:
    import viskex
except ImportError:
    print("viskex not found, skipping visualization")
    viskex = None


# Create output directory
output_dir = Path("output_inf_sup")
os.makedirs(output_dir, exist_ok=True)

def create_unit_square_mesh(n: int = 32) -> mesh.Mesh:
    """Create a unit square mesh with n×n elements."""
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh

def wall(x: np.ndarray) -> np.ndarray:
    """Determine the position of the wall."""
    return np.logical_or(
        x[1] < 0 + np.finfo(float).eps, x[1] > 1 - np.finfo(float).eps)

def get_function_spaces(case: str, msh: mesh.Mesh):
    """
    Define function spaces for different velocity-pressure element pairings.
    
    Parameters:
        case: String identifier for the pairing (e.g., "P2_x_P1", "RT_x_P0")
        msh: Mesh to define the spaces on
    
    Returns:
        Tuple of (velocity_space, pressure_space)
    """
    if case == "P2_x_P1":
        # Taylor-Hood elements (stable)
        V = fem.functionspace(
            msh, 
            basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
        )
        Q = fem.functionspace(
            msh, 
            basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        )
    elif case == "RT_x_P1":
        # Raviart-Thomas elements (stable)
        V = fem.functionspace(
            msh, 
            basix.ufl.element("RT", msh.basix_cell(), 2)
        )
        Q = fem.functionspace(
            msh, 
            basix.ufl.element("Discontinuous Lagrange", msh.basix_cell(), 1)
        )
    elif case == "BDM_x_P1":
        # Brezzi-Douglas-Marini elements (stable)
        V = fem.functionspace(
            msh, 
            basix.ufl.element("BDM", msh.basix_cell(), 2)
        )
        Q = fem.functionspace(
            msh, 
            basix.ufl.element("Discontinuous Lagrange", msh.basix_cell(), 1)
        )
    else:
        raise ValueError(f"Unknown case: {case}")
    
    print(f"Created function spaces for {case}")
    return V, Q

def normalize(u1: fem.Function, u2: fem.Function, p: fem.Function) -> None:
    """Normalize an eigenvector."""
    scaling_operations = [
        # Scale functions with a W^{1,1} (for velocity) or L^1 (for pressure) norm
        (u1, lambda u: (u.dx(0) + u.dx(1)) * ufl.dx, lambda x: x),
        (u2, lambda u: (u.dx(0) + u.dx(1)) * ufl.dx, lambda x: x),
        (p, lambda p: p * ufl.dx, lambda x: x),
        # Normalize functions with a H^1 (for velocity) or L^2 (for pressure) norm
        (u1, lambda u: ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx, lambda x: np.sqrt(x)),
        (u2, lambda u: ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx, lambda x: np.sqrt(x)),
        (p, lambda p: ufl.inner(p, p) * ufl.dx, lambda x: np.sqrt(x))
    ]
    
    for (function, bilinear_form, postprocess) in scaling_operations:
        scalar = postprocess(MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(bilinear_form(function))), op=MPI.SUM))
        function.x.petsc_vec.scale(1. / scalar if scalar != 0 else 1.0)
        function.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

def compute_inf_sup_constant(case: str, n: int = 32) -> tuple[float, fem.Function, fem.Function, fem.Function]:
    """
    Compute the inf-sup constant for a specific element pairing.
    
    Parameters:
        case: String identifier for the velocity-pressure pairing
        n: Mesh resolution
    
    Returns:
        Tuple of (inf_sup_constant, u1_eigenfunction, u2_eigenfunction, p_eigenfunction)
    """
    print(f"\nComputing inf-sup constant for {case} elements on {n}x{n} mesh")
    
    # Create mesh
    msh = create_unit_square_mesh(n)
    
    # Get boundary facets
    boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, wall)
    
    # Create function spaces based on the specified case
    V_element, Q_element = get_function_spaces(case, msh)[0].ufl_element(), get_function_spaces(case, msh)[1].ufl_element()
    
    # Create mixed function space
    W_element = basix.ufl.mixed_element([V_element, Q_element])
    W = fem.functionspace(msh, W_element)
    
    # Test and trial functions
    vq = ufl.TestFunction(W)
    (v, q) = ufl.split(vq)
    up = ufl.TrialFunction(W)
    (u, p) = ufl.split(up)
    
    # Variational forms for the generalized eigenvalue problem
    lhs = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(p, ufl.div(v)) * ufl.dx
           - ufl.inner(ufl.div(u), q) * ufl.dx)
    rhs = - ufl.inner(p, q) * ufl.dx
    
    # Define restriction for DOFs associated to homogenous Dirichlet boundary conditions
    dofs_W = np.arange(0, W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts)
    W0 = W.sub(0)
    V, _ = W0.collapse()
    bdofs_V = fem.locate_dofs_topological((W0, V), msh.topology.dim - 1, boundary_facets)[0]
    restriction = dolfinx.fem.DofMapRestriction(W.dofmap, np.setdiff1d(dofs_W, bdofs_V))
    
    # Assemble lhs and rhs matrices
    A = dolfinx.fem.petsc.assemble_matrix(
        fem.form(lhs), restriction=(restriction, restriction))
    A.assemble()
    B = dolfinx.fem.petsc.assemble_matrix(
        fem.form(rhs), restriction=(restriction, restriction))
    B.assemble()
    
    # Solve the eigenvalue problem
    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(1, PETSc.DECIDE, PETSc.DECIDE)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eps.setTarget(1.e-5)
    eps.getST().setType(SLEPc.ST.Type.SINVERT)
    eps.getST().getKSP().setType("preonly")
    eps.getST().getKSP().getPC().setType("lu")
    eps.getST().getKSP().getPC().setFactorSolverType("mumps")
    eps.solve()
    
    if eps.getConverged() < 1:
        print(f"WARNING: Eigenvalue solver did not converge for {case}")
        return 0.0, None, None, None
    
    # Extract leading eigenvalue and eigenvector
    vr = dolfinx.cpp.fem.petsc.create_vector_block([(restriction.index_map, restriction.index_map_bs)])
    vi = dolfinx.cpp.fem.petsc.create_vector_block([(restriction.index_map, restriction.index_map_bs)])
    eigv = eps.getEigenpair(0, vr, vi)
    r, i = eigv.real, eigv.imag
    
    if abs(i) > 1.e-10:
        print(f"WARNING: Eigenvalue has significant imaginary part: {i}")
    
    inf_sup = np.sqrt(r) if r > 0 else 0.0
    print(f"Inf-sup constant for {case}: {inf_sup}")
    
    # Transform eigenvector into eigenfunction
    r_fun = fem.Function(W)
    vr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    with r_fun.x.petsc_vec.localForm() as r_fun_local, \
            dolfinx.fem.petsc.VecSubVectorWrapper(vr, W.dofmap, restriction) as vr_wrapper:
        r_fun_local[:] = vr_wrapper
    
    u_fun = r_fun.sub(0).collapse()
    (u_fun_1, u_fun_2) = (u_fun.sub(0).collapse(), u_fun.sub(1).collapse())
    p_fun = r_fun.sub(1).collapse()
    
    # Normalize the eigenfunctions
    normalize(u_fun_1, u_fun_2, p_fun)
    
    eps.destroy()
    return inf_sup, u_fun_1, u_fun_2, p_fun

def save_plots(case: str, u1: fem.Function, u2: fem.Function, p: fem.Function) -> None:
    """Save plots of the eigenfunctions."""
    if viskex is None:
        return
    
    # Create a figure directory for this case
    case_dir = output_dir / case
    os.makedirs(case_dir, exist_ok=True)
    
    # Plot velocity components
    fig1 = viskex.dolfinx.plot_scalar_field(u1, "u1")
    fig1.savefig(case_dir / "u1.png")
    plt.close(fig1)
    
    fig2 = viskex.dolfinx.plot_scalar_field(u2, "u2")
    fig2.savefig(case_dir / "u2.png")
    plt.close(fig2)
    
    # Plot pressure
    fig3 = viskex.dolfinx.plot_scalar_field(p, "p")
    fig3.savefig(case_dir / "p.png")
    plt.close(fig3)

def main():
    """Main function to run the comparison."""
    # List of element pairings to test
    pairings = ["P2_x_P1", "RT_x_P1", "BDM_x_P1"]
    
    # List of mesh resolutions
    resolutions = [8, 16, 32]
    
    # Store results
    results = {}
    
    for n in resolutions:
        results[n] = {}
        for case in pairings:
            try:
                inf_sup, u1, u2, p = compute_inf_sup_constant(case, n)
                results[n][case] = inf_sup
                
                # Save plots for the finest resolution
                if n == max(resolutions) and u1 is not None:
                    save_plots(case, u1, u2, p)
            except Exception as e:
                print(f"Error computing inf-sup for {case}: {e}")
                results[n][case] = float('nan')
    
    # Print summary table
    print("\n----- Inf-Sup Constant Summary -----")
    header = "Resolution | " + " | ".join(pairings)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for n in resolutions:
        row = f"{n:^10} | "
        row += " | ".join(f"{results[n][case]:.6f}" if not np.isnan(results[n][case]) else "  N/A  " for case in pairings)
        print(row)
    
    print("-" * len(header))
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, case in enumerate(pairings):
        values = [results[n][case] for n in resolutions if not np.isnan(results[n][case])]
        resols = [n for n in resolutions if not np.isnan(results[n][case])]
        if values:
            plt.plot(resols, values, marker=markers[i % len(markers)], label=case)
    
    plt.xlabel("Mesh Resolution")
    plt.ylabel("Inf-Sup Constant")
    plt.title("Comparison of Inf-Sup Constants for Different Element Pairings")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "inf_sup_comparison.png", dpi=300)
    print(f"\nComparison plot saved to {output_dir / 'inf_sup_comparison.png'}")

if __name__ == "__main__":
    main()
