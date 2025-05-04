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

def get_function_spaces(case: str, msh: dolfinx.mesh.Mesh):
    """
    Define function spaces for different velocity-pressure element pairings.
    
    Parameters:
        case: String identifier for the pairing (e.g., "P2_x_P1", "RT_x_P0")
        msh: Mesh to define the spaces on
    
    Returns:
        Tuple of (velocity_space, pressure_space)
    """
    from dolfinx import mesh, fem
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
            basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        )
    elif case == "BDM_x_P1":
        # Brezzi-Douglas-Marini elements (stable)
        V = fem.functionspace(
            msh, 
            basix.ufl.element("BDM", msh.basix_cell(), 2)
        )
        Q = fem.functionspace(
            msh, 
            basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        )
    else:
        raise ValueError(f"Unknown case: {case}")
    
    print(f"Created function spaces for {case}")
    return V, Q


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
    boundary_facets = dolfinx.mesh.locate_entities_boundary(msh, msh.topology.dim - 1, wall)
    
    # Create function spaces based on the specified case
    V_element, Q_element = get_function_spaces(case, msh)[0].ufl_element(), get_function_spaces(case, msh)[1].ufl_element()
    
    # Create mixed function space

    V = dolfinx.fem.functionspace(mesh, V_element)
    Q = dolfinx.fem.functionspace(mesh, Q_element)

    (v, q) = (ufl.TestFunction(V), ufl.TestFunction(Q))
    (u, p) = (ufl.TrialFunction(V), ufl.TrialFunction(Q))
            
    # Variational forms
    lhs = [[ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, - ufl.inner(p, ufl.div(v)) * ufl.dx],
           [- ufl.inner(ufl.div(u), q) * ufl.dx, None]]
    rhs = [[None, None],
           [None, - ufl.inner(p, q) * ufl.dx]]
    rhs[0][0] = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0)) * ufl.inner(u, v) * ufl.dx

    # Define restriction for DOFs associated to homogenous Dirichlet boundary conditions
    dofs_V = np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
    bdofs_V = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    dofs_Q = np.arange(0, Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts)
    restriction_V = multiphenicsx.fem.DofMapRestriction(V.dofmap, np.setdiff1d(dofs_V, bdofs_V))
    restriction_Q = multiphenicsx.fem.DofMapRestriction(Q.dofmap, dofs_Q)
    restriction = [restriction_V, restriction_Q]

    # Assemble lhs and rhs matrices
    A = multiphenicsx.fem.petsc.assemble_matrix_block(
        dolfinx.fem.form(lhs), bcs=[], restriction=(restriction, restriction))
    A.assemble()
    B = multiphenicsx.fem.petsc.assemble_matrix_block(
        dolfinx.fem.form(rhs), bcs=[], restriction=(restriction, restriction))
    B.assemble()
    
    # Solve
    eps = slepc4py.SLEPc.EPS().create(mesh.comm)
    eps.setOperators(A, B)
    eps.setProblemType(slepc4py.SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(1, petsc4py.PETSc.DECIDE, petsc4py.PETSc.DECIDE)
    eps.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_REAL)
    eps.setTarget(1.e-5)
    eps.getST().setType(slepc4py.SLEPc.ST.Type.SINVERT)
    eps.getST().getKSP().setType("preonly")
    eps.getST().getKSP().getPC().setType("lu")
    eps.getST().getKSP().getPC().setFactorSolverType("mumps")
    eps.solve()
    assert eps.getConverged() >= 1

    # Extract leading eigenvalue and eigenvector
    vr = dolfinx.cpp.fem.petsc.create_vector_block(
        [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction])
    vi = dolfinx.cpp.fem.petsc.create_vector_block(
        [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction])
    eigv = eps.getEigenpair(0, vr, vi)
    r, i = eigv.real, eigv.imag
    assert abs(i) < 1.e-10
    assert r > 0., "r = " + str(r) + " is not positive"
    print("Inf-sup constant (block): ", np.sqrt(r))

    eps.destroy()
    return r


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
    pairings = ["P2_x_P1", "RT_x_P1", "BDM_x_P1"]
    
    # List of mesh resolutions
    resolutions = [8, 16, 32]

    # Store results
    results = {}

    for n in resolutions:
        results[n] = {}
        for case in pairings:
            try:
                inf_sup = compute_inf_sup_constant(case, n)
                results[n][case] = inf_sup
                
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

    import matplotlib.pyplot as plt
    try:
        import scienceplots
    except:
        !pip install --quiet scienceplots
        import scienceplots
    plt.style.use(['science', 'no-latex'])

    # Then continue with your plotting code
    with plt.style.context(['science', 'no-latex']):
    
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
