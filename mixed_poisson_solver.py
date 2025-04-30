# -*- coding: utf-8 -*-
import logging
import dolfinx
import ufl
import numpy as np
from petsc4py import PETSc
from basix.ufl import element
from dolfinx import fem, mesh
from dolfinx.mesh import CellType, create_unit_square
from mpi4py import MPI
import pyvista as pv

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"DOLFINx version: {dolfinx.__version__}")

# Define the scalar and real types
dtype = dolfinx.default_scalar_type
xdtype = dolfinx.default_real_type


def convert_float64_to_float(data):
    if isinstance(data, dict):
        return {k: convert_float64_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_float64_to_float(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    else:
        return data


# Create mesh function with cell type parameter
def create_mesh(mesh_type="triangle", nx=32, ny=32):
    if mesh_type == "triangle":
        msh = create_unit_square(MPI.COMM_WORLD,
                                 nx,
                                 ny,
                                 CellType.triangle,
                                 dtype=xdtype)
        logger.info("Created a 2D unit square mesh with triangles.")
    elif mesh_type == "quadrilateral":
        msh = create_unit_square(MPI.COMM_WORLD,
                                 nx,
                                 ny,
                                 CellType.quadrilateral,
                                 dtype=xdtype)
        logger.info("Created a 2D unit square mesh with quadrilaterals.")
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}")
    return msh


# Function to define function spaces for different cases
def get_function_spaces(case, msh):
    k = 1  # Polynomial degree
    if case == "P1_x_P1":
        # Case 1: P1 x P1 (Leads to singular system)
        V = fem.functionspace(
            msh,
            element("Lagrange",
                    msh.basix_cell(),
                    k,
                    shape=(msh.geometry.dim, ),
                    dtype=xdtype))
        W = fem.functionspace(
            msh, element("Lagrange", msh.basix_cell(), k, dtype=xdtype))
    elif case == "P1_x_P0":
        # Case 2: P1 x P0 (Oscillatory solution)
        V = fem.functionspace(
            msh,
            element("Lagrange",
                    msh.basix_cell(),
                    k,
                    shape=(msh.geometry.dim, ),
                    dtype=xdtype))
        W = fem.functionspace(
            msh,
            element("Discontinuous Lagrange",
                    msh.basix_cell(),
                    k - 1,
                    dtype=xdtype))
    elif case == "RT_x_P0":
        # Case 3: RT x P0 (Stable solution)
        V = fem.functionspace(msh,
                              element("RT", msh.basix_cell(), k, dtype=xdtype))
        W = fem.functionspace(
            msh,
            element("Discontinuous Lagrange",
                    msh.basix_cell(),
                    k - 1,
                    dtype=xdtype))
    elif case == "BDM_x_P0":
        # Case 4: BDM x P0 (Stable solution)
        V = fem.functionspace(
            msh, element("BDM", msh.basix_cell(), k, dtype=xdtype))
        W = fem.functionspace(
            msh,
            element("Discontinuous Lagrange",
                    msh.basix_cell(),
                    k - 1,
                    dtype=xdtype))
    elif case == "QUADS_x_P0":
        # Case 5: QUADS x P0 (Stable solution)
        V = fem.functionspace(
            msh,
            element("Lagrange",
                    msh.basix_cell(),
                    k,
                    shape=(msh.geometry.dim, ),
                    dtype=xdtype))
        W = fem.functionspace(
            msh,
            element("Discontinuous Lagrange",
                    msh.basix_cell(),
                    k - 1,
                    dtype=xdtype))
    else:
        raise ValueError(f"Unknown case: {case}")
    logger.info("Defined function spaces.")
    return V, W


# Solve the Mixed Poisson problem for a given case
def solve_case(case, msh):
    logger.info(f"Solving case {case}")
    V, W = get_function_spaces(case, msh)

    # Define trial and test functions
    (sigma, u) = ufl.TrialFunction(V), ufl.TrialFunction(W)
    (tau, v) = ufl.TestFunction(V), ufl.TestFunction(W)

    # Define the source function
    x = ufl.SpatialCoordinate(msh)
    f = -2 * (np.pi**2) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Define bilinear and linear forms
    dx = ufl.Measure("dx", msh)
    a = [
        [ufl.inner(sigma, tau) * dx,
         ufl.inner(u, ufl.div(tau)) * dx],
        [ufl.inner(ufl.div(sigma), v) * dx, None],
    ]
    zero_function = fem.Function(V)
    L = [ufl.inner(zero_function, tau) * dx, -ufl.inner(f, v) * dx]

    a, L = fem.form(a, dtype=dtype), fem.form(L, dtype=dtype)

    # Boundary conditions
    fdim = msh.topology.dim - 1
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    # Get all boundary DOFs for the vector function space
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Create a zero function for Dirichlet BC
    u_bc = fem.Function(V, dtype=dtype)
    u_bc.x.array[:] = 0.0  # Set all components to zero

    # Apply homogeneous Dirichlet boundary condition on all boundaries
    bcs = [fem.dirichletbc(u_bc, boundary_dofs)]
    logger.info(
        "Defined homogeneous Dirichlet boundary conditions on all boundaries.")

    # Assemble matrix operator and apply boundary conditions
    A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    b = fem.petsc.assemble_vector_nest(L)
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)

    bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)
    logger.info("Assembled matrix and RHS vector.")

    # Define preconditioner
    a_p = fem.form(
        [
            [
                ufl.inner(sigma, tau) * dx +
                ufl.inner(ufl.div(sigma), ufl.div(tau)) * dx, None
            ],
            [None, ufl.inner(u, v) * dx],
        ],
        dtype=dtype,
    )
    P = fem.petsc.assemble_matrix_nest(a_p, bcs=bcs)
    P.assemble()

    # Solve the system
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setMonitor(lambda ksp, its, rnorm: logger.info(
        f"Iteration: {its}, residual: {rnorm}") if its % 100 == 0 else None)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-6, max_it=1000)
    ksp.setGMRESRestart(200)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    sigma, u = fem.Function(V, dtype=dtype), fem.Function(W, dtype=dtype)
    x = PETSc.Vec().createNest([sigma.x.petsc_vec, u.x.petsc_vec])
    ksp.solve(b, x)

    reason = ksp.getConvergedReason()
    if reason <= 0:
        logger.warning(f"Krylov solver has not converged: {reason}")
    else:
        logger.info(f"Case {case}: Krylov solver converged successfully.")

    return u


def compute_brezzi_infsup(case, msh):
    # Compute the Brezzi inf-sup constant for a given case.
    from slepc4py import SLEPc
    from basix.ufl import mixed_element as MixedElement
    from dolfinx.fem.petsc import assemble_matrix
    logger.info(f"Computing Brezzi inf-sup constant for case {case}")

    # Get the function spaces
    V, W = get_function_spaces(case, msh)

    # Create mixed function space
    mixed_element = MixedElement([V.ufl_element(), W.ufl_element()])
    Z = fem.functionspace(msh, mixed_element)

    # Define trial and test functions on the mixed space
    z = ufl.TrialFunction(Z)
    w = ufl.TestFunction(Z)

    # Extract components (this is a valid way to mix spaces)
    sigma, u = ufl.split(z)
    tau, v = ufl.split(w)

    dx = ufl.Measure("dx", msh)
    # Define forms for eigenvalue problem
    lhs = ufl.inner(sigma, tau) * dx + ufl.inner(
        u, ufl.div(tau)) * dx + ufl.inner(ufl.div(sigma), v) * dx
    rhs = -ufl.inner(u, v) * dx

    # Convert to forms
    a_form = fem.form(lhs, dtype=dtype)
    b_form = fem.form(rhs, dtype=dtype)

    # Assemble matrices
    A = assemble_matrix(a_form)
    A.assemble()

    B = assemble_matrix(b_form)
    B.assemble()

    # Set up eigenvalue solver with shift-and-invert to avoid zero pivots
    eps = SLEPc.EPS().create(MPI.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setProblemType(
        SLEPc.EPS.ProblemType.GNHEP)  # Generalized Hermitian eigenproblem.

    eps.setDimensions(1, PETSc.DECIDE, PETSc.DECIDE)
    # Set target and which eigenpairs(Smallest real parts) to compute
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

    # Avoid numerical artifacts
    eps.setTarget(1.e-5)
    eps.getST().setType(SLEPc.ST.Type.SINVERT)
    eps.getST().getKSP().setType("preonly")
    eps.getST().getKSP().getPC().setType("lu")
    eps.getST().getKSP().getPC().setFactorSolverType("mumps")

    # Solve eigenvalue problem
    logger.info("Solving eigenvalue problem for inf-sup constant...")
    eps.solve()

    # Get eigenvalues
    nconv = eps.getConverged()
    logger.info(f"Number of converged eigenvalues: {nconv}")

    if nconv > 0:
        eigenvalues = [abs(eps.getEigenvalue(i).real) for i in range(nconv)]
        eigenvalues.sort()
        non_zero_eigvals = [ev for ev in eigenvalues if ev > 1e-10]

        if non_zero_eigvals:
            inf_sup = np.sqrt(non_zero_eigvals[0])
            logger.info(f"Inf-sup constant for case {case}: {inf_sup}")
            return inf_sup
        else:
            logger.warning(f"No positive eigenvalues found for case {case}")
            return 0.0
    else:
        logger.warning(f"No converged eigenvalues for case {case}")
        return 0.0


# Plot function using PyVista
def plot_contour(u, case):
    from dolfinx.plot import vtk_mesh
    topology, cell_types, geometry = vtk_mesh(
        u.function_space.mesh, u.function_space.mesh.topology.dim)
    values = u.x.array

    min_val = u.x.array.min()
    max_val = u.x.array.max()
    logger.info(
        f"Solution range for case {case}: min={min_val}, max={max_val}")
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid["u"] = values

    # Create a plotter object for saving PNG
    plotter = pv.Plotter(off_screen=True)
    abs_max = max(abs(min_val), abs(max_val))
    plotter.add_mesh(grid,
                     scalars="u",
                     cmap="coolwarm",
                     show_edges=True,
                     clim=[-abs_max, abs_max])
    plotter.add_text(
        f"Contour Plot of u for Case {case} (min={min_val:.4f}, max={max_val:.4f})",
        position="upper_edge")

    plotter.add_scalar_bar("u", vertical=True, position_x=0.85)

    # Save as PNG
    png_filename = f"case_{case}_contour_plot.png"
    plotter.screenshot(png_filename, transparent_background=False)
    plotter.close()

    filename = f"case_{case}_contour_plot.vtk"
    grid.plot(scalars="u",
              cmap="coolwarm",
              show_edges=True,
              text=f"Contour Plot of u for Case {case}")
    # grid.save(filename)
    logger.info(f"Saved contour plot for Case {case} as PNG and VTK files")


# Run different mesh sizes and compute inf-sup constants
def compute_mesh_size(msh):
    """Compute the maximum element diameter in the mesh"""
    h_max = 0.0
    tdim = msh.topology.dim

    # Get coordinates
    x = msh.geometry.x

    # Create connectivity
    msh.topology.create_connectivity(tdim, 0)
    conn = msh.topology.connectivity(tdim, 0)

    # Loop over cells and compute maximum diameter
    for cell in range(msh.topology.index_map(tdim).size_local):
        vertices = conn.links(cell)
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(x[vertices[i]] - x[vertices[j]])
                h_max = max(h_max, dist)

    # Get global maximum across all processes
    h_max = msh.comm.allreduce(h_max, op=MPI.MAX)
    return h_max


if __name__ == "__main__":
    # Solve and plot all cases with different mesh sizes
    mesh_sizes = [2, 4, 8, 16, 32, 64]  # Different mesh resolutions
    cases = ["P1_x_P1", "P1_x_P0", "RT_x_P0", "BDM_x_P0", "QUADS_x_P0"]
    results = {
        case: {
            "mesh_size": [],
            "h": [],
            "inf_sup": []
        }
        for case in cases
    }

    for case in cases:
        logger.info(f"Testing case {case} with different mesh sizes")

        for nx in mesh_sizes:
            if case == "QUADS_x_P0":
                msh = create_mesh(mesh_type="quadrilateral", nx=nx, ny=nx)
            else:
                msh = create_mesh(mesh_type="triangle", nx=nx, ny=nx)

            # Compute mesh size
            h = compute_mesh_size(msh)
            logger.info(f"Case {case}, mesh size h = {h}")

            try:
                # Compute inf-sup constant
                inf_sup = compute_brezzi_infsup(case, msh)
                logger.info(
                    f"Inf-sup constant for {case} with mesh size={nx} h={h}: {inf_sup}"
                )

                # Store results
                results[case]["mesh_size"].append(nx)
                results[case]["h"].append(h)
                results[case]["inf_sup"].append(inf_sup)

                # Solve the problem only for the finest mesh
                if nx == mesh_sizes[-1]:
                    u_sol = solve_case(case, msh)
                    plot_contour(u_sol, case)

            except Exception as e:
                logger.error(
                    f"Error in case {case} with mesh size={nx} h={h}: {e}")
    results = convert_float64_to_float(results)
    logger.info(f"Results: {results}")
    # Plot h vs inf-sup constant
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    with plt.style.context('science'):
        plt.figure(figsize=(16, 12))
        markers = ['o', 's', '^', 'D']
        line_styles = ['-', '--', '-.', ':']
        colors = ['C0', 'C1', 'C2', 'C3']

        for i, case in enumerate(cases[1:]):
            plt.subplot(2, 2, i + 1)
            if results[case]["mesh_size"] and results[case]["inf_sup"]:
                plt.plot(results[case]["mesh_size"],
                         results[case]["inf_sup"],
                         color=colors[i],
                         marker=markers[i],
                         linestyle=line_styles[i],
                         linewidth=2,
                         markersize=10)

            plt.xlabel('Mesh size h', fontsize=12)
            plt.ylabel('Inf-sup constant b_h', fontsize=12)
            plt.title(f'Case: {case}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.suptitle('Inf-sup constant vs mesh size for different cases',
                     fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("infsup_vs_meshsize_subplots.png", dpi=300)

        plt.figure(figsize=(12, 8))
        for i, case in enumerate(cases[1:]):
            if results[case]["mesh_size"] and results[case]["inf_sup"]:
                plt.plot(results[case]["mesh_size"],
                         results[case]["inf_sup"],
                         color=colors[i],
                         marker=markers[i],
                         linestyle=line_styles[i],
                         linewidth=2.5,
                         markersize=10,
                         label=case)

        plt.xlabel('Mesh size h', fontsize=14)
        plt.ylabel('Inf-sup constant b_h', fontsize=14)
        plt.title('Inf-sup constant vs mesh size', fontsize=16)
        plt.legend(fontsize=12,
                   frameon=True,
                   facecolor='white',
                   edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("infsup_vs_meshsize_combined.png", dpi=300)
        print("Saved subplots and combined plot of h vs b_h")
        plt.close('all')
