"""
    search_direction!(solver::Solver)

Compute the search direction by solving the KKT system. Includes both inertia
correction and iterative refinement.
"""
function search_direction!(solver::Solver)
    matrix!(solver.data, solver.problem, solver.indices, solver.variables,
        solver.central_path, solver.penalty, solver.dual,
        solver.primal_regularization, solver.dual_regularization)
    
    inertia_correction!(solver)

    step_symmetric!(solver.data.step, solver.data.residual, solver.data.matrix, 
        solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.matrix_symmetric, 
        solver.indices, solver.linear_solver)

    solver.options.iterative_refinement && iterative_refinement!(solver.data.step, solver)

    solver.data.step .*= -1.0
    return
end
