function tolerance(central_path, penalty, solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, solver.variables;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)

    dual_equality = solver.variables[solver.indices.equality]
    dual_inequality = solver.variables[solver.indices.inequality]
    slack_primal = solver.variables[solver.indices.slack_primal]
    slack_dual = solver.variables[solver.indices.slack_dual]

    lagrangian_gradient_variables = solver.problem.objective_gradient 
    lagrangian_gradient_variables .+= solver.problem.equality_jacobian' * dual_equality 
    lagrangian_gradient_variables .+= solver.problem.inequality_jacobian' * dual_inequality
    lagrangian_gradient_slack = -dual_inequality - slack_dual
    lagrangian_gradient = [lagrangian_gradient_variables; lagrangian_gradient_slack]

    equality = solver.problem.equality 
    if penalty[1] > 0.0 
        equality .+= 1.0 / penalty[1] * (solver.dual - dual_equality)
    end

    inequality = solver.problem.inequality - slack_primal

    return max(norm(lagrangian_gradient, Inf),
                norm(equality, Inf),
                norm(inequality, Inf),
                norm(slack_primal .* slack_dual .- central_path, Inf),
                )
end

"""
    eval_objective!(s::Solver)

Evaluate the objective value and it's first and second-order derivatives
"""
function eval_objective!(solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, solver.variables;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)
    
    return nothing
end

"""
    eval_constraints!(s::Solver)

Evaluate the constraints and their first and second-order derivatives. Also compute the
constraint residual `constraint_violation`.
"""
function eval_constraints!(solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, solver.variables;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)

    c = [solver.problem.equality; solver.problem.inequality - solver.variables[solver.indices.slack_primal]]
    c = solver.scaling_constraints * c
   
    solver.constraint_violation = norm(c, 1)
    return nothing
end

"""
    eval_lagrangian!(s::Solver)

Evaluate the first and second derivatives of the Lagrangian
"""
function eval_lagrangian!(solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, solver.variables;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)

    dual_equality = solver.variables[solver.indices.equality]
    dual_inequality = solver.variables[solver.indices.inequality]
    slack_primal = solver.variables[solver.indices.slack_primal]
    slack_dual = solver.variables[solver.indices.slack_dual]

    lagrangian_gradient_variables = solver.problem.objective_gradient 
    lagrangian_gradient_variables .+= solver.problem.equality_jacobian' * dual_equality 
    lagrangian_gradient_variables .+= solver.problem.inequality_jacobian' * dual_inequality
    lagrangian_gradient_slack = - dual_inequality - slack_dual
    solver.lagrangian_gradient .= [lagrangian_gradient_variables; lagrangian_gradient_slack]

    return nothing
end

"""
    eval_barrier(s::Solver)

Evaluate merit objective and it's gradient
"""
function eval_barrier!(solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, solver.variables;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)

    slack_primal = solver.variables[solver.indices.slack_primal]

    solver.merit = solver.methods.objective(solver.variables)
    solver.merit -= solver.central_path[1] * sum(log.(slack_primal))
    solver.merit += solver.dual' * solver.problem.equality + 0.5 * solver.penalty[1] * solver.problem.equality' * solver.problem.equality

    solver.merit_gradient[solver.indices.variables] .= solver.problem.objective_gradient
    solver.merit_gradient[solver.indices.slack_primal] -= solver.central_path[1] ./ slack_primal
    solver.merit_gradient[solver.indices.variables] += solver.problem.equality_jacobian' * (solver.dual + solver.penalty[1] * solver.problem.equality)
    
    return nothing
end

"""
    evaluate!(s::Solver)

Evaluate all critical values for the current iterate stored in `s.x` and `s.y`, including
bound constraints, objective, constraints, Lagrangian, and merit objective, and their
required derivatives.
"""
function evaluate!(s::Solver)
    eval_objective!(s)
    eval_constraints!(s)
    eval_lagrangian!(s)
    eval_barrier!(s)
    residual!(s.data, s.problem, s.indices, s.variables, s.central_path, s.penalty, s.dual)

    s.constraint_violation = norm(s.data.residual[s.indices.constraints])

    return nothing
end

"""
    central_path(central_path, scaling_central_path, exponent_central_path, residual_tolerance)
    central_path(s::Solver)

Update the penalty parameter (Eq. 7) with constants scaling_central_path ∈ (0,1), exponent_central_path ∈ (1,2)
"""
central_path(central_path, scaling_central_path, exponent_central_path, residual_tolerance) = max(residual_tolerance/10.,min(scaling_central_path*central_path[1], central_path[1]^exponent_central_path))
function central_path!(s::Solver)
    s.central_path[1] = central_path(s.central_path, s.options.scaling_central_path, s.options.exponent_central_path, s.options.residual_tolerance)
    return nothing
end

"""
    fraction_to_boundary(central_path, min_fraction_to_boundary)
    fraction_to_boundary(s::Solver)

Update the "fraction-to-boundary" parameter (Eq. 8) where min_fraction_to_boundary ∈ (0,1) is it's minimum value.
"""
fraction_to_boundary(central_path,min_fraction_to_boundary) = max(min_fraction_to_boundary,1.0-central_path[1])
function fraction_to_boundary!(s::Solver)
    s.fraction_to_boundary = fraction_to_boundary(s.central_path, s.options.min_fraction_to_boundary)
    return nothing
end

"""
    fraction_to_boundary(x, d, step_size, fraction_to_boundary)

Check if the `x` satisfies the "fraction-to-boundary" rule (Eq. 15)
"""
fraction_to_boundary(x,d,step_size,fraction_to_boundary) = all(x + step_size*d .>= (1 - fraction_to_boundary)*x)

function initialize_max_constraint_violation(constraint_violation)
    max_constraint_violation = 1.0e4*max(1.0,constraint_violation)
    return max_constraint_violation
end

function initialize_min_constraint_violation(constraint_violation)
    min_constraint_violation = 1.0e-4*max(1.0,constraint_violation)
    return min_constraint_violation
end

"""
    constraint_violation(x, s::Solver)

Calculate the 1-norm of the constraints
"""
function constraint_violation(x, solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, x;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)

    constraints = [solver.problem.equality; solver.problem.inequality - solver.variables[solver.indices.slack_primal]]
    constraints = solver.scaling_constraints * constraints

    return norm(constraints, 1)
end

function merit(x, solver::Solver)
    problem!(solver.problem, solver.methods, solver.indices, x;
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)

    constraints = [solver.problem.equality; solver.problem.inequality - solver.variables[solver.indices.slack_primal]]
    constraints = solver.scaling_constraints * constraints

    # TODO: scaling
    M = solver.methods.objective(x[solver.indices.variables]) 
    M -= solver.central_path[1] * sum(log.(x[solver.indices.slack_primal]))
    M += solver.dual' * solver.problem.equality + 0.5 * solver.penalty[1] * solver.problem.equality' * solver.problem.equality

    return M
end

"""
    accept_step!(s::Solver)

Accept the current step, copying the candidate primals and duals into the current iterate.
"""
function accept_step!(s::Solver)
    s.variables[s.indices.primal] .= s.candidate[s.indices.primal]
    s.variables[s.indices.equality] .+= s.step_size * s.data.step[s.indices.equality]
    s.variables[s.indices.inequality] .+= s.step_size * s.data.step[s.indices.inequality]
    s.variables[s.indices.slack_dual] .+= s.dual_step_size * s.data.step[s.indices.slack_dual]
    return nothing
end
