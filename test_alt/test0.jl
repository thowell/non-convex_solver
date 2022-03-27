using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
using NonConvexSolver 

num_variables = 50
num_equality = 30
num_inequality = 3

x0 = ones(num_variables)

obj(x) = transpose(x) * x
eq(x) = x[1:30].^2 .- 1.2
ineq(x) = [x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

# solver
options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        iterative_refinement=true,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        max_residual_iterations=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                    )

methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(x0, methods, num_variables, num_equality, num_inequality, options=options)
solve!(solver, x0)

solver.linear_solver.inertia
solver.data.matrix_symmetric
inertia(solver)

solver.filter

tolerance(0.0, 0.0, solver)
solver.central_path
solver.penalty

tolerance(solver.central_path[1], solver.penalty[1], solver)

search_direction!(solver)
solver.data.step
norm(solver.data.step)


solver.primal_regularization
solver.dual_regularization
solver.constraint_violation

norm(solver.data.step)

solver.variables[solver.indices.slack_primal] + solver.data.step[solver.indices.slack_primal]
solver.variables[solver.indices.slack_dual] + solver.data.step[solver.indices.slack_dual]

maximum_step_size!(solver)
solver.step_size

maximum_dual_step_size!(solver)
solver.dual_step_size

solver.variables[solver.indices.slack_dual] + solver.dual_step_size * solver.data.step[solver.indices.slack_dual]
solver.minimum_step_size

minimum_step_size!(solver)

candidate_step!(solver)  # update s.candidate

solver.variables[solver.indices.slack_primal]
solver.candidate[solver.indices.slack_dual]

check_filter(solver.constraint_violation_candidate, solver.merit_candidate, solver.filter)

switching_condition(solver.data.step[solver.indices.primal], solver)
solver.constraint_violation
solver.min_constraint_violation

sufficient_progress(solver)
solver.fraction_to_boundary

armijo(solver)

solver.merit
solver.merit_candidate
