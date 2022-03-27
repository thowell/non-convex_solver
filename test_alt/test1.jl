using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
using NonConvexSolver 

num_variables = 10
num_equality = 0
num_inequality = 5

x0 = ones(num_variables)

obj(x) = transpose(x) * x
eq(x) = zeros(0)
ineq(x) = x[1:5]

# solver
options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        iterative_refinement=false,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        max_residual_iterations=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                    )

methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(x0, methods, num_variables, num_equality, num_inequality, options=options)
solve!(solver, x0)
