using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
using NonConvexSolver 

num_variables = 2
num_equality = 2
num_inequality = 0

x0 = rand(num_variables)

obj(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
eq(x) = [-(x[1] -1)^3 + x[2] - 1;
             -x[1] - x[2] + 2]
ineq(x) = zeros(0) 

options = Options{Float64}(
                max_residual_iterations=100,
                residual_tolerance=1.0e-5,
                equality_tolerance=1.0e-5,
                verbose=true,
                linear_solver=:QDLDL,
                )

methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(x0, methods, num_variables, num_equality, num_inequality, options=options)
solve!(solver, x0)
