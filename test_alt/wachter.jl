# NOTE: Ipopt fails to converge on this problem.
num_variables = 3
num_equality = 2 
num_inequality = 2

x0 = [-2.0;3.0;1.0]

obj(x) = x[1]
eq(x) = [x[1]^2 - x[2] - 1.0;
         x[1] - x[3] - 0.5;]
ineq(x) = x[2:3]

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
