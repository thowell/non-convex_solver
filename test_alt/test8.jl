num_variables = 3
num_equality = 1
num_inequality = 0

x0 = rand(num_variables)

obj(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
eq(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]
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