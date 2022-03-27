num_variables = 8
num_equality = 7
num_inequality = 8

x0 = zeros(num_variables)

obj(x) = (x[1] - 5.0)^2 + (2.0 * x[2] + 1.0)^2
eq(x) = [
            2.0 * (x[2] - 1.0) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
            3*x[1] - x[2] - 3.0 - x[6];
            -x[1] + 0.5*x[2] + 4.0 - x[7];
            -x[1] - x[2] + 7.0 - x[8];
            x[3]*x[6];
            x[4]*x[7];
            x[5]*x[8]
        ]
ineq(x) = x

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

x = solver.variables
x[3]
x[6]

x[4]
x[7]

x[5]
x[8]
