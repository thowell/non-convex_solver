num_variables = 8
num_equality = 7
num_inequality = 8

x0 = zeros(num_variables)

obj(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
eq(x) = [
            2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
            3*x[1] - x[2] - 3.0 - x[6];
            -x[1] + 0.5*x[2] + 4.0 - x[7];
            -x[1] - x[2] + 7.0 - x[8];
            x[3]*x[6];
            x[4]*x[7];
            x[5]*x[8]
        ]
ineq(x) = x

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solv



options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        max_residual_iterations=1000,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = Solver(x0,model,options=options)
@time solve!(s)

x = s.x
x[3]
x[6]

x[4]
x[7]

x[5]
x[8]
