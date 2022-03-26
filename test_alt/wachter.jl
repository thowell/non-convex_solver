# NOTE: Ipopt fails to converge on this problem.
num_variables = 3
num_equality = 2 
num_inequality = 2

x0 = [-2.0;3.0;1.0]

obj(x) = x[1]
eq(x) = [x[1]^2 - x[2] - 1.0;
         x[1] - x[3] - 0.5;]
ineq(x) = x[2:3]

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = SolverAlt(methods, num_variables, num_equality, num_inequality)

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=[zeros(Bool,2); ones(Bool,2)], cA_idx=[ones(Bool,2); zeros(Bool,2)])

options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        iterative_refinement=true,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        max_residual_iterations=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = Solver(x0,model,options=options)
@time solve!(s)
