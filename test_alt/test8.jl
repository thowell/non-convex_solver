num_variables = 3
num_equality = 1

x0 = rand(num_variables)

obj(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
eq(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]
ineq(x) = zeros(0)

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = SolverAlt(methods, num_variables, num_equality, num_inequality)


model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

s = Solver(x0,model,options=Options{Float64}(linear_solve_type=:symmetric,
                                                        residual_tolerance=1.0e-5,
                                                        equality_tolerance=1.0e-5,
                                                        linear_solver=:QDLDL,
                                                        verbose=true,
                                                        max_residual_iterations=250
                                                        ))
@time solve!(s)
