num_variables = 2
num_equality = 2

x0 = rand(num_variables)

obj(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
eq(x) = [-(x[1] -1)^3 + x[2] - 1;
             -x[1] - x[2] + 2]
ineq(x) = zeros(0) 

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = SolverAlt(methods, num_variables, num_equality, num_inequality)



model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

s = Solver(x0,model,options=options=Options{Float64}(linear_solve_type=:symmetric,
                                                        residual_tolerance=1.0e-5,
                                                        equality_tolerance=1.0e-5,
                                                        verbose=true,
                                                        linear_solver=:QDLDL,
                                                        max_residual_iterations=500,
                                                        ))
@time solve!(s)
