num_variables = 2
num_equality = 1
num_inequality = 0

x0 = [2.0; 1.0]

obj(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
eq(x) = [x[1]^2 + x[2]^2 - 1.0]
ineq(x) = zeros(0) 

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = SolverAlt(methods, num_variables, num_equality, num_inequality)



model = Model(n,m,xL,xU,f_func,c_func,cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m))

options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        iterative_refinement=false,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        max_residual_iterations=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = Solver(x0,model,options=options)
@time solve!(s)
