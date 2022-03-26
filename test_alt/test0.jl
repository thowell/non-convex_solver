num_variables = 50
num_equality = 30
num_inequality =  3

x0 = ones(num_variables)

obj(x) = x'*x
eq(x) = [x[1:30].^2 .- 1.2; x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]
ineq(x) = zeros(0) 

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = SolverAlt(methods, num_variables, num_equality, num_inequality)


model = Model(n,m,xL,xU,f_func,c_func,cI_idx=[zeros(Bool,30); ones(Bool,3)],cA_idx=[ones(Bool,30); zeros(Bool,3)])

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
