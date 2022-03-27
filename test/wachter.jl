# NOTE: Ipopt fails to converge on this problem.
n = 3
m = 2 + 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)

f_func(x) = x[1]

c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5;
             x[2:3]]

model = Model(n,m,xL,f_func,c_func,cI_idx=[zeros(Bool,2); ones(Bool,2)], cA_idx=[ones(Bool,2); zeros(Bool,2)])

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
