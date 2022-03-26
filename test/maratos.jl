n = 2
m = 1

x0 = [2.0; 1.0]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
c_func(x) = [x[1]^2 + x[2]^2 - 1.0]

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
