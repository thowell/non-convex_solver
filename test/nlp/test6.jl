n = 2
m = 2

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = -x[1]*x[2] + 2/(3*sqrt(3))

c_func(x) = [-x[1] - x[2]^2 + 1.0;
             x[1] + x[2]]

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

options = Options{Float64}(linear_solve_type=:symmetric,
                max_residual_iterations=100,
                residual_tolerance=1.0e-5,
                equality_tolerance=1.0e-5,
                verbose=true,
                linear_solver=:QDLDL,
                )

s = Solver(x0,model,options=options)

@time solve!(s)
