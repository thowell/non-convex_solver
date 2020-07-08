include("../src/non-convex_solver.jl")

n = 2
m = 2

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = -x[1]*x[2] + 2/(3*sqrt(3))

c_func(x) = [-x[1] - x[2]^2 + 1.0;
             x[1] + x[2]]

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

opts = Options{Float64}(kkt_solve=:symmetric,
                max_iter=100,
                quasi_newton=:bfgs,
                ϵ_tol=1.0e-8,
                ϵ_al_tol=1.0e-8,
                verbose=true,
                lbfgs_length=6)

s = NonConvexSolver(x0,model,opts=opts)

@time solve!(s)
