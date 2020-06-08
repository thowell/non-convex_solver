include("../src/non-convex_solver.jl")

n = 2
m = 2

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2

c_func(x) = [-(x[1] -1)^3 + x[2] - 1;
             -x[1] - x[2] + 2]

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m))

s = NonConvexSolver(x0,model,opts=opts=Options{Float64}(kkt_solve=:symmetric,
                                                        quasi_newton=:bfgs,
                                                        quasi_newton_approx=:lagrangian,
                                                        ϵ_tol=1.0e-8,
                                                        ϵ_al_tol=1.0e-8,
                                                        verbose=true,
                                                        max_iter=250))
@time solve!(s)
