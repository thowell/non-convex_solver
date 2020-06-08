include("../src/non-convex_solver.jl")

n = 3
m = 1

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=ones(Bool,m))

s = NonConvexSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,
                                                        quasi_newton=:bfgs,
                                                        quasi_newton_approx=:lagrangian,
                                                        ϵ_tol=1.0e-8,
                                                        ϵ_al_tol=1.0e-8,
                                                        verbose=true,
                                                        max_iter=250))
@time solve!(s)
