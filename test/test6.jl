include("../src/interior_point.jl")

n = 2
m = 2

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = -x[1]*x[2] + 2/(3*sqrt(3))
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [-x[1] - x[2]^2 + 1.0;
             x[1] + x[2]]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=ones(Bool,m))

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,
                                                        quasi_newton=:lbfgs,
                                                        ϵ_tol=1.0e-5,
                                                        ϵ_al_tol=1.0e-5,
                                                        verbose=true))
@time solve!(s)
