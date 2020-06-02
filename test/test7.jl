include("../src/interior_point.jl")

n = 2
m = 2

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [-(x[1] -1)^3 + x[2] - 1;
             -x[1] - x[2] + 2]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

s = InteriorPointSolver(x0,model,opts=opts=Options{Float64}(kkt_solve=:symmetric,
                                                        quasi_newton=:bfgs,
                                                        quasi_newton_approx=:lagrangian,
                                                        ϵ_tol=1.0e-8,
                                                        ϵ_al_tol=1.0e-8,
                                                        verbose=true,
                                                        max_iter=250))
@time solve!(s)
s.s.qn.fail_cnt
