include("src/interior_point.jl")

n = 3
m = 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]
c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}(max_iter=500))
solve!(s,verbose=true)

eval_iterate!(s)

s̄ = RestorationSolver(s)
eval_iterate!(s̄)

kkt_hessian_unreduced!(s̄)
kkt_gradient_unreduced!(s̄)

s̄.DR'*s̄.DR
s̄.μ
s̄.W
ForwardDiff.jacobian(s̄.∇f_func,s̄.x)
tmp(x) = s̄.∇c_func(x)'*s̄.λ
s̄.W .= ForwardDiff.jacobian(tmp,s̄.x)
Array(s̄.Hu)
