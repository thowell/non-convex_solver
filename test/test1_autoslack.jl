# NOTE: Ipopt fails to converge on this problem. This solver also fail, unless
#    δc ≈ 0.1. or augmented Lagrangian is used for constraints

include("../src/interior_point.jl")

n = 1
m = 2

x0 = [-2.0]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = x[1]
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [x[1]^2 - 1.0;
             x[1] - 0.5]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

cI_idx=ones(Bool,m)
cA_idx=ones(Bool,m)
model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)

s = InteriorPointSolver(x0,model)
@time solve!(s)
