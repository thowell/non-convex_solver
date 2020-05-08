# NOTE: Ipopt fails to converge on this problem. This solver also fail, unless
#    δc ≈ 0.1. or augmented Lagrangian is used for constraints

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

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)
opts = Options{Float64}(kkt_solve=:symmetric,
                        relax_bnds=true,
                        nlp_scaling=true,
                        single_bnds_damping=true,
                        iterative_refinement=true,
                        max_iterative_refinement=100,
                        max_iter=100,
                        verbose=false)

s = InteriorPointSolver(x0,model,cI_idx=ones(Bool,m),opts=opts)
s.s.ρ = 100.
@time solve!(s)
