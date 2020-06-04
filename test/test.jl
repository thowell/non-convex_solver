include("../src/interior_point.jl")

n = 50
m = 30

x0 = ones(n)

xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = x[1:m].^2 .- 1.2
# c_func(x) = x[1:m] .- 1.2

c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m))

opts = Options{Float64}(
                        kkt_solve=:symmetric,
                        iterative_refinement=false,
                        ϵ_tol=1.0e-8,
                        ϵ_al_tol=1.0e-8,
                        max_iterative_refinement=10,
                        max_iter=250,
                        verbose=true,
                        quasi_newton=:none,
                        quasi_newton_approx=:lagrangian
                        )

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)

solve(inertia_correction_qdldl(s.s),-s.s.h_sym)

F = inertia_correction_qdldl(s.s)
s.s.dxy .= solve(F,-s.s.h_sym)
# cholesky(s.s.∇²L + 1.0*I)
# rank(s.s.∇²L)
# rank(get_∇c(s.s.model))
# s.s.model.m
#
# sum(eigen(Array(s.s.H_sym)).values .> 0)
# sum(eigen(Array(s.s.H_sym)).values .< 0)
#
# inertia_correction!(s.s)
# s.s.δ
# s.s.δw
# s.s.inertia
# s.s.LBL
#
# qdldl(s.s.H_sym + 1.0*I)
