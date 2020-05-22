# NOTE: Ipopt fails to converge on this problem. This solver also fail, unless
#    δc ≈ 0.1. or augmented Lagrangian is used for constraints

include("../src/interior_point.jl")

n = 3
m = 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cA_idx=ones(Bool,m))

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,verbose=true))#,ϵ_tol=1.0e-6))
@time solve!(s)


# eval_step!(s.s)
# s.s.opts.iterative_refinement = true
# search_direction_symmetric!(s.s)
#
# hx = s.s.h[1:n]
# hs = s.s.h[get_s_idx(s.s)]
# hr = s.s.h[get_r_idx(s.s)]
# hyI = s.s.h[s.s.idx.yI]
# hyE = s.s.h[s.s.idx.yE]
# hyA = s.s.h[s.s.idx.yA]
# hzL = s.s.h[s.s.idx.zL[1:s.s.model_opt.nL]]
# hzs = s.s.h[s.s.idx.zL[s.s.model_opt.nL .+ (1:s.s.model_opt.mI)]]
# hzU = s.s.h[s.s.idx.zU]
#
#
# dx = s.s.d[1:n]
# ds = s.s.d[get_s_idx(s.s)]
# dr = s.s.d[get_r_idx(s.s)]
# dyI = s.s.d[s.s.idx.yI]
# dyE = s.s.d[s.s.idx.yE]
# dyA = s.s.d[s.s.idx.yA]
# dzL = s.s.d[s.s.idx.zL[1:s.s.model_opt.nL]]
# dzs = s.s.d[s.s.idx.zL[s.s.model_opt.nL .+ (1:s.s.model_opt.mI)]]
# dzU = s.s.d[s.s.idx.zU]
#
# ΔxL = s.s.ΔxL[1:s.s.model_opt.nL]
# ΔsL = s.s.ΔxL[s.s.model_opt.nL .+ (1:s.s.model.mI)]
# ΔxU = s.s.ΔxU
# zL = s.s.zL[1:s.s.model_opt.nL]
# zS = s.s.zL[s.s.model_opt.nL .+ (1:s.s.model.mI)]
# zU = s.s.zU
#
# norm(dr - 1.0/s.s.ρ*(dyA - hr))
# norm(dzL - -(zL.*s.s.d[s.s.idx.xL][1:s.s.model_opt.nL] + hzL)./ΔxL)
# norm(dzU - (zU.*s.s.d[s.s.idx.xU][1:s.s.model_opt.nU] - hzU)./ΔxU)
# norm(dzs - (-dyI + hs))
# norm(ds - -(ΔsL.*dzs + hzs)./zS)
#
#
#
# mI = s.s.model.mI
# mE = s.s.model.mE
# mA = s.s.model.mA
# nL = s.s.model_opt.nL
# nU = s.s.model_opt.nU
# idx = s.s.idx
#
# H = spzeros(n+m,n+m)
# r = zeros(n+m)
#
# # view(H,1:n,1:n) .= s.s.∇²L[1:n,1:n]
# # view(H,CartesianIndex.(idx.xL[1:nL],idx.xL[1:nL])) .+= s.s.σL[1:nL]
# # view(H,CartesianIndex.(idx.xU[1:nU],idx.xU[1:nU])) .+= s.s.σU[1:nU]
# view(H,1:n,1:n) .= s.s.H_sym[1:n,1:n]
#
# H[1:n,n .+ (1:m)] .= s.s.H_sym[1:n,s.s.model.n .+ (1:m)]
# H[n .+ (1:m),1:n] .= s.s.H_sym[s.s.model.n .+ (1:m),1:n]
#
# view(H,CartesianIndex.(n .+ (1:mI),n .+ (1:mI))) .= -ΔsL./zS
# view(H,CartesianIndex.(n+mI+mE .+ (1:mA),n+mI+mE .+ (1:mA))) .= -1.0/s.s.ρ
#
# r = zeros(n+m)
# r[1:n] .= copy(hx)
# r[idx.xL[1:nL]] .+= hzL./ΔxL
# r[idx.xU[1:nU]] .-= hzU./ΔxU
# r[n .+ (1:mI)] .= hyI + (ΔsL.*hs + hzs)./zS
# r[n+mI .+ (1:mE)] .= copy(hyE)
# r[n+mI+mE .+ (1:mA)] .= hyA + 1/s.s.ρ*hr
#
# cond(Array(s.s.H))
# cond(Array(H))
# d = -H\r
# norm(s.s.d[1:n] - d[1:n])
# norm(s.s.d[idx.yI] - d[n .+ (1:mI)])
# norm(s.s.d[idx.yE] - d[n+mI .+ (1:mE)])
# norm(s.s.d[idx.yA] - d[n+mI+mE .+ (1:mA)])
#
# dx = d[1:n]
# dyI = d[n .+ (1:mI)]
# dyE = d[n+mI .+ (1:mE)]
# dyA = d[n+mI+mE .+ (1:mA)]
#
# dr = 1.0/s.s.ρ*(dyA - hr)
# dzL = -(zL.*s.s.d[s.s.idx.xL][1:s.s.model_opt.nL] + hzL)./ΔxL
# dzU = (zU.*s.s.d[s.s.idx.xU][1:s.s.model_opt.nU] - hzU)./ΔxU
# dzs = (-dyI + hs)
# ds = -(ΔsL.*dzs + hzs)./zS
#
# println("d norm: $(norm(s.s.d - [dx;ds;dr;dyI;dyE;dyA;dzL;dzs;dzU]))")
#
# norm(_s.d[1:model.n] - dx)
#
# s.s.d - [dx;ds;dr;dyI;dyE;dyA;dzL;dzs;dzU]


# rank(s.s.H)
# cond(Array(s.s.H))
# search_direction_slack!(s.s)


# ######
# using Ipopt, MathOptInterface
# const MOI = MathOptInterface
#
# struct NLPProblem <: MOI.AbstractNLPEvaluator
#     enable_hessian::Bool
# end
#
#
# function MOI.eval_objective(prob::NLPProblem, Z)
#     return f_func(Z)
# end
#
# function MOI.eval_objective_gradient(prob::NLPProblem, grad_f, Z)
#     ∇f!(grad_f,Z)
#     return nothing
# end
#
# function MOI.eval_constraint(prob::NLPProblem, g, Z)
#     c!(g,Z)
# end
#
# function constraint_bounds()
#     c_l = zeros(m)
#     c_u = zeros(m)
#     return c_l, c_u
# end
#
# function MOI.eval_constraint_jacobian(prob::NLPProblem, jac, Z)
#     ∇c!(reshape(jac,m,n),Z)
# end
#
# function row_col!(row,col,r,c)
#     for cc in c
#         for rr in r
#             push!(row,convert(Int,rr))
#             push!(col,convert(Int,cc))
#         end
#     end
#     return row, col
# end
#
# function sparsity()
#     row = []
#     col = []
#
#     r = 1:m
#     c = 1:n
#
#     row_col!(row,col,r,c)
#
#     return collect(zip(row,col))
# end
#
# # sparsity(nlp)
#
# MOI.features_available(prob::NLPProblem) = [:Grad, :Jac]
# MOI.initialize(prob::NLPProblem, features) = nothing
# MOI.jacobian_structure(prob::NLPProblem) = sparsity()
# MOI.hessian_lagrangian_structure(prob::NLPProblem) = []
# MOI.eval_hessian_lagrangian(prob::NLPProblem, H, x, σ, μ) = nothing
#
# function solve(Z0)
#
#     z_l = xL
#     z_u = xU
#     c_l, c_u = constraint_bounds()
#
#     nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
#     block_data = MOI.NLPBlockData(nlp_bounds,NLPProblem(false),true)
#
#     solver = Ipopt.Optimizer()
#     solver.options["max_iter"] = 100000
#     # solver.options["nlp_scaling_method"] = "none"
#     # solver.options["linear_system_scaling"] = "none"
#
#     Z = MOI.add_variables(solver,n)
#
#     for i = 1:n
#         zi = MOI.SingleVariable(Z[i])
#         MOI.add_constraint(solver, zi, MOI.LessThan(z_u[i]))
#         MOI.add_constraint(solver, zi, MOI.GreaterThan(z_l[i]))
#         MOI.set(solver, MOI.VariablePrimalStart(), Z[i], Z0[i])
#     end
#
#     # Solve the problem
#     MOI.set(solver, MOI.NLPBlock(), block_data)
#     MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
#     MOI.optimize!(solver)
#
#     # Get the solution
#     res = MOI.get(solver, MOI.VariablePrimal(), Z)
#
#     return res
# end
#
# sol = solve(x0)
