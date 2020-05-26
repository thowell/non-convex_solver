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
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m))

opts = Options{Float64}(kkt_solve=:slack,
                        iterative_refinement=true,
                        ϵ_tol=1.0e-8,
                        max_iterative_refinement=10,
                        verbose=true
                        )

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)
# eval_step!(s.s)
# s.s.h
# initialize_restoration_solver!(s.s̄,s.s)
# eval_step!(s.s̄)
# search_direction_fullspace!(s.s̄)
# s.s̄.d
#
#
# n = s.s.model_opt.n
# nL = s.s.model_opt.nL
# nL_slack = s.s.model.nL
# nU = s.s.model_opt.nU
# m = s.s.model_opt.m
# n_slack = s.s.model.n
# n_rest = s.s̄.model.n
# mI = s.s.model_opt.mI
# mE = s.s.model_opt.mE
# mA = s.s.model_opt.mA
# nL_slack
# nL + mI
# xL = s.s̄.ΔxL[1:nL]
# sL = s.s̄.ΔxL[nL .+ (1:mI)]
# pI = s.s̄.ΔxL[nL_slack .+ (1:mI)]
# pE = s.s̄.ΔxL[nL_slack+mI .+ (1:mE)]
# pA = s.s̄.ΔxL[nL_slack+mI+mE .+ (1:mA)]
# nI = s.s̄.ΔxL[nL_slack+m .+ (1:mI)]
# nE = s.s̄.ΔxL[nL_slack+m+mI .+ (1:mE)]
# nA = s.s̄.ΔxL[nL_slack+m+mI+mE .+ (1:mA)]
# xU = s.s̄.ΔxU[1:nU]
# zL = s.s̄.zL[1:nL]
# zs = s.s̄.zL[nL .+ (1:mI)]
# zpI = s.s̄.zL[nL_slack .+ (1:mI)]
# zpE = s.s̄.zL[nL_slack+mI .+ (1:mE)]
# zpA = s.s̄.zL[nL_slack+mI+mE .+ (1:mA)]
# znI = s.s̄.zL[nL_slack+m .+ (1:mI)]
# znE = s.s̄.zL[nL_slack+m+mI .+ (1:mE)]
# znA = s.s̄.zL[nL_slack+m+mI+mE .+ (1:mA)]
# zU = s.s̄.zU[1:nU]
#
# hx = s.s̄.h[1:n]
# hs = s.s̄.h[n .+ (1:mI)]
# hr = s.s̄.h[n+mI .+ (1:mA)]
# hpI = s.s̄.h[n_slack .+ (1:mI)]
# hpE = s.s̄.h[n_slack + mI .+ (1:mE)]
# hpA = s.s̄.h[n_slack + mI + mE .+ (1:mA)]
# hnI = s.s̄.h[n_slack+m .+ (1:mI)]
# hnE = s.s̄.h[n_slack+m+mI .+ (1:mE)]
# hnA = s.s̄.h[n_slack+m+mI+mE .+ (1:mA)]
# hyI = s.s̄.h[n_rest .+ (1:mI)]
# hyE = s.s̄.h[n_rest+mI .+ (1:mE)]
# hyA = s.s̄.h[n_rest+mI+mE .+ (1:mA)]
# hzL = s.s̄.h[n_rest+m .+ (1:nL)]
# hzs = s.s̄.h[n_rest+m+nL .+ (1:mI)]
# hzpI = s.s̄.h[n_rest+m+nL_slack .+ (1:mI)]
# hzpE = s.s̄.h[n_rest+m+nL_slack+mI .+ (1:mE)]
# hzpA = s.s̄.h[n_rest+m+nL_slack+mI+mE .+ (1:mA)]
# hznI = s.s̄.h[n_rest+m+nL_slack+m .+ (1:mI)]
# hznE = s.s̄.h[n_rest+m+nL_slack+m+mI .+ (1:mE)]
# hznA = s.s̄.h[n_rest+m+nL_slack+m+mI+mE .+ (1:mA)]
# hzU = s.s̄.h[n_rest+m+nL_slack+2m.+ (1:nU)]
#
# # dzL
# norm(s.s̄.d[n_rest+m .+ (1:nL)] - -(hzL + zL.*s.s̄.d[s.s̄.idx.xL][1:nL])./xL)
#
# # dzs
# norm(s.s̄.d[n_rest+m+nL .+ (1:mI)] - -(hzs + zs.*s.s̄.d[s.s̄.idx.xL][nL .+ (1:mI)])./sL)
#
# # dpI
# norm(s.s̄.d[n_slack .+ (1:mI)] - -(hpI + pI.*s.s̄.d[s.s̄.idx.zL][nL+mI .+ (1:mI)])./zpI)
# #dpE
# norm(s.s̄.d[n_slack+mI .+ (1:mE)] - -(hpE + pE.*s.s̄.d[s.s̄.idx.zL][nL+mI+mI .+ (1:mE)])./zpE)
#
# #dpA
# norm(s.s̄.d[n_slack+mI+mE .+ (1:mA)] - -(hpA + pA.*s.s̄.d[n_rest+m+nL_slack+mI+mE .+ (1:mA)])./zpA)
#
# s.s̄.d[n_slack+mI .+ (1:mE)]
# (hpE + pE.*s.s̄.d[s.s̄.idx.zL][nL+mI .+ (1:m)])
#
#
# # dzU
# norm(s.s̄.d[s.s̄.idx.zU] - -(hzU - zU.*s.s̄.d[s.s̄.idx.xU])./xU)
#
# # ds
# norm(s.s̄.d[s.s̄.idx.s] - s.s̄.H[s.s̄.idx.s,s.s̄.idx.s]\(-hs + s.s̄.d[s.s̄.idx.yI] + s.s̄.d[s.s̄.idx.zL][nL .+ (1:mI)]))
#
# # dr
