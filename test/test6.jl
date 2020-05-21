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

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:slack))
@time solve!(s)

# _s = s.s
#
# eval_step!(_s)
#
# search_direction!(_s)
#
# _s.d
#
# H = spzeros(model.n+model.m,model.n+model.m)
# r = zeros(model.n+model.m)
#
# view(H,1:model.n,1:model.n) .= _s.H[1:model.n,1:model.n]
# view(H,CartesianIndex.(_s.idx.xL[1:model.nL],_s.idx.xL[1:model.nL])) .+= _s.σL[1:model.nL]
# view(H,CartesianIndex.(_s.idx.xU[1:model.nU],_s.idx.xU[1:model.nU])) .+= _s.σU[1:model.nU]
#
# view(H,1:model.n,model.n .+ (1:model.m)) .= _s.H[1:model.n,_s.idx.y]
# view(H,model.n .+ (1:model.m),1:model.m) .= _s.H[_s.idx.y,1:model.n]
#
# ΔxL = _s.ΔxL[1:model.nL]
# ΔsL = _s.ΔxL[model.nL .+ (1:model.mI)]
# ΔxU = _s.ΔxU
# zL = _s.zL[1:model.nL]
# zS = _s.zL[model.nL .+ (1:model.mI)]
# zU = _s.zU
# view(H,CartesianIndex.(model.n .+ (1:model.mI),model.n .+ (1:model.mI))) .= -ΔsL./zS
# view(H,CartesianIndex.(model.n+model.mI+model.mE .+ (1:model.mA),model.n+model.mI+model.mE .+ (1:model.mA))) .= -1.0/_s.ρ
#
# hx = _s.h[1:model.n]
# hs = _s.h[get_s_idx(_s)]
# hr = _s.h[get_r_idx(_s)]
# hyI = _s.h[_s.idx.y[1:model.mI]]
# hyE = _s.h[_s.idx.y[model.mI .+ (1:model.mE)]]
# hyA = _s.h[_s.idx.y[model.mI+model.mE .+ (1:model.mA)]]
# hzL = _s.h[_s.idx.zL[1:model.nL]]
# hzs = _s.h[_s.idx.zL[model.nL .+ (1:model.mI)]]
# hzr = _s.h[_s.idx.zL[model.nL+model.mI .+ (1:model.mA)]]
# hzU = _s.h[_s.idx.zU]
#
# r[1:model.n] .= copy(hx)
# r[_s.idx.xL[1:model.nL]] .+= hzL./ΔxL
# r[_s.idx.xU[1:model.nU]] .-= hzU./ΔxU
# r[model.n .+ (1:model.mI)] .= hyI + (ΔsL.*hs + hzs)./zS
# r[model.n+model.mI .+ (1:model.mE)] .= copy(hyE)
# r[model.n+model.mI+model.mE .+ (1:model.mA)] .= hyA + hr
#
# d = -H\r
#
# dx = d[1:model.n]
# dyI = d[model.n .+ (1:model.mI)]
# dyE = d[model.n+model.mI .+ (1:model.mE)]
# dyA = d[model.n+model.mI+model.mE .+ (1:model.mA)]
#
# dr = 1.0/_s.ρ*dyA - hr
# dzL = -(zL.*dx[model.xL_bool] + hzL)./ΔxL
# dzU = (zU.*dx[model.xU_bool] - hzU)./ΔxU
# dzs = -dyI + hs
# ds = -(ΔsL.*dzs + hzs)./zS
#
# norm(_s.d - [dx;ds;dr;dyI;dyE;dyA;dzL;dzs;dzU])
