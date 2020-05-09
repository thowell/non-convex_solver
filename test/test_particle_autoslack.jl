include("../src/interior_point.jl")


nc = 1
nf = 2
nq = 3
nu = 2
nβ = nc*nf

nx = nq+nu+nc+nβ+nc
np = nq+nβ+4nc

dt = 0.1

M(q) = 1.0*Matrix(I,nq,nq)
B(q) = [1. 0. 0.;0. 1. 0.]
P(q) = [1. 0. 0.;0. 1. 0.]

G(q) = [0; 0; 9.8]

N(q) = [0; 0; 1]

qpp = [0.,0.,10.]
v0 = [10.,-7.0, 0.]
v1 = v0 - G(qpp)*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G(qp)*dt
q1 = qp + 0.5*dt*(v1 + v2)

qf = [0.; 0.; 0.]
uf = [0.; 0.]

W = 10.0*Matrix(I,nq,nq)
w = -W*qf
R = 1.0e-1*Matrix(I,nu,nu)
r = -R*uf
obj_c = 0.5*qf'*W*qf + 0.5*uf'*R*uf

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = x[nq+nu+nc]
    β = x[nq+nu+nc .+ (1:nβ)]
    ψ = x[nq+nu+nc+nβ+nc]

    return q,u,y,β,ψ
end

function f_func(x)
    q,u,y,β,ψ = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x)
    q,u,y,β,ψ = unpack(x)
    [(N(q)'*q);
     (((0.5*y)^2 - β'*β));
     (M(q)*(2*qp - qpp - q)/dt - G(q)*dt + B(q)'*u + P(q)'*β + N(q)*y);
     (P(q)*(q-qp)/dt + 2.0*β*ψ);
     y*(N(q)'*q);
     ψ*((0.5*y)^2 - β'*β)]
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

n = nx
m = np
xL = zeros(nx)
xL[1:(nq+nu)] .= -Inf
xL[nq+nu+nc .+ (1:nβ)] .= -Inf
xU = Inf*ones(nx)
nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

cI_idx = zeros(Bool,nlp_model.m)
cI_idx[1:nc+nc] .= 1

cA_idx = ones(Bool,nlp_model.m)
cA_idx[1:nq+nβ+nc+nc] .= 0
q0 = q1
u0 = 1.0e-3*rand(nu)
y0 = 1.0*rand(1)[1]
β0 = 1.0*rand(nβ)
ψ0 = 1.0*rand(1)[1]
x0 = [q0;u0;y0;β0;ψ0]

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        max_iter=500,
                        relax_bnds=true,
                        y_init_ls=true,
                        ϵ_tol=1.0e-6,
                        verbose=false)

s = InteriorPointSolver(x0,nlp_model,cI_idx=cI_idx,cA_idx=cA_idx,opts=opts)
@time solve!(s)
# norm(c_func(s.s.x)[cA_idx .== 0],1)
# norm(c_func(s.s.x)[cA_idx],1)
# view(s.s.H,s.s.idx.x,s.s.idx.x)[end-1:end,end-1:end]
#
# s.s.∇c[m-s.s.mA+1:m,n+mI .+ (1:s.s.mA)]
# s.s.mA
#
#
#
#
# eval_step!(s.s)
# search_direction_fullspace!(s.s)
# s.s.d
#
#
#
#
#
#
#
#
#
#
#
#
# search_direction_symmetric!(s.s)
#
# H = spzeros(n+m,n+m)
# h = zeros(n+m)
# mI = s.s.mI
# mA = s.s.mA
# mE = s.s.mE
# nL = s.s.nL - mI
# nU = s.s.nU
# r = s.s.h
# rx = r[1:n]
# rs = r[n .+ (1:mI)]
# rr = r[n+mI .+ (1:mA)]
# ry = r[s.s.idx.y]
# ryI = ry[1:mI]
# ryE = ry[mI .+ (1:mE)]
# ryA = ry[mI+mE .+ (1:mA)]
# _rzL = r[s.s.idx.zL]
# rzL = _rzL[1:nL]
# rzS = _rzL[nL .+ (1:mI)]
# rzU = r[s.s.idx.zU]
#
# idx_yI = (n .+ (1:mI))
# idx_yE = (n+mI .+ (1:mE))
# idx_yA = (n .+ (1:m))[end-mA+1:end]
# H[1:n,1:n] = (s.s.∇²L + s.s.∇²cy)[1:n,1:n]
# view(H,CartesianIndex.(s.s.idx.xL[1:nL],s.s.idx.xL[1:nL])) .+= s.s.σL[1:nL]
# view(H,CartesianIndex.(s.s.idx.xU[1:nU],s.s.idx.xU[1:nU])) .+= s.s.σU[1:nU]
# H[1:n,n .+ (1:m)] .= s.s.∇c[1:m,1:n]'
# H[n .+ (1:m),1:n] .= s.s.∇c[1:m,1:n]
# view(H,CartesianIndex.(idx_yI,idx_yI)) .= -1.0./s.s.σL[nL .+ (1:mI)]
# view(H,CartesianIndex.(idx_yA,idx_yA)) .= -1.0/s.s.ρ
# n+m
# Array(H)
# h[1:n] = rx
# h[s.s.idx.xL[1:nL]] += rzL./s.s.ΔxL[1:nL]
# h[s.s.idx.xU[1:nU]] -= rzU./s.s.ΔxU[1:nU]
# h[idx_yI] = ryI + (s.s.ΔxL[nL .+ (1:mI)].*rs + rzS)./s.s.zL[nL .+ (1:mI)]
# h[idx_yE] = ryE
# h[idx_yI] = ryA + rr
#
# -H\h
#
#
#
#
#
#
#
#
#
#
#
#
# s.s.d
