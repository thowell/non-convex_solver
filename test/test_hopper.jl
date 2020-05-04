include("../src/interior_point.jl")


mutable struct Hopper{T,S} <: AbstractModel
    mb::T
    ml::T
    Jb::T
    Jl::T
    r::T
    μ::T
    g::T
    k::S
    Δt::T
end

# Dimensions
nq = 5 # configuration dim
nu = 2 # control dim
nc = 1 # number of contact points
nf = 1 # number of faces for friction cone pyramid
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nβ+2nc
np = nq+nβ+5nc
T = 3 # number of time steps to optimize

# Parameters
g = 9.81 # gravity
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
mb = 10. # body mass
ml = 1.  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

# Kinematics
r = 0.7
p1(q) = [q[1] + q[3]*sin(q[5]), q[2] - q[3]*cos(q[5])]

# Methods
M(h::Hopper,q) = Diagonal([h.mb+h.ml, h.mb+h.ml, h.ml, h.Jb, h.Jl])
∇V(h::Hopper,q) = [0., (h.mb+h.ml)*h.g, 0., 0., 0.]

C(h::Hopper,qk,qn) = zeros(nq)

function ϕ(::Hopper,q)
    q[2] - q[3]*cos(q[5])
end

N(::Hopper,q) = ([0., 1., -cos(q[5]), 0., q[3]*sin(q[5])])'

function P(::Hopper,q)
        ([1., 0., sin(q[5]), 0., q[3]*cos(q[5])])'
end

B(::Hopper,q) = [0. 0. 0. 1. -1.;
                 0. 0. 1. 0. 0.]

model = Hopper(mb,ml,Jb,Jl,r,μ,g,p1,Δt)

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = nc == 1 ? x[nq+nu+nc] : x[nq+nu .+ (1:nc)]
    β = nβ == 1 ? x[nq+nu+nc+nβ] : x[nq+nu+nc .+ (1:nβ)]
    ψ = nc == 1 ? x[nq+nu+nc+nβ+nc] : x[nq+nu+nc+nβ .+ (1:nc)]
    η = nβ == 1 ? x[nq+nu+nc+nβ+nc+nc] : x[nq+nu+nc+nβ+nc .+ (1:nc)]
    sϕ = nc == 1 ? x[nq+nu+nc+nβ+nc+nc+nc] : x[nq+nu+nc+nβ+nc+nc .+ (1:nc)]
    sfc = nc == 1 ? x[nq+nu+nc+nβ+nc+nc+2nc] : x[nq+nu+nc+nβ+nc+nc+1nc .+ (1:nc)]

    return q,u,y,β,ψ,η,sϕ,sfc
end

W = Diagonal([1e-3,1e-3,1e-3,1e-3,1e-3])
R = Diagonal([1.0e-1,1.0e-3])
Wf = Diagonal(5.0*ones(nq))
q0 = [0., r, r, 0., 0.]
qf = [1., r, r, 0., 0.]
uf = zeros(nu)
w = -W*qf
wf = -Wf*qf
rr = -R*uf
obj_c = 0.5*(qf'*W*qf + uf'*R*uf)
obj_cf = 0.5*(qf'*Wf*qf + uf'*R*uf)

function linear_interp(x0,xf,T)
    n = length(x0)
    X = [copy(Array(x0)) for t = 1:T]

    for t = 1:T
        for i = 1:n
            X[t][i] = (xf[i]-x0[i])/(T-1)*(t-1) + x0[i]
        end
    end

    return X
end

Q0 = linear_interp(q0,qf,T+2)

qpp = Q0[2]
qp = Q0[2]

function f_func(z)
    _sum = 0.
    for t = 1:T
        q,u,y,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])

        if t != T
            _sum += 0.5*q'*W*q + w'*q + 0.5*u'*R*u + rr'*u + obj_c
        else
            _sum += 0.5*q'*Wf*q + wf'*q + 0.5*u'*R*u + rr'*u + obj_cf
        end
    end
    return _sum
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(z)
    c = zeros(eltype(z),np*T)

    for t = 1:T
        q,u,y,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])

        if t == 1
            _qpp = qpp
            _qp = qp
        elseif t == 2
            _qpp = qp
            _qp = z[(t-2)*nx .+ (1:nq)]
        else
            _qpp = z[(t-3)*nx .+ (1:nq)]
            _qp = z[(t-2)*nx .+ (1:nq)]
        end

        c[(t-1)*np .+ (1:np)] .= [(1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*∇V(model,_qp) + B(model,q)'*u +  N(model,q)'*y + P(model,q)'*β);
                                  (P(model,q)*(q-_qp)/model.Δt + 2.0*β*ψ);
                                  (sϕ - ϕ(model,q));
                                  (sfc - ((model.μ*y)^2 - β'*β));
                                  (ψ - η);
                                  y*sϕ;
                                  sfc*η]
     end
     return c
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

n = T*nx
m = T*np
xL = zeros(T*nx)
xU = Inf*ones(T*nx)

for t = 1:T
    xL[(t-1)*nx .+ (1:nq+nu)] .= -Inf
    xL[(t-1)*nx + 3] = model.r/2.
    xU[(t-1)*nx + 3] = model.r

    xL[(t-1)*nx+nq+nu+nc .+ (1:nβ)] .= -Inf
    xL[(t-1)*nx+nq+nu+nc+nβ .+ (1:nc)] .= -Inf
end

nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

cA_idx_t = ones(Bool,np)
cA_idx_t[1:nq+nβ+nc+nc+nc] .= 0
cA_idx = ones(Bool,nlp_model.m)

for t = 1:T
    cA_idx[(t-1)*np .+ (1:np)] .= cA_idx_t
end

u0 = 1.0e-3*rand(nu)
y0 = 1.0e-3*rand(1)[1]
β0 = 1.0e-3*rand(nβ)[1]
ψ0 = 1.0e-3*rand(1)[1]
η0 = 1.0e-3*rand(nc)[1]
s0 = 1.0e-3*rand(1)[1]

x0 = zeros(T*nx)
for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];u0;y0;β0;ψ0;η0;ϕ(model,Q0[t+2]);(model.μ*y0)^2 - β0'*β0]
end

opts = Options{Float64}(kkt_solve=:symmetric,
                       max_iter=500,
                       iterative_refinement=true,
                       relax_bnds=true,
                       max_iterative_refinement=100,
                       ϵ_tol=1.0e-8,
                       verbose=false)

s = InteriorPointSolver(x0,nlp_model,cA_idx=cA_idx,opts=opts)

@time solve!(s)
norm(c_func(s.s.x)[cA_idx .== 0],1)
norm(c_func(s.s.x)[cA_idx],1)


function get_q(z)
    Q = [qpp,qp]
    for t = 1:T
        q,u,y,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])
        push!(Q,q)
    end
    return Q
end

q = get_q(s.s.x)

max_dis(qp,q,β,ψ) = P(model,q)*(q-qp)/model.Δt + 2.0*β[1]*ψ[1]

function max_dis(z)
    max_dis(z[1:nq],z[nq .+ (1:nq)],z[2nq .+ (1:nβ)],z[2nq+nβ .+ (1:nc)])
end

max_dis(qpp,qp,β0,ψ0)
max_dis([qpp;qp;β0;ψ0])

y = rand(nc)[1]
function max_dis_y(z)
    max_dis(z[1:nq],z[nq .+ (1:nq)],z[2nq .+ (1:nβ)],z[2nq+nβ .+ (1:nc)])*y
end

max_dis_y([qpp;qp;β0;ψ0])

ForwardDiff.hessian(max_dis_y,[qpp;qp;β0;ψ0])

q1_idx = 1:nq
q2_idx = nx .+ (1:nq)
b1_idx = nx + nq + nu + nc .+ (1:nβ)
ψ1_idx = nx + nq + nu + nc + nβ .+ (1:nc)

idx = [q1_idx...,q2_idx...,b1_idx...,ψ1_idx...]

y = rand(m)
nlp_model.∇²cy_func!(nlp_model.∇²cy,x0,y,nlp_model)


function max_dis_y(z)
    max_dis(z[1:nq],z[nq .+ (1:nq)],z[2nq .+ (1:nβ)],z[2nq+nβ .+ (1:nc)])*y[np+nq+nβ]
end

max_dis_y([Q0[3];Q0[4];β0;ψ0])

ForwardDiff.hessian(max_dis_y,[qpp;Q0[3];β0;ψ0])

function dyn(z)
    _qpp = z[1:nq]
    _qp = z[nq .+ (1:nq)]
    q = z[2nq .+ (1:nq)]
    u = z[3nq .+ (1:nu)]
    λ = z[3nq+nu .+ (1:nc)]
    β = z[3nq+nu+nc .+ (1:nβ)]
    (1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*∇V(model,_qp) + B(model,q)'*u +  N(model,q)'*λ[1] + P(model,q)'*β[1])
end

function dyn_y(z)
    dyn(z)'*y[1:nq]
end

dyn_y([Q0[1];Q0[2];Q0[3];u0;y0;β0])

ForwardDiff.hessian(dyn_y,[Q0[1];Q0[2];Q0[3];u0;y0;β0])[2nq+1:end,2nq+1:end]

idx = [(1:nq)...,(nq .+ (1:nu))...,(nq+nu .+ (1:nc))...,(nq+nu+nc .+ (1:nβ))...]


nlp_model.∇²cy .= 0.




Array(nlp_model.∇²cy[idx,idx])
