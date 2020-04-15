include("src/interior_point.jl")

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
nf = 2 # number of faces for friction cone pyramid
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nβ+2nc
np = nq+2nc+nβ+nc+nβ+nc
T = 20 # number of time steps to optimize

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
∇V_ng(h::Hopper,q) = [0., 0., 0., 0., 0.]

C(h::Hopper,qk,qn) = zeros(nq)

function ϕ(::Hopper,q)
    q[2] - q[3]*cos(q[5])
end

N(::Hopper,q) = ([0., 1., -cos(q[5]), 0., q[3]*sin(q[5])])'

function P(::Hopper,q)
        [1. 0. sin(q[5]) 0. q[3]*cos(q[5]);
         -1. 0. -sin(q[5]) 0. -q[3]*cos(q[5])]
end

B(::Hopper,q) = [0. 0. 0. 1. -1.;
                 0. 0. 1. 0. 0.]

model = Hopper(mb,ml,Jb,Jl,r,μ,g,p1,Δt)

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    λ = nc == 1 ? x[nq+nu+1] : x[nq+nu .+ (1:nc)]
    β = x[nq+nu+nc .+ (1:nβ)]
    ψ = nc == 1 ? x[nq+nu+nc+nβ+1] : x[nq+nu+nc+nβ .+ (1:nc)]
    η = x[nq+nu+nc+nβ+nc .+ (1:nβ)]
    sϕ = nc == 1 ? x[nq+nu+nc+nβ+nc+nβ+nc] : x[nq+nu+nc+nβ+nc+nβ .+ (1:nc)]
    sfc = nc == 1 ? x[nq+nu+nc+nβ+nc+nβ+2nc] : x[nq+nu+nc+nβ+nc+nβ+1nc .+ (1:nc)]

    return q,u,λ,β,ψ,η,sϕ,sfc
end

W = Diagonal([1e-3,1e-3,1e-3,1e-3,1e-3])
R = Diagonal([1.0e-1,1.0e-3])
Wf = Diagonal(10.0*ones(nq))
q0 = [0., r, r, 0., 0.]
qf = [3., r, r, 0., 0.]
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

qpp = Q0[1]
qp = Q0[2]

function f_func(x)
    q,u,λ,β,ψ,η,sϕ,sfc = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + rr'*u + obj_c
end

function f_func(z)
    _sum = 0.
    for t = 1:T
        q,u,λ,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])

        if t != T
            _sum += 0.5*q'*W*q + w'*q + 0.5*u'*R*u + rr'*u + obj_c
        else
            _sum += 0.5*q'*Wf*q + wf'*q + 0.5*u'*R*u + rr'*u + obj_cf
        end
    end
    return _sum
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x)
    q,u,λ,β,ψ,η,sϕ,sfc = unpack(x)
    [1/model.Δt*(M(model,qpp)*(qp - qpp) - M(model,qp)*(q - qp)) - model.Δt*∇V(model,qp) + B(model,q)'*u +  N(model,q)'*λ + P(model,q)'*β;
     sϕ - ϕ(model,q);
     λ*sϕ;
     P(model,q)*(q-qp)/Δt + ψ*ones(nβ) - η;
     sfc - (μ*λ - β'*ones(nβ));
     ψ*sfc;
     β.*η]
end

function c_func(z)
    c = zeros(eltype(z),np*T)

    for t = 1:T
        q,u,λ,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])

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

        c[(t-1)*np .+ (1:np)] .= [1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*∇V(model,_qp) + B(model,q)'*u +  N(model,q)'*λ + P(model,q)'*β;
                                 sϕ - ϕ(model,q);
                                 λ*sϕ;
                                 P(model,q)*(q-_qp)/Δt + ψ*ones(nβ) - η;
                                 sfc - (μ*λ - β'*ones(nβ));
                                 ψ*sfc;
                                 β.*η]
     end
     return c
end
c!, ∇c!, ∇²cλ! = constraint_functions(c_func)

n = T*nx
m = T*np
xL = zeros(T*nx)
xU = Inf*ones(T*nx)

for t = 1:T
    xL[(t-1)*nx .+ (1:nq+nu)] .= -Inf
    xL[(t-1)*nx + 3] = model.r/2.
    xU[(t-1)*nx + 3] = model.r
end

nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cλ!)

u0 = 1.0e-2*rand(nu)
λ0 = 1.0e-2*rand(1)[1]
β0 = 1.0e-2*rand(nβ)
ψ0 = 1.0e-2*rand(1)[1]
η0 = 1.0e-2*rand(nβ)
s0 = 1.0e-2*rand(1)[1]

x0 = zeros(T*nx)
for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];u0;λ0;β0;ψ0;η0;s0;s0]
end

s = InteriorPointSolver(x0,nlp_model,opts=Options{Float64}(kkt_solve=:symmetric,
                                                           max_iter=500,
                                                           iterative_refinement=true,
                                                           relax_bnds=false,
                                                           max_iterative_refinement=100))
@time solve!(s,verbose=true)

function get_q(z)
    Q = [qpp,qp]
    for t = 1:T
        q,u,λ,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])
        push!(Q,q)
    end
    return Q
end

q = get_q(s.s.x)