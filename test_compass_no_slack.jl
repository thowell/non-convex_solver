include("src/interior_point.jl")

mutable struct Compass{T} <: AbstractModel
    m1::T
    m2::T
    m3::T
    J1::T
    J2::T
    J3::T
    r::T
    μ::T
    k1::Function
    k2::Function
    Δt::T
end

# Dimensions
nq = 5 # configuration dim
nu = 2 # control dim
nc = 2 # number of contact points
nf = 2 # number of faces for friction cone pyramid
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nβ+2nc
np = nq+2nc+nβ+nc+nβ+nc
T = 20 # number of time steps to optimize

# Parameters
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
m1 = 10.   # mass
m2 = 1.  # mass
m3 = 1.  # mass

J1 = 2.5  # inertia
J2 = 0.25 # inertia
J3 = 0.25   # inertia

# Kinematics
r = 1.0
p1(q) = [q[1] + r*sin(q[4]), q[2] - r*cos(q[4])]
p2(q) = [q[1] + r*sin(q[5]), q[2] - r*cos(q[5])]

# Methods
M(c::Compass,q) = [(c.m1+c.m2+c.m3) 0. 0. 0.5*c.m2*c.r*cos(q[4]) 0.5*c.m3*c.r*cos(q[5]);
                 0. (c.m1+c.m2+c.m3) 0. 0.5*c.m2*c.r*sin(q[4]) 0.5*c.m3*c.r*sin(q[5]);
                 0. 0. (c.J1+c.J2+c.J3) c.J2 c.J3;
                 0.5*c.m2*c.r*cos(q[4]) 0.5*c.m2*c.r*sin(q[4]) c.J2 (0.25*c.m2*c.r^2 + c.J2) 0.;
                 0.5*c.m3*c.r*cos(q[5]) 0.5*c.m3*c.r*sin(q[5]) c.J3 0. (0.25*c.m3*c.r^2 + c.J3)]

∇M(c::Compass,q) = [zeros(3,5);
         zeros(1,3) -0.5*c.m2*c.r*sin(q[4]) 0.;
         zeros(1,4) -0.5*c.m3*c.r*sin(q[5]);
         zeros(3,5);
         zeros(1,3) 0.5*c.m2*c.r*cos(q[4]) 0.;
         zeros(1,4) 0.5*c.m3*c.r*cos(q[5]);
         zeros(5,5);
         zeros(1,3) -0.5*c.m2*c.r*sin(q[4]) 0.;
         zeros(1,3) 0.5*c.m2*c.r*cos(q[4]) 0.
         zeros(3,5);
         zeros(1,4) -0.5*c.m2*c.r*sin(q[5]);
         zeros(1,4) 0.5*c.m3*c.r*cos(q[5]);
         zeros(3,5)
        ]

function ∇V(c::Compass,q)
    [0., (c.m1+c.m2+c.m3)*9.8, 0., 0.5*c.m2*c.r*sin(q[4]), 0.5*c.m2*c.r*sin(q[5])]
end

function C(c::Compass,qk,qn)
    v1 = (qn-qk)/Δt

    a = -0.5*c.m2*c.r*sin(qk[4])*v1[1]*v1[4] + 0.5*c.m2*c.r*cos(qk[4])*v1[2]*v1[4] + -0.5*c.m2*c.r*sin(qk[4])*v1[4]*v1[1] + 0.5*c.m2*c.r*cos(qk[4])*v1[4]*v1[2]
    b = -0.5*c.m3*c.r*sin(qk[5])*v1[1]*v1[5] + 0.5*c.m3*c.r*cos(qk[5])*v1[2]*v1[5] + -0.5*c.m2*c.r*sin(qk[5])*v1[5]*v1[1] + 0.5*c.m3*c.r*cos(qk[5])*v1[5]*v1[2]
    [0., 0., 0., a, b]
end

ϕ(c::Compass,q) = [q[2] - c.r*cos(q[4]), q[2] - c.r*cos(q[5])]
N(c::Compass,q) = [0. 1. 0. c.r*sin(q[4]) 0.;
                 0. 1. 0. 0. c.r*sin(q[5])]

p(q) = [q[1] + r*sin(q[4]), -1.0*(q[1] + r*sin(q[4])), q[1] + r*sin(q[5]), -1.0*(q[1] + r*sin(q[5]))]
P(c::Compass,q) = [1. 0. 0. c.r*cos(q[4]) 0.;
                 -1. 0. 0. -c.r*cos(q[4]) 0.;
                 1. 0. 0. 0. c.r*cos(q[5]);
                 -1. 0. 0. 0. -c.r*cos(q[5])]
B(::Compass,q) = [0. 0. 0. 1. 0.; 0. 0. 0. 0. 1.]

model = Compass(m1,m2,m3,J1,J2,J3,r,μ,p1,p2,Δt)

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = x[nq+nu .+ (1:nc)]
    β = x[nq+nu+nc .+ (1:nβ)]
    ψ = x[nq+nu+nc+nβ .+ (1:nc)]
    η = x[nq+nu+nc+nβ+nc .+ (1:nβ)]
    sϕ = x[nq+nu+nc+nβ+nc+nβ .+ (1:nc)]
    sfc = x[nq+nu+nc+nβ+nc+nβ+1nc .+ (1:nc)]

    return q,u,y,β,ψ,η,sϕ,sfc
end

W = Diagonal(1.0*ones(nq))
R = Diagonal(1.0e-1*ones(nu))
Wf = Diagonal(10.0*ones(nq))

theta = pi/12
h = 0.5*sqrt(4*model.r^2 - (2*model.r*sin(theta))^2)
q0 = [0., h, 0., -theta, theta]
qf = [2., h, 0., -theta, theta]
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

        c[(t-1)*np .+ (1:np)] .= [1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*∇V(model,_qp) + 0.5*model.Δt*C(model,_qp,q) + B(model,q)'*u +  N(model,q)'*y + P(model,q)'*β;
                                  P(model,q)*(q-_qp)/model.Δt + [ψ[1]*ones(nf);ψ[2]*ones(nf)] - η;
                                  sϕ - ϕ(model,q);
                                  sfc[1] - (model.μ*y[1] - β[1:nf]'*ones(nf));
                                  sfc[2] - (model.μ*y[2] - β[nf .+ (1:nf)]'*ones(nf))
                                  y.*sϕ;
                                  ψ.*sfc;
                                  β.*η]
     end
     return c
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

n = T*nx
m = T*np
xL = zeros(T*nx)
for t = 1:T
    xL[(t-1)*nx .+ (1:nq+nu)] .= -Inf
end
xU = Inf*ones(T*nx)

nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

c_al_idx_t = ones(Bool,np)
c_al_idx_t[1:nq+nβ+nc+nc] .= 0
c_al_idx = ones(Bool,nlp_model.m)

for t = 1:T
    c_al_idx[(t-1)*np .+ (1:np)] .= c_al_idx_t
end

u0 = 1.0e-3*rand(nu)
y0 = 1.0e-3*rand(nc)
β0 = 1.0e-3*rand(nβ)
ψ0 = 1.0e-3*rand(nc)
η0 = 1.0e-3*rand(nβ)
s0 = 1.0e-3*rand(nc)

x0 = zeros(T*nx)
for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];u0;y0;β0;ψ0;η0;ϕ(model,Q0[t+2]);model.μ*y0[1] - β0[1:nf]'*ones(nf);model.μ*y0[2] - β0[nf .+ (1:nf)]'*ones(nf)]
end

opts = Options{Float64}(kkt_solve=:symmetric,
                       max_iter=500,
                       iterative_refinement=true,
                       relax_bnds=false,
                       max_iterative_refinement=100,
                       ϵ_tol=1.0e-6)

s = InteriorPointSolver(x0,nlp_model,c_al_idx=c_al_idx,opts=opts)

@time solve!(s)
norm(c_func(s.s.x)[c_al_idx .== 0],1)
norm(c_func(s.s.x)[c_al_idx],1)

# s_new = InteriorPointSolver(s.s.x,nlp_model,c_al_idx=c_al_idx,opts=opts)
# s_new.s.y .= s.s.y
# s_new.s.λ .= s.s.λ + s.s.ρ*s.s.c[c_al_idx]
# s_new.s.ρ = s.s.ρ*10.0
# solve!(s_new,verbose=true)
# s = s_new
# norm(c_func(s.s.x)[c_al_idx .== 0],1)
# norm(c_func(s.s.x)[c_al_idx],1)
#
function get_q(z)
    Q = [qpp,qp]
    for t = 1:T
        q,u,y,β,ψ,η,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])
        push!(Q,q)
    end
    return Q
end

q = get_q(s.s.x)
