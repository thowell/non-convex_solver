""" Cartpole example
    -control limits
    -initial condition
    -goal constraint
"""

mutable struct Cartpole{T}
    mc::T # mass of the cart in kg (10)
    mp::T # mass of the pole (point mass at the end) in kg
    l::T  # length of the pole in m
    g::T  # gravity m/s^2
    μ::T  # friction coefficient
    nx
    nu
end

function dynamics(model::Cartpole, x, u)
    H = [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = [0.0 -model.mp*x[4]*model.l*sin(x[2]); 0.0 0.0]
    G = [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = [1.0, 0.0]
    qdd = -H\(C*view(x,3:4) + G - B*u[1])

    return [x[3];x[4];qdd[1];qdd[2]]
end

function discrete_dynamics(model,x⁺,x,u,Δt)
    x⁺ - (x + Δt*dynamics(model,0.5*(x + x⁺),u))
end

nq,nu = 4,1
model = Cartpole(1.0,0.2,0.5,9.81,0.1,nq,nu)
ul = -10.0
uu = 10.0

T = 20 # number of time steps to optimize

dt = 0.1

nx = nq + nu # dimension of each stage
np = nq # number of general stage constraints

# tracking objective
W = Diagonal(ones(nq))
R = Diagonal([1.0e-1])
Wf = Diagonal(zeros(nq))
q0 = [0., 0., 0., 0.]
qf = [0.0, π, 0., 0.]
uf = zeros(nu)
w = -W*qf
wf = -Wf*qf
rr = -R*uf
obj_c = 0.5*(qf'*W*qf + uf'*R*uf)
obj_cf = 0.5*(qf'*Wf*qf)

# initial guess
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

Q0 = linear_interp(q0,qf,T)

# objective functions
function f_func(z)
    _sum = 0.
    for t = 1:T-1
        q = z[(t-1)*nx .+ (1:nq)]
        u = z[(t-1)*nx+nq .+ (1:nu)]

        _sum += 0.5*q'*W*q + w'*q + 0.5*u'*R*u + rr'*u + obj_c
    end

    q = z[(T-1)*nx .+ (1:nq)]
    _sum += 0.5*q'*Wf*q + wf'*q + obj_cf
    return _sum
end

f, ∇f!, ∇²f! = objective_functions(f_func)

# constraint functions
function c_func(z)
    c = zeros(eltype(z),np*(T-1) + 2*nq)

    for t = 1:T-1
        q = z[(t-1)*nx .+ (1:nq)]
        u = z[(t-1)*nx + nq .+ (1:nu)]
        q⁺ = z[t*nx .+ (1:nq)]
        c[(t-1)*np .+ (1:np)] = discrete_dynamics(model,q⁺,q,u,dt)
    end
    c[(T-1)*np .+ (1:nq)] = z[(1:nq)] - q0 # initial condition
    c[(T-1)*np + nq.+ (1:nq)] = z[(T-1)*nx .+ (1:nq)] - qf # goal constraint

    return c
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

# problem dimensions
n = T*nx
m = (T-1)*np + 2*nq

# primal boumds
xL = -Inf*ones(T*nx)
xU = Inf*ones(T*nx)

for t = 1:T
    xL[(t-1)*nx + nq .+ (1:nu)] .= ul
    xU[(t-1)*nx + nq .+ (1:nu)] .= uu
end

# cI_idx_t = zeros(Bool,np)
# cI_idx = zeros(Bool,m)
# for t = 1:T
#     cI_idx[(t-1)*np .+ (1:np)] .= cI_idx_t
# end

# cA_idx_t = zeros(Bool,np)
# cA_idx = zeros(Bool,m)
#
# for t = 1:T
#     cA_idx[(t-1)*np .+ (1:np)] .= cA_idx_t
# end

nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)#,cI_idx=cI_idx)#,cA_idx=cA_idx)

u0 = 1.0e-3*ones(nu)
x0 = zeros(T*nx)

for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t]; u0]
end

# standard solve using MA57 and second derivatives of constraints
# opts = Options{Float64}(kkt_solve=:symmetric,
#                        max_iter=250,
#                        iterative_refinement=true,
#                        relax_bnds=true,
#                        max_iterative_refinement=10,
#                        linear_solver=:QDLDL,
#                        ϵ_tol=1.0e-5,
#                        ϵ_al_tol=1.0e-5,
#                        verbose=true)


# QDLDL needs some help (almost works) with regularization
opts = Options{Float64}(kkt_solve=:symmetric,
                      linear_solver=:QDLDL,
                      max_iter=250,
                      iterative_refinement=true,
                      relax_bnds=true,
                      max_iterative_refinement=10,
                      ϵ_tol=1.0e-6,
                      ϵ_al_tol=1.0e-6,
                      verbose=true)#,
                      # quasi_newton=:none,
                      # quasi_newton_approx=:lagrangian,
                      # lbfgs_length=6,
                      # )

# L-BFGS usually works could be tuned a bit
# opts = Options{Float64}(kkt_solve=:symmetric,
#                       max_iter=500,
#                       iterative_refinement=true,
#                       relax_bnds=true,
#                       max_iterative_refinement=10,
#                       ϵ_tol=1.0e-6,
#                       ϵ_al_tol=1.0e-6,
#                       verbose=true,#,
#                       quasi_newton=:lbfgs,
#                       quasi_newton_approx=:lagrangian,
#                       lbfgs_length=6,
#                       )

# BFGS is typically more reliable than L-BFGS
# opts = Options{Float64}(kkt_solve=:symmetric,
#                     max_iter=500,
#                     iterative_refinement=true,
#                     relax_bnds=true,
#                     max_iterative_refinement=10,
#                     ϵ_tol=1.0e-6,
#                     ϵ_al_tol=1.0e-6,
#                     verbose=true,#,
#                     quasi_newton=:bfgs,
#                     )

s = NCSolver(x0,nlp_model,opts=opts)
@time solve!(s)

q_sol = [s.s.x[(t-1)*nx .+ (1:nq)] for t = 1:T]

using Plots
plot(hcat(q_sol...)[1:nq,:]',xlabel="time step",ylabel="state",title="Cartpole")
