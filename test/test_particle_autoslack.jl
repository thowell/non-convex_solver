include("../src/non-convex_solver.jl")


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

cI_idx = zeros(Bool,m)
cI_idx[1:nc+nc] .= 1

cA_idx = zeros(Bool,m)
cA_idx[2nc+nq+nf*nc .+ (1:2nc)] .= 1
# cA_idx = ones(Bool,m)
# cA_idx[1:2nc] .= 0.0


nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)

q0 = q1
u0 = 1.0e-5*ones(nu)
y0 = 1.0e-5*ones(1)[1]
β0 = 1.0e-5*ones(nβ)
ψ0 = 1.0e-5*ones(1)[1]
x0 = [q0;u0;y0;β0;ψ0]

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        max_iter=500,
                        relax_bnds=true,
                        y_init_ls=true,
                        ϵ_tol=1.0e-8,
                        ϵ_al_tol=1.0e-8,
                        quasi_newton=:lbfgs,
                        quasi_newton_approx=:lagrangian,
                        verbose=true)

s = NonConvexSolver(x0,nlp_model,opts=opts)
@time solve!(s)
