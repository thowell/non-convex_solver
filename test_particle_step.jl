include("src/interior_point.jl")

nc = 1
nf = 4
nq = 3
nu = 2
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nβ+2nc
np = nq+2nc+nβ+nc+nβ+nc

dt = 0.1

M = 1.0*Matrix(I,nq,nq)
B = [1. 0.;0. 1.;0. 0.]
G = [0; 0; 9.8]

P = [1. 0. 0.;
     0. 1. 0.;
     -1. 0. 0.;
     0. -1. 0.]

N = [0; 0; 1]

qpp = [0., 0., .15]
v0 = [5., 0., 0.]
v1 = v0 - G*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G*dt
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
    λ = x[nq+nu+nc]
    β = x[nq+nu+nc .+ (1:nβ)]
    ψ = x[nq+nu+nc+nβ+nc]
    η = x[nq+nu+nc+nβ+nc .+ (1:nβ)]
    sϕ = x[nq+nu+nc+nβ+nc+nβ+nc]
    sfc = x[nq+nu+nc+nβ+nc+nβ+2nc]

    return q,u,λ,β,ψ,η,sϕ,sfc
end

function f_func(x)
    q,u,λ,β,ψ,η,sϕ,sfc = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x,ρ)
    q,u,λ,β,ψ,η,sϕ,sfc = unpack(x)
    [M*(2*qp - qpp - q)/dt - G*dt + B*u + P'*β + N*λ;
     sϕ - N'*q;
     λ*(N'*q) - ρ;
     P*(q-qp)/dt + ψ*ones(nβ) - η;
     sfc - (0.5*λ - β'*ones(nβ));
     ψ*(0.5*λ - β'*ones(nβ)) - ρ;
     β.*η .- ρ]
end
c!, ∇c!, ∇²cλ! = constraint_functions(c_func)

n = nx
m = np
xL = -Inf*ones(nx)
xL[(nq+nu+1):end] .= 0.
xU = Inf*ones(nx)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cλ!)

q0 = 1.0e-2*rand(nq)
u0 = 1.0e-2*rand(nu)
λ0 = 1.0e-2*rand(1)[1]
β0 = 1.0e-2*rand(nβ)
ψ0 = 1.0e-2*rand(1)[1]
η0 = 1.0e-2*rand(nβ)
s0 = 1.0e-2*rand(1)[1]
x0 = [q0;u0;λ0;β0;ψ0;η0;s0;s0]

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,iterative_refinement=true,max_iter=500,relax_bnds=true))
@time solve!(s,verbose=true)

q,u,λ,β,ψ,η,sϕ,sfc = unpack(s.s.x)
