include("../src/interior_point.jl")


nc = 1
nf = 4
nq = 3
nu = 2
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nβ+5nc+nβ
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
    y = x[nq+nu+nc]
    β = x[nq+nu+nc .+ (1:nβ)]
    ψ = x[nq+nu+nc+nβ+nc]
    η = x[nq+nu+nc+nβ+nc .+ (1:nβ)]
    s = x[nq+nu+nc+nβ+nc+nβ+nc]
    sϕ = x[nq+nu+nc+nβ+nc+nβ+2nc]
    syϕ = x[nq+nu+nc+nβ+nc+nβ+3nc]
    sfc = x[nq+nu+nc+nβ+nc+nβ+4nc]
    sψfc = x[nq+nu+nc+nβ+nc+nβ+5nc]
    sβη = x[nq+nu+nc+nβ+nc+nβ+5nc .+ (1:nβ)]

    return q,u,y,β,ψ,η,s,sϕ,syϕ,sfc,sψfc,sβη
end

function f_func(x)
    q,u,y,β,ψ,η,s,sϕ,syϕ,sfc,sψfc,sβη = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c + 10.0*s
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x)
    q,u,y,β,ψ,η,s,sϕ,syϕ,sfc,sψfc,sβη = unpack(x)
    [M*(2*qp - qpp - q)/dt - G*dt + B*u + P'*β + N*y;
     sϕ - N'*q;
     syϕ - (s - y*(N'*q));
     P*(q-qp)/dt + ψ*ones(nβ) - η;
     sfc - (0.5*y - β'*ones(nβ));
     sψfc - (s - ψ*(0.5*y - β'*ones(nβ)));
     sβη - (s*ones(nβ) - β.*η)]
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

n = nx
m = np
xL = -Inf*ones(nx)
xL[(nq+nu+1):end] .= 0.
xU = Inf*ones(nx)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

q0 = 0.1*rand(nq)
u0 = 0.1*rand(nu)
y0 = 0.1*rand(1)[1]
β0 = 0.1*rand(nβ)
ψ0 = 0.1*rand(1)[1]
η0 = 0.1*rand(nβ)
s0 = 0.1*rand(1)[1]
x0 = [q0;u0;y0;β0;ψ0;η0;s0;s0;s0;s0;s0;0.1*rand(nβ)]

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        max_iter=500,
                        relax_bnds=false)

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)
norm(c_func(s.s.x),1)

s_new = InteriorPointSolver(s.s.x,model,opts=opts)
s_new.s.y .= s.s.y
s_new.s.λ .= s.s.λ + s.s.ρ*s.s.c
s_new.s.ρ = s.s.ρ*2.0
solve!(s_new,verbose=true)
s = s_new
norm(c_func(s.s.x),1)






q,u,y,β,ψ,η,_s,sϕ,syϕ,sfc,sψfc,sβη = unpack(s.s.x)

println("s: $_s")