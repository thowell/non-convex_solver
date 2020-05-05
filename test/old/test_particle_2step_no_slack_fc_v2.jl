include("../src/interior_point.jl")


T = 2
nc = 1
nf = 2
nq = 3
nu = 2
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nc+2nc
np = nq+nβ+5nc

dt = 0.1

M(q) = 1.0*Matrix(I,nq,nq)
B(q) = [1. 0. 0.;0. 1. 0.]
P(q) = [1. 0. 0.;0. 1. 0.]

G(q) = [0; 0; 9.8]

# P(q) = [1. 0. 0.;
#      0. 1. 0.;
#      -1. 0. 0.;
#      0. -1. 0.]

N(q) = [0; 0; 1]

qpp = [0.,0.,10.]
v0 = [10.,-7.0, 0.]
v1 = v0 - G(qpp)*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G(qp)*dt
q1 = qp + 0.5*dt*(v1 + v2)

qf = [0.; 0.; 0.]
uf = [0.; 0.]

W = 1.0*Matrix(I,nq,nq)
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
    η = x[nq+nu+nc+nβ+nc+nc]
    sϕ = x[nq+nu+nc+nβ+nc+nc+nc]
    sfc = x[nq+nu+nc+nβ+nc+nc+2nc]

    return q,u,y,β,ψ,η,sϕ,sfc
end

function f_func(x)
    q1,u1,y1,β1,ψ1,η1,sϕ1,sfc1 = unpack(x[1:nx])
    q2,u2,y2,β2,ψ2,η2,sϕ2,sfc2 = unpack(x[nx .+ (1:nx)])

    return 0.5*q1'*W*q1 + w'*q1 + 0.5*u1'*R*u1 + r'*u1 + obj_c + 0.5*q2'*W*q2 + w'*q2 + 0.5*u2'*R*u2 + r'*u2 + obj_c
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x)
    q1,u1,y1,β1,ψ1,η1,sϕ1,sfc1 = unpack(x[1:nx])
    q2,u2,y2,β2,ψ2,η2,sϕ2,sfc2 = unpack(x[nx .+ (1:nx)])
    [(M(qp)*(2*qp - qpp - q1)/dt - G(qp)*dt + B(q1)'*u1 + P(q1)'*β1 + N(q1)*y1);
     (P(q1)*(q1-qp)/dt + 2.0*β1*ψ1);
     (sϕ1 - N(q1)'*q1);
     (sfc1 - ((0.5*y1)^2 - β1'*β1));
     (ψ1 - η1);
     y1*sϕ1;
     sfc1*η1;

     (M(q1)*(2*q1 - qp - q2)/dt - G(q1)*dt + B(q2)'*u2 + P(q2)'*β2 + N(q2)*y2);
      (P(q2)*(q2-q1)/dt + 2.0*β2*ψ2);
      (sϕ2 - N(q2)'*q2);
      (sfc2 - ((0.5*y2)^2 - β2'*β2));
      (ψ2 - η2);
      y2*sϕ2;
      sfc2*η2]
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

n = nx*T
m = np*T
xL = zeros(n)
xL[1:(nq+nu)] .= -Inf
xL[nq+nu+nc .+ (1:nβ)] .= -Inf
xL[nq+nu+nc+nβ+nc] = -Inf

xL[nx .+ (1:(nq+nu))] .= -Inf
xL[nx+nq+nu+nc .+ (1:nβ)] .= -Inf
xL[nx+nq+nu+nc+nβ+nc] = -Inf
xU = Inf*ones(n)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

cA_idx = ones(Bool,m)
cA_idx[1:nq+nβ+nc+nc+nc] .= 0
cA_idx[np .+ (1:nq+nβ+nc+nc+nc)] .= 0

q0 = q1
u0 = 1.0e-3*rand(nu)
y0 = 1.0e-2*rand(1)[1]
β0 = 1.0e-2*rand(nβ)
ψ0 = 1.0e-2*rand(1)[1]
η0 = 1.0e-2*rand(1)[1]
x0 = [q0;u0;y0;β0;ψ0;η0; N(q0)'*q0;(0.5*y0)^2 - β0'*β0;q0;u0;y0;β0;ψ0;η0; N(q0)'*q0;(0.5*y0)^2 - β0'*β0]

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        max_iter=500,
                        relax_bnds=false,
                        y_init_ls=true,
                        ϵ_tol=1.0e-6,
                        ϵ_al_tol=1.0e-5)

s = InteriorPointSolver(x0,model,cA_idx=cA_idx,opts=opts)
@time solve!(s)
norm(c_func(s.s.x)[cA_idx .== 0],1)
norm(c_func(s.s.x)[cA_idx],1)
