include("../src/interior_point.jl")

# Dimensions
nq = 3 # configuration dim
nu = 0 # control dim
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone pyramid
nβ = nc*nf

nx = nq+nu+nc+nβ+nc+nc+nc
np = nq+nβ+4nc
T = 10 # number of time steps to optimize
Δt = 0.1
# Parameters
M(q) = 1.0*Matrix(I,nq,nq)
B(q) = [1. 0. 0.;0. 1. 0.]
P(q) = [1. 0. 0.;0. 1. 0.]
G(q) = [0; 0; 9.8]
N(q) = [0 0 1]
ϕ(q) = q[3]

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = nc == 1 ? x[nq+nu+nc] : x[nq+nu .+ (1:nc)]
    β = nβ == 1 ? x[nq+nu+nc+nβ] : x[nq+nu+nc .+ (1:nβ)]
    ψ = nc == 1 ? x[nq+nu+nc+nβ+nc] : x[nq+nu+nc+nβ .+ (1:nc)]
    sϕ = nc == 1 ? x[nq+nu+nc+nβ+nc+nc] : x[nq+nu+nc+nβ+nc .+ (1:nc)]
    sfc = nc == 1 ? x[nq+nu+nc+nβ+nc+nc+nc] : x[nq+nu+nc+nβ+nc+nc .+ (1:nc)]

    return q,u,y,β,ψ,sϕ,sfc
end

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
q0 = [0, 0., 1.0e-8]
qf = [5, 1, 0]
Q0 = linear_interp(q0,qf,T+2)

qpp = Q0[1]
qp = Q0[2]

function f_func(z)
    _sum = 0.
    return _sum
end
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(z)
    c = zeros(eltype(z),np*T)

    for t = 1:T
        q,u,y,β,ψ,sϕ,sfc = unpack(z[(t-1)*nx .+ (1:nx)])

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

        c[(t-1)*np .+ (1:np)] .= [vec(1/Δt*(M(_qpp)*(_qp - _qpp) - M(_qp)*(q - _qp)) - Δt*G(_qp) +  N(q)'*y + P(q)'*β);
                                  (P(q)*(q-_qp)/Δt + 2.0*β*ψ);
                                  (sϕ - ϕ(q));
                                  (sfc - ((0.5*y)^2 - β'*β));
                                  y*sϕ;
                                  sfc*ψ]
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
    xL[(t-1)*nx+nq+nu+nc .+ (1:nβ)] .= -Inf
end

nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

c_al_idx_t = ones(Bool,np)
c_al_idx_t[1:nq+nβ+nc+nc] .= 0
c_al_idx = ones(Bool,nlp_model.m)

for t = 1:T
    c_al_idx[(t-1)*np .+ (1:np)] .= c_al_idx_t
end

y0 = 1.0*ones(1)[1]
β0 = 1.0*ones(nβ)
ψ0 = 1.0*ones(1)[1]
s0 = 1.0*ones(1)[1]

x0 = zeros(T*nx)
for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];y0;β0;ψ0;ϕ(Q0[t+2]);(0.5*y0)^2 - β0'*β0]
end

opts = Options{Float64}(kkt_solve=:symmetric,
                       max_iter=500,
                       iterative_refinement=true,
                       relax_bnds=true,
                       max_iterative_refinement=100,
                       ϵ_tol=1.0e-5,
                       verbose=true)

s = InteriorPointSolver(x0,nlp_model,c_al_idx=c_al_idx,opts=opts)

@time solve!(s)
norm(c_func(s.s.x)[c_al_idx .== 0],1)
norm(c_func(s.s.x)[c_al_idx],1)

Q = [qpp,qp]
for t = 1:T
    q,u,y,β,ψ,sϕ,sfc = unpack(s.s.x[(t-1)*nx .+ (1:nx)])
    push!(Q,q)
end
Q[end]

Q_nl = Q

vis = Visualizer()
open(vis)
visualize!(vis,Q)
using Colors # Handle RGB colors
using CoordinateTransformations # Translations and rotations
using FileIO # Save and load files
using GeometryTypes:Vec,HyperRectangle,HyperSphere,Point3f0,Cylinder # Define geometric shape
using LinearAlgebra
using MeshCat # Visualize 3D animations
using MeshIO # Load meshes in MeshCat

function visualize!(vis,q; r=0.25)

    setobject!(vis["particle"], HyperRectangle(Vec(0,0,0),Vec(2r,2r,2r)))

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim,t) do
            settransform!(vis["particle"], Translation(q[t][1:3]...))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
