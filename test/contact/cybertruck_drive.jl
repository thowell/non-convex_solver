include("../src/non-convex_solver.jl")
using StaticArrays

mutable struct Cybertruck{T} <: AbstractModel
    m::T
    J::T
    μ::T
    g::T
    Δt::T
end

# Dimensions
nq = 4 # configuration dim
nu = 2 # control dim
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone pyramid
nβ = nc*nf

nx = nq+nu+nc+nβ+nc
np = nq+nβ+4nc+nu

T = 20 # number of time steps to optimize

# Parameters
g = 9.81 # gravity
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
mass = 1.0 # body mass
J = 1.0 # body inertia

# Methods
M(c::Cybertruck,q) = Diagonal([c.m, c.m, c.m, c.J])
G(c::Cybertruck,q) = [0.0, 0.0, c.m*c.g, 0.0]
C(c::Cybertruck,qk,qn) = zeros(nq)
jump_slope=0.3
jump_length=1.0
function ϕ(::Cybertruck,q)
    if q[1] < jump_length
        return q[3] - jump_slope*q[1]
    else
        return q[3]
    end
end
N(::Cybertruck,q) = ([0.0, 0.0, 1.0, 0.0])'
P(::Cybertruck,q) = [1.0 0.0 0.0 0.0;
                     0.0 1.0 0.0 0.0]

B(::Cybertruck,q) = [cos(q[4]) sin(q[4]) 0.0 0.0;
                     0.0 0.0 0.0 1.0]

model = Cybertruck(mass,J,μ,g,Δt)

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = nc == 1 ? x[nq+nu+nc] : x[nq+nu .+ (1:nc)]
    β = nβ == 1 ? x[nq+nu+nc+nβ] : x[nq+nu+nc .+ (1:nβ)]
    ψ = nc == 1 ? x[nq+nu+nc+nβ+nc] : x[nq+nu+nc+nβ .+ (1:nc)]

    return q,u,y,β,ψ
end

W = Diagonal([1.0, 1.0, 1.0, 1.0])
R = Diagonal([1.0e-1,1.0e-1])
Wf = Diagonal([10.0, 10.0, 10.0, 10.0])
q0 = [-1.0, 0.0, 0.0, 0.0]
qf = [3.0, 0.0, 0.0, 0.0]
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
        q,u,y,β,ψ = unpack(z[(t-1)*nx .+ (1:nx)])

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
        q,u,y,β,ψ = unpack(z[(t-1)*nx .+ (1:nx)])

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

        c[(t-1)*np .+ (1:np)] .= [ϕ(model,q);
                                  ((model.μ*y)^2 - β'*β);
                                  (1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*G(model,_qp) + B(model,q)'*u +  N(model,q)'*y + P(model,q)'*β);
                                  (P(model,q)*(q-_qp)/model.Δt + 2.0*β*ψ);
                                  ϕ(model,q)*y;
                                  ((model.μ*y)^2 - β'*β)*ψ;
                                  ϕ(model,q)*u[1];
                                  ϕ(model,q)*u[2]]
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

cI_idx_t = zeros(Bool,np)
cI_idx_t[1:nc+nc] .= 1
cI_idx = zeros(Bool,m)

for t = 1:T
    cI_idx[(t-1)*np .+ (1:np)] .= cI_idx_t
end

cA_idx_t = zeros(Bool,np)
cA_idx_t[nc+nc+nq+nβ .+ (1:2nc+nu)] .= 1
cA_idx = zeros(Bool,m)

for t = 1:T
    cA_idx[(t-1)*np .+ (1:np)] .= cA_idx_t
end

nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)

u0 = 1.0e-8*ones(nu)
y0 = 1.0e-8*ones(1)[1]
β0 = 1.0e-8*ones(nβ)
ψ0 = 1.0e-8*ones(1)[1]

x0 = zeros(T*nx)
for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];u0;y0;β0;ψ0]
end

opts = Options{Float64}(kkt_solve=:symmetric,
                       max_iter=1000,
                       iterative_refinement=true,
                       relax_bnds=true,
                       max_iterative_refinement=10,
                       ϵ_tol=1.0e-3,
                       ϵ_al_tol=1.0e-3,
                       verbose=true,
                       quasi_newton=:none,
                       quasi_newton_approx=:lagrangian,
                       lbfgs_length=6)

s = NCSolver(x0,nlp_model,opts=opts)
@time solve!(s)

x_sol = get_solution(s)
Q = []
push!(Q,Q0[1])
push!(Q,Q0[2])
for t = 1:T
    q,u,y,β,ψ = unpack(x_sol[(t-1)*nx .+ (1:nx)])
    push!(Q,q)
end

using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshIO
using Rotations

function visualize!(vis,p::Cybertruck,q; r=0.25)

    obj_path = "/home/taylor/Downloads/cybertruck.obj"
    mtl_path = "/home/taylor/Downloads/cybertruck.mtl"

    ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.1)
    setobject!(vis["cybertruck"],ctm)
    settransform!(vis["cybertruck"], LinearMap(RotZ(pi)*RotX(pi/2.0)))

    anim = MeshCat.Animation(convert(Int,floor(1/p.Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim,t) do
            settransform!(vis["cybertruck"], compose(Translation((q[t][1:3] + ((q[t][1] < jump_length+1.0 && q[t][1] > -0.1) ? [0.0;0.0;0.01] : zeros(3))) ...),LinearMap(RotZ(q[t][4]+pi)*RotY((q[t][1] < jump_length+0.0 && q[t][1] > -0.25) ? tan(jump_slope) : 0.0)*RotX(pi/2.0))))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end

function ModifiedMeshFileObject(obj_path::String, material_path::String; scale::T=0.1) where {T}
    obj = MeshFileObject(obj_path)
    rescaled_contents = rescale_contents(obj_path, scale=scale)
    material = select_material(material_path)
    mod_obj = MeshFileObject(
        rescaled_contents,
        obj.format,
        material,
        obj.resources,
        )
    return mod_obj
end

function rescale_contents(obj_path::String; scale::T=0.1) where T
    lines = readlines(obj_path)
    rescaled_lines = copy(lines)
    for (k,line) in enumerate(lines)
        if length(line) >= 2
            if line[1] == 'v'
                stringvec = split(line, " ")
                vals = map(x->parse(Float64,x),stringvec[2:end])
                rescaled_vals = vals .* scale
                rescaled_lines[k] = join([stringvec[1]; string.(rescaled_vals)], " ")
            end
        end
    end
    rescaled_contents = join(rescaled_lines, "\r\n")
    return rescaled_contents
end

function select_material(material_path::String)
    mtl_file = open(material_path)
    mtl = read(mtl_file, String)
    return mtl
end

vis = Visualizer()
open(vis)
visualize!(vis,model,Q)


Ns = 100
h = range(0,stop=jump_slope*jump_length,length=Ns)
w = range(0,stop=jump_length,length=Ns)
wid = jump_length/Ns

for i = 1:Ns
    setobject!(vis["stair$i"], HyperRectangle(Vec(0., 0.0, 0.0), Vec(0.01, 1.0, h[i])))
    settransform!(vis["stair$i"], Translation(w[i], -0.5, 0))
end
