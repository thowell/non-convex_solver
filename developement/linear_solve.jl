using LinearAlgebra, ForwardDiff, StaticArrays, SparseArrays, BenchmarkTools, HSL, SuiteSparse

T = 10
n = 6
m = 3
Δt = 0.1
A = @SMatrix [1. 0. 0. Δt 0. 0.;
              0. 1. 0. 0. Δt 0.;
              0. 0. 1. 0. 0. Δt;
              0. 0. 0. 1. 0. 0.;
              0. 0. 0. 0. 1. 0.;
              0. 0. 0. 0. 0. 1.]

B = [0. 0. 0.;
     0. 0. 0.;
     0. 0. 0.;
     1. 0. 0.;
     0. 1. 0.;
     0. 0. 1.]

Q = Diagonal(@SVector [0.1,0.1,0.1,0.1,0.1,0.1])
Qf = Diagonal(@SVector [1.,1.,1.,1.,1.,1.])
R = Diagonal(@SVector [0.01,0.01,0.01])

x0 = @SVector zeros(n)
xf = @SVector [1.,1.,1.,0.,0.,0.]

function obj(z)
    s = 0
    for t = 1:T-1
        x = view(z,(t-1)*(n+m) .+ (1:n))
        u = view(z,(t-1)*(n+m) + n .+ (1:m))
        s += (x-xf)'*Q*(x-xf) + u'*R*u
    end
    x = view(z,(T-1)*(n+m) .+ (1:n))
    s += (x-xf)'*Qf*(x-xf)
    return s
end

function ∇obj(z)
    ∇f = zero(z)
    for t = 1:T-1
        for t = 1:T-1
            x = view(z,(t-1)*(n+m) .+ (1:n))
            u = view(z,(t-1)*(n+m) + n .+ (1:m))
            view(∇f,(t-1)*(n+m) .+ (1:n)) .= 2*Q*(x-xf)
            view(∇f,(t-1)*(n+m) + n .+ (1:m)) .= 2*R*u
        end
        x = view(z,(T-1)*(n+m) .+ (1:n))
        view(∇f,(T-1)*(n+m) .+ (1:n)) .= 2*Qf*(x-xf)
    end
    return ∇f
end

function ∇²obj(z)
    ∇²f = zeros(n*T+m*(T-1),n*T+m*(T-1))
    for t = 1:T-1
        for t = 1:T-1
            view(∇²f,(t-1)*(n+m) .+ (1:n),(t-1)*(n+m) .+ (1:n)) .= 2*Q
            view(∇²f,(t-1)*(n+m) + n .+ (1:m),(t-1)*(n+m) + n .+ (1:m)) .= 2*R
        end
        view(∇²f,(T-1)*(n+m) .+ (1:n),(T-1)*(n+m) .+ (1:n)) .= 2*Qf
    end
    return ∇²f
end

function con(z)
    c = zeros(eltype(z),n*T)

    view(c,(1:n)) .= view(z,1:n) - x0

    for t = 1:T-1
        x = view(z,(t-1)*(n+m) .+ (1:n))
        x⁺ = view(z,t*(n+m) .+ (1:n))
        u = view(z,(t-1)*(n+m) + n .+ (1:m))
        view(c,t*n .+ (1:n)) .= A*x + B*u - x⁺
    end

    return c
end

function ∇con(z)
    ∇c = zeros(eltype(z),n*T,n*T+m*(T-1))

    view(∇c,CartesianIndex.(1:n,1:n)) .= 1.0

    for t = 1:T-1
        x = view(z,(t-1)*(n+m) .+ (1:n))
        x⁺ = view(z,t*(n+m) .+ (1:n))
        u = view(z,(t-1)*(n+m) + n .+ (1:m))
        view(∇c,t*n .+ (1:n),(t-1)*(n+m) .+ (1:n)) .= A
        view(∇c,t*n .+ (1:n),(t-1)*(n+m) + n .+ (1:m)) .= B
        view(∇c,CartesianIndex.(t*n .+ (1:n),t*(n+m) .+ (1:n))) .= -1.0
    end

    return ∇c
end

n̄ = n*T + m*(T-1)
m̄ = n*T
z = rand(n̄)
y = rand(m̄)
f = obj(z)
∇f = ∇obj(z)
∇²f = ∇²obj(z)
c = con(z)
∇c = ∇con(z)
∇²cy = zeros(n̄,n̄)

H = spzeros(n̄+m̄,n̄+m̄)
h = zeros(n̄+m̄)

view(H,1:n̄,1:n̄) .= ∇²f + ∇²cy
view(H,1:n̄,n̄ .+ (1:m̄)) .= ∇c'
view(H,n̄ .+ (1:m̄),1:n̄) .= ∇c

view(h,1:n̄) .= ∇f + ∇c'*y
view(h,n̄ .+ (1:m̄)) .= c

d = zeros(n̄+m̄)

function s1!(d::Vector{T},H::SparseMatrixCSC{T,Int},h::Vector{T}) where T
    d .= H\h
end
function s2!(d::Vector{T},H::SparseMatrixCSC{T,Int},h::Vector{T}) where T
    d .= Symmetric(H)\h
end
function s3!(d::Vector{T},H::SparseMatrixCSC{T,Int},h::Vector{T}) where T
    d .= lu(H)\h
end
@benchmark s1!($d,$H,$h)
@benchmark s2!($d,$H,$h)
@benchmark s3!($d,$H,$h)


function schur!(dx::Vector{T},dy::Vector{T},
    A11::SparseMatrixCSC{T,Int},A12::SparseMatrixCSC{T,Int},
    A21::SparseMatrixCSC{T,Int},A22::SparseMatrixCSC{T,Int},
    b1::Vector{T},b2::Vector{T},
    tmp1::SparseMatrixCSC{T,Int},tmp2::Vector{T},
    S::SparseMatrixCSC{T,Int},
    b̃::Vector{T}) where T

    tmp1 .= A11\A12
    tmp2 .= A11\b1
    S .= A22 - A21*tmp1
    b̃ .= b2 - A21*tmp2
    dy .= S\b̃
    dx .= tmp2 - tmp1*dy
end
@benchmark cholesky($∇²f)
dx = zeros(n̄)
dy = zeros(m̄)
A11 = sparse(∇²f)
A12 = sparse(∇c')
A21 = sparse(∇c)
A22 = spzeros(m̄,m̄)
b1 = h[1:n̄]
b2 = h[n̄ .+ (1:m̄)]
S = spzeros(m̄,m̄)
b̃ = zeros(m̄)
tmp1 = spzeros(n̄,m̄)
tmp2 = zeros(n̄)
# schur!(dx,dy,A11,A12,A21,A22,b1,b2,tmp1,tmp2,S,b̃)
A11\A12
typeof(A11)
@benchmark schur!($dx,$dy,$A11,$A12,$A21,$A22,$b1,$b2,$tmp1,$tmp2,$S,$b̃)

norm([dx;dy] - H\h)

LBL = Ma57(H)

function solve_HSL!(d::Vector{T},LBL::Ma57{T},h::Vector{T}) where T
    ma57_factorize(LBL)
    d .= ma57_solve(LBL,h)
    return nothing
end

@benchmark solve_HSL!($d,$LBL,$h)
