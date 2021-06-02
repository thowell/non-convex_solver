using LinearAlgebra, SparseArrays

n = 100
nL = 20
nU = 30

m = 50
mI = 25
mE = 15
mA = 10

nn = n + mI + mE + mA + nL + nU + mI + mI + mA

H = spzeros(nn,nn)
r = rand(nn)

Hsym = spzeros(n+m,n+m)
rsym = zeros(n+m)

struct Inds
    x
    xL
    xU
    yI
    yE
    yA
    zL
    zU
    s
    zs
    r
end

1:convert(Int,floor(10/3))
idx = Inds(1:n,
           1:nL,
           1:nU,
           n .+ (1:mI),
           n+mI .+ (1:mE),
           n+mI+mE .+ (1:mA),
           n+m .+ (1:nL),
           n+m+nL .+ (1:nU),
           n+m+nL+nU .+ (1:mI),
           n+m+nL+nU+mI .+ (1:mI),
           n+m+nL+nU+mI+mI .+ (1:mA))


W = Diagonal(ones(n))
∇cI = zeros(mI,n)
∇cI[CartesianIndex.(1:mI,1:mI)] .= 1.0
∇cE = zeros(mE,n)
∇cE[CartesianIndex.(1:mE,mI .+ (1:mE))] .= 1.0
∇cA = zeros(mA,n)
∇cA[CartesianIndex.(1:mA,mI+mE .+ (1:mA))] .= 1.0


xL = ones(nL)
xU = ones(nU)
s = ones(mI)

zL = ones(nL)
zU = ones(nU)
zs = ones(mI)

view(H,idx.x,idx.x) .= W
view(H,idx.x,idx.yI) .= ∇cI'
view(H,idx.x,idx.yE) .= ∇cE'
view(H,idx.x,idx.yA) .= ∇cA'
view(H,CartesianIndex.(idx.xL,idx.zL)) .= -1.0
view(H,CartesianIndex.(idx.xU,idx.zU)) .= 1.0

view(H,idx.yI,idx.x) .= ∇cI
view(H,CartesianIndex.(idx.yI,idx.s)) .= -1.0

view(H,idx.yE,idx.x) .= ∇cE

view(H,idx.yA,idx.x) .= ∇cA
view(H,CartesianIndex.(idx.yA,idx.r)) .= -1.0

view(H,CartesianIndex.(idx.zL,idx.xL)) .= zL
view(H,CartesianIndex.(idx.zL,idx.zL)) .= xL

view(H,CartesianIndex.(idx.zU,idx.xU)) .= -zU
view(H,CartesianIndex.(idx.zU,idx.zU)) .= xU

view(H,CartesianIndex.(idx.s,idx.yI)) .= -1.0
view(H,CartesianIndex.(idx.s,idx.zs)) .= -1.0

view(H,CartesianIndex.(idx.zs,idx.s)) .= zs
view(H,CartesianIndex.(idx.zs,idx.zs)) .= s

view(H,CartesianIndex.(idx.r,idx.yA)) .= -1.0
view(H,CartesianIndex.(idx.r,idx.r)) .= 1.0

view(Hsym,idx.x,idx.x) .= W
view(Hsym,CartesianIndex.(idx.xL,idx.xL)) .+= zL./xL
view(Hsym,CartesianIndex.(idx.xU,idx.xU)) .+= zU./xU
view(Hsym,idx.x,idx.yI) .= ∇cI'
view(Hsym,idx.x,idx.yE) .= ∇cE'
view(Hsym,idx.x,idx.yA) .= ∇cA'

view(Hsym,idx.yI,idx.x) .= ∇cI
view(Hsym,CartesianIndex.(idx.yI,idx.yI)) .= -s./zs

view(Hsym,idx.yE,idx.x) .= ∇cE

view(Hsym,idx.yA,idx.x) .= ∇cA
view(Hsym,CartesianIndex.(idx.yA,idx.yA)) .= -1.0

rsym[idx.x] .= r[idx.x]
rsym[idx.xL] .+= r[idx.zL]./xL
rsym[idx.xU] .-= r[idx.zU]./xU

rsym[idx.yI] .= view(r,idx.yI) + (s.*view(r,idx.s) + view(r,idx.zs))./zs

rsym[idx.yE] .= view(r,idx.yE)

rsym[idx.yA] .= view(r,idx.yA) + view(r,idx.r)

rank(H)
d = H\(-r)

dsym = Hsym\(-rsym)

norm(d[1:n+m]-dsym)


norm(d[idx.yA] - r[idx.r] -d[idx.r])

norm(-zL./xL.*d[idx.xL] - r[idx.zL]./xL - d[idx.zL])

norm(zU./xU.*d[idx.xU] - r[idx.zU]./xU - d[idx.zU])

norm(-d[idx.yI] + r[idx.s] - d[idx.zs])

norm(-s./zs.*d[idx.zs] - r[idx.zs]./zs - d[idx.s])
