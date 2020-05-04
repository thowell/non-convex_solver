using LinearAlgebra, SparseArrays

n = 100
nL = 20
nU = 30

m = 50
mI = 25
mE = 15
mA = 10

nn = n + mI + mE + mA + nL + nU + mI + mI + mA + 4m

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
    pI
    pE
    pA
    nI
    nE
    nA
    zpI
    zpE
    zpA
    znI
    znE
    znA
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
           n+m+nL+nU+mI+mI .+ (1:mA),

           n+m+nL+nU+mI+mI+mA .+ (1:mI),
           n+m+nL+nU+mI+mI+mA+mI .+ (1:mE),
           n+m+nL+nU+mI+mI+mA+mI+mE .+ (1:mA),

           n+m+nL+nU+mI+mI+mA+m .+ (1:mI),
           n+m+nL+nU+mI+mI+mA+m+mI .+ (1:mE),
           n+m+nL+nU+mI+mI+mA+m+mI+mE .+ (1:mA),

           n+m+nL+nU+mI+mI+mA+2m .+ (1:mI),
           n+m+nL+nU+mI+mI+mA+2m+mI .+ (1:mE),
           n+m+nL+nU+mI+mI+mA+2m+mI+mE .+ (1:mA),

           n+m+nL+nU+mI+mI+mA+3m .+ (1:mI),
           n+m+nL+nU+mI+mI+mA+3m+mI .+ (1:mE),
           n+m+nL+nU+mI+mI+mA+3m+mI+mE .+ (1:mA),
           )


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

p = ones(m)
_n = ones(m)
zp = ones(m)
zn = ones(m)

view(H,idx.x,idx.x) .= W
view(H,idx.x,idx.yI) .= ∇cI'
view(H,idx.x,idx.yE) .= ∇cE'
view(H,idx.x,idx.yA) .= ∇cA'
view(H,CartesianIndex.(idx.xL,idx.zL)) .= -1.0
view(H,CartesianIndex.(idx.xU,idx.zU)) .= 1.0

view(H,idx.yI,idx.x) .= ∇cI
view(H,CartesianIndex.(idx.yI,idx.s)) .= -1.0
view(H,CartesianIndex.(idx.yI,idx.pI)) .= -1.0
view(H,CartesianIndex.(idx.yI,idx.nI)) .= 1.0


view(H,idx.yE,idx.x) .= ∇cE
view(H,CartesianIndex.(idx.yE,idx.pE)) .= -1.0
view(H,CartesianIndex.(idx.yE,idx.nE)) .= 1.0

view(H,idx.yA,idx.x) .= ∇cA
view(H,CartesianIndex.(idx.yA,idx.r)) .= -1.0
view(H,CartesianIndex.(idx.yA,idx.pA)) .= -1.0
view(H,CartesianIndex.(idx.yA,idx.nA)) .= 1.0

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

view(H,CartesianIndex.(idx.pI,idx.yI)) .= -1.0
view(H,CartesianIndex.(idx.pI,idx.zpI)) .= -1.0

view(H,CartesianIndex.(idx.pE,idx.yE)) .= -1.0
view(H,CartesianIndex.(idx.pE,idx.zpE)) .= -1.0

view(H,CartesianIndex.(idx.pA,idx.yA)) .= -1.0
view(H,CartesianIndex.(idx.pA,idx.zpA)) .= -1.0

view(H,CartesianIndex.(idx.nI,idx.yI)) .= 1.0
view(H,CartesianIndex.(idx.nI,idx.znI)) .= -1.0

view(H,CartesianIndex.(idx.nE,idx.yE)) .= 1.0
view(H,CartesianIndex.(idx.nE,idx.znE)) .= -1.0

view(H,CartesianIndex.(idx.nA,idx.yA)) .= 1.0
view(H,CartesianIndex.(idx.nA,idx.znA)) .= -1.0

view(H,CartesianIndex.(idx.zpI,idx.pI)) .= zp[1:mI]
view(H,CartesianIndex.(idx.zpI,idx.zpI)) .= p[1:mI]

view(H,CartesianIndex.(idx.zpE,idx.pE)) .= zp[mI .+ (1:mE)]
view(H,CartesianIndex.(idx.zpE,idx.zpE)) .= p[mI .+ (1:mE)]

view(H,CartesianIndex.(idx.zpA,idx.pA)) .= zp[mI+mE .+ (1:mA)]
view(H,CartesianIndex.(idx.zpA,idx.zpA)) .= p[mI+mE .+ (1:mA)]

view(H,CartesianIndex.(idx.znI,idx.nI)) .= zn[1:mI]
view(H,CartesianIndex.(idx.znI,idx.znI)) .= _n[1:mI]

view(H,CartesianIndex.(idx.znE,idx.nE)) .= zn[mI .+ (1:mE)]
view(H,CartesianIndex.(idx.znE,idx.znE)) .= _n[mI .+ (1:mE)]

view(H,CartesianIndex.(idx.znA,idx.nA)) .= zn[mI+mE .+ (1:mA)]
view(H,CartesianIndex.(idx.znA,idx.znA)) .= _n[mI+mE .+ (1:mA)]


view(Hsym,idx.x,idx.x) .= W
view(Hsym,CartesianIndex.(idx.xL,idx.xL)) .+= zL./xL
view(Hsym,CartesianIndex.(idx.xU,idx.xU)) .+= zU./xU
view(Hsym,idx.x,idx.yI) .= ∇cI'
view(Hsym,idx.x,idx.yE) .= ∇cE'
view(Hsym,idx.x,idx.yA) .= ∇cA'

view(Hsym,idx.yI,idx.x) .= ∇cI
view(Hsym,CartesianIndex.(idx.yI,idx.yI)) .= -s./zs - p[1:mI]./zp[1:mI] - _n[1:mI]./zn[1:mI]

view(Hsym,idx.yE,idx.x) .= ∇cE
view(Hsym,CartesianIndex.(idx.yE,idx.yE)) .= -p[mI .+ (1:mE)]./zp[mI .+ (1:mE)] - _n[mI .+ (1:mE)]./zn[mI .+ (1:mE)]

view(Hsym,idx.yA,idx.x) .= ∇cA
view(Hsym,CartesianIndex.(idx.yA,idx.yA)) .= -p[mI+mE .+ (1:mA)]./zp[mI+mE .+ (1:mA)] - _n[mI+mE .+ (1:mA)]./zn[mI+mE .+ (1:mA)]
view(Hsym,CartesianIndex.(idx.yA,idx.yA)) .+= -1.0

rsym[idx.x] .= r[idx.x]
rsym[idx.xL] .+= r[idx.zL]./xL
rsym[idx.xU] .-= r[idx.zU]./xU

rsym[idx.yI] .= view(r,idx.yI) + (s.*view(r,idx.s) + view(r,idx.zs))./zs + (p[1:mI].*view(r,idx.pI) + view(r,idx.zpI))./zp[1:mI] - (_n[1:mI].*view(r,idx.nI) + view(r,idx.znI))./zn[1:mI]
rsym[idx.yE] .= view(r,idx.yE) + (p[mI .+ (1:mE)].*view(r,idx.pE) + view(r,idx.zpE))./zp[mI .+ (1:mE)] - (_n[mI .+ (1:mE)].*view(r,idx.nE) + view(r,idx.znE))./zn[mI .+ (1:mE)]
rsym[idx.yA] .= view(r,idx.yA) + view(r,idx.r) + (p[mI+mE .+ (1:mA)].*view(r,idx.pA) + view(r,idx.zpA))./zp[mI+mE .+ (1:mA)] - (_n[mI+mE .+ (1:mA)].*view(r,idx.nA) + view(r,idx.znA))./zn[mI+mE .+ (1:mA)]

rank(H)
d = H\(-r)
#
dsym = Hsym\(-rsym)

norm(d[1:n]-dsym[1:n])


norm(d[idx.yA] - r[idx.r] -d[idx.r])

norm(-zL./xL.*d[idx.xL] - r[idx.zL]./xL - d[idx.zL])

norm(zU./xU.*d[idx.xU] - r[idx.zU]./xU - d[idx.zU])

norm(-d[idx.yI] + r[idx.s] - d[idx.zs])

norm(-s./zs.*d[idx.zs] - r[idx.zs]./zs - d[idx.s])

norm(-d[idx.yI] + r[idx.pI] - d[idx.zpI])
norm(-d[idx.yE] + r[idx.pE] - d[idx.zpE])
norm(-d[idx.yA] + r[idx.pA] - d[idx.zpA])

norm(d[idx.yI] + r[idx.nI] - d[idx.znI])
norm(d[idx.yE] + r[idx.nE] - d[idx.znE])
norm(d[idx.yA] + r[idx.nA] - d[idx.znA])

norm((-p[1:mI].*d[idx.zpI] - r[idx.zpI])./zp[1:mI] - d[idx.pI])
norm((-p[mI .+ (1:mE)].*d[idx.zpE] - r[idx.zpE])./zp[mI .+ (1:mE)] - d[idx.pE])
norm((-p[mI+mE .+ (1:mA)].*d[idx.zpA] - r[idx.zpA])./zp[mI+mE .+ (1:mA)] - d[idx.pA])

norm((-_n[1:mI].*d[idx.znI] - r[idx.znI])./zn[1:mI] - d[idx.nI])
norm((-_n[mI .+ (1:mE)].*d[idx.znE] - r[idx.znE])./zn[mI .+ (1:mE)] - d[idx.nE])
norm((-_n[mI+mE .+ (1:mA)].*d[idx.znA] - r[idx.znA])./zn[mI+mE .+ (1:mA)] - d[idx.nA])

norm(-(_n[1:mI].*d[idx.yI] + _n[1:mI].*r[idx.nI] + r[idx.znI])./zn[1:mI] -d[idx.nI])
norm(-(-p[1:mI].*d[idx.yI] + p[1:mI].*r[idx.pI] + r[idx.zpI])./zp[1:mI] -d[idx.pI])
