include("src/interior_point.jl")

n = 3
m = 2
x0 = [-1.0; 3.; 1.]
x0 = [-2.0; 3.0; 1.0]
# x0 = [0.;0.;0.]

xl = -Inf*ones(n)
xl[2] = 0.
xl[3] = 0.
xu = Inf*ones(n)

f_func(x) = x[1]
c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}(max_iter=1000))
sr = RestorationSolver_l1(s)

search_direction!(sr)
sr.d
p
x = sr.x[1:s.n]
xr = s.x
p = sr.x[s.n .+ (1:s.m)]
n = sr.x[(s.n+s.m) .+ (1:s.m)]
zl = sr.zl[1:s.nl]
zp = sr.zl[s.nl .+ (1:s.m)]
zn = sr.zl[(s.nl + s.m) .+ (1:s.m)]
zu = sr.zu[1:s.nu]
ρ = s.opts.ρ
ζ = sqrt(sr.μ)
∇L(x) = s.∇c_func(x)'*s.λ
s.W .= ForwardDiff.jacobian(∇L,x)
s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= zl./((x - s.xl)[s.xl_bool])
s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= zu./((s.xu - x)[s.xu_bool])

Σp = Diagonal(p)\Diagonal(zp)
Σn = Diagonal(n)\Diagonal(zn)


s.c .= s.c_func(x)
s.A .= s.∇c_func(x)

s.H
s.H[1:s.n,1:s.n] .= (s.W + ζ*sr.DR'*sr.DR + s.Σl + s.Σu)
s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
s.H[s.n .+ (1:s.m),1:s.n] .= s.A
s.H[s.n .+ (1:s.m),s.n .+ (1:s.m)] .= -inv(Σp) - inv(Σn)

s.h[1:s.n] .= 0.
s.h[1:s.n] .= ζ*sr.DR'*sr.DR*(x - xr) + s.A'*sr.λ
s.h[(1:s.n)[s.xl_bool]] .-= sr.μ./(x - s.xl)[s.xl_bool]
s.h[(1:s.n)[s.xu_bool]] .+= sr.μ./(s.xu - x)[s.xu_bool]
s.h[s.n .+ (1:s.m)] .= s.c - p + n + ρ*Diagonal(zp)\(sr.μ*ones(s.m) - p) + ρ*Diagonal(zn)\(sr.μ*ones(s.m) - n)

d = -s.H\s.h
dx = d[1:s.n]
dλ = d[s.n .+ (1:s.m)]
dp = Diagonal(zp)\(sr.μ*ones(s.m) + Diagonal(p)*(sr.λ + dλ) - ρ*p)
dn = Diagonal(zn)\(sr.μ*ones(s.m) - Diagonal(n)*(sr.λ + dλ) - ρ*n)
dzp = sr.μ*Diagonal(p)\ones(s.m) - zp - Σp*dp
dzn = sr.μ*Diagonal(n)\ones(s.m) - zn - Σn*dn
dx
sr.d
[dx;dp;dn;dλ]

sr.d
_d = search_direction_restoration!(sr,s)
s.DR
sr.DR

# solve!(s)
# search_direction!(s)
# s.H
# H_ur = [s.H zeros(s.n+s.m,s.nl+s.nu); zeros(s.nl+s.nu,s.n+s.m+s.nl+s.nu)]
# H_ur[(1:s.n)[s.xl_bool],s.n+s.m .+ (1:s.nl)] .= Diagonal(-1.0*ones(s.nl))
# H_ur[(1:s.n)[s.xu_bool],s.n+s.m+s.nl .+ (1:s.nu)] .= Diagonal(-1.0*ones(s.nu))
#
# H_ur[s.n+s.m .+ (1:s.nl),(1:s.n)[s.xl_bool]] .= s.zl
# H_ur[s.n+s.m+s.nl .+ (1:s.nu),(1:s.n)[s.xu_bool]] .= s.zu
#
# H_ur[s.n+s.m .+ (1:s.nl),s.n+s.m .+ (1:s.nl)] .= s.x[s.xl_bool]
# H_ur[s.n+s.m+s.nl .+ (1:s.nu),s.n+s.m+s.nl .+ (1:s.nu)] .= s.x[s.xu_bool]
#
# s.c .= s.c_func(s.x)
# s.∇f .= s.∇f_func(s.x)
# s.A .= s.∇c_func(s.x)
#
# h_ur = zeros(s.n+s.m+s.nl+s.nu)
#
# h_ur[1:s.n] .= s.∇f + s.A'*s.λ
# h_ur[1:s.n][s.xl_bool] .-= s.zl
# h_ur[1:s.n][s.xu_bool] .+= s.zu
#
# h_ur[s.n .+ (1:s.m)] .= s.c
# h_ur[s.n + s.m .+ (1:s.nl)] .= ((s.x - s.xl)[s.xl_bool]).*s.zl .- s.μ
# h_ur[s.n + s.m + s.nl .+ (1:s.nu)] .= ((s.xu - s.x)[s.xu_bool]).*s.zu .- s.μ
#
# d = [s.d;s.dzl;s.dzu]
#
# iterative_refinement(d,H_ur,[s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nl+s.nu)],h_ur,verbose=true)
#
