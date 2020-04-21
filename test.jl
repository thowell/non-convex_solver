include("src/interior_point.jl")

n = 100
m = 50

x0 = ones(n)

xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = x[1:m].^2 .- 1.2
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

opts = Options{Float64}(kkt_solve=:symmetric,iterative_refinement=true)
c_al_idx = ones(Bool,model.m)
s = InteriorPointSolver(x0,model,c_al_idx=c_al_idx,opts=opts)
# s.s.ρ = 1.0/s.s.μ
solve!(s,verbose=true)
norm(c_func(s.s.x),1)

# s_new = InteriorPointSolver(s.s.x,model,c_al_idx=c_al_idx,opts=opts)
# s_new.s.y .= s.s.y
# s_new.s.y_al .= s.s.y_al + s.s.ρ*s.s.c[c_al_idx]
# s_new.s.ρ = s.s.ρ*10.0
# solve!(s_new,verbose=true)
# s = s_new
# norm(c_func(s.s.x),1)
