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
c!, ∇c!, ∇²cλ! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cλ!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,iterative_refinement=true))
solve!(s,verbose=true)

# s_new = InteriorPointSolver(s.s.x,model,opts=Options{Float64}(kkt_solve=:symmetric,iterative_refinement=true))
# s_new.s.λ .= s.s.λ
# s_new.s.λ_al .+= s.s.ρ*s.s.c_tmp
# s_new.s.ρ = s.s.ρ*10.0
# solve!(s_new,verbose=true)
# s = s_new
