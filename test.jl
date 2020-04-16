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

c_func(x,ρ) = x[1:m].^2 .- 1.2
c!, ∇c!, ∇²cλ! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cλ!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,iterative_refinement=true))
@time solve!(s,verbose=true)
