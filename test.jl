include("src/interior_point.jl")

n = 20
m = 10
x0 = rand(n)
xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.
f_func(x) = x'*x
c_func(x) = x[1:m].^2 .- 1.2

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s,verbose=true)
