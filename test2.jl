include("src/interior_point.jl")

n = 2
m = 0
x0 = [0.; 0.]

xl = zeros(n)
xu = ones(n)

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
c_func(x) = 0.

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = zeros(0,n)

s = Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s)
