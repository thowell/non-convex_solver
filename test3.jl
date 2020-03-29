include("src/interior_point.jl")

n = 2
m = 1
x0 = [-1.; 10.]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 2.0*(x[1]^2 + x[2]^2 - 1.0) - x[1]
c_func(x) = [x[1]^2 + x[2]^2 - 1.0]
c_func_d(x) = x[1]^2 + x[2]^2 - 1.0
∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = Array(ForwardDiff.gradient(c_func_d,x)')

s = Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s,verbose=true)
