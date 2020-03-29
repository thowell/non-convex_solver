include("src/interior_point.jl")

n = 2
m = 0
x0 = [0.; 0.]

xL = -Inf*ones(n)
xU = [0.25; 0.25]#Inf*ones(n)

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
c_func(x) = zeros(m)

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = zeros(m,n)

s = Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}(max_iter=1e3,kkt_solve=:unreduced))
solve!(s,verbose=true)
