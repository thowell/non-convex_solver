include("src/interior_point.jl")

n = 3
m = 2
x0 = [-2.0; 3.; 1.]
# x0 = [-10.0; 3.0; 1.0]
x0 = [-.5;1.;1.]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]
c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s,verbose=true)
