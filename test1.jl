include("src/interior_point.jl")

n = 3
m = 2
x0 = [-2.; 3.; 1.]
# x0 = [0.; 3; 1]

xl = -Inf*ones(n)
xl[2] = 0.
xl[3] = 0.
xu = Inf*ones(n)

f_func(x) = x[1]
c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s)
s.x
s.λ

s.d

# a = (1.,1.)
# b = (0.,2.)
# c = (0.1,3.)
#
# f = Tuple[]
# push!(f,a)
# push!(f,b)
# push!(f,c)
#
# f
#
#
#
# add_to_filter!((-10.,0.),s)
