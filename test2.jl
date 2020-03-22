include("src/interior_point.jl")

n = 2
m = 0
x0 = [0.; 0.]

xl = zeros(n)
xu = ones(n)

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
c_func(x) = 0.
norm(c_func(x0),1)

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = zeros(0,n)
∇c_func(x0)
s = Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s)
s.x

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
