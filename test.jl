include("src/interior_point.jl")

n = 20
m = 10
x0 = rand(n)
xl = -Inf*ones(n)
xl[1] = -10.
xl[2] = -5.
xu = Inf*ones(n)
xu[5] = 20.
f_func(x) = x'*x
c_func(x) = x[1:m].^2 .- 1.2

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s)

search_direction!(s)
# s.n
# s.m
# rs = RestorationSolver(s)
# rs.n
#
# solve_restoration!(rs,s)
#

# rs.f_func(rs.x)
# rs.∇f_func(rs.x)
# rs.c_func(rs.x)
# rs.∇c_func(rs.x)
#
# rs.restoration
#
# update_restoration_objective!(rs,s.n,s.x)
