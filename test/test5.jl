include("../src/interior_point.jl")

n = 50
m = 10

x0 = rand(n+m)

xL = -Inf*ones(n+m)
xL[n .+ (1:m)] .= 0.
xU = Inf*ones(n+m)

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = 5.0*x[1:m].^2 .- 3.0 - x[n .+ (1:m)]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n+m,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

s = InteriorPointSolver(x0,model)
@time solve!(s)
s.s.x

##
n = 50
m = 10

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = 5.0*x[1:m].^2 .- 3.0
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=ones(Bool,model.m))

s = InteriorPointSolver(x0,model)
@time solve!(s)

s.s.x
