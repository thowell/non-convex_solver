include("../src/interior_point.jl")

n = 10
m = 5

x0 = ones(n+m)

xL = -Inf*ones(n+m)
xL[n .+ (1:m)] .= 0.
xU = Inf*ones(n+m)

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = x[1:m] .- 1.0 - x[n .+ (1:m)]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n+m,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

opts = Options{Float64}(kkt_solve=:unreduced,
                        iterative_refinement=false,
                        verbose=false)

s = InteriorPointSolver(x0,model,cI_idx=zeros(Bool,model.m),opts=opts)
@time solve!(s)

s.s.x

##
x0 = ones(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = x[1:m] .- 1.0
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

opts = Options{Float64}(kkt_solve=:unreduced,
                        iterative_refinement=false,
                        verbose=false)

s = InteriorPointSolver(x0,model,cI_idx=ones(Bool,model.m),opts=opts)
@time solve!(s)

s.s.x
s.s.sL
s.s.s
s.s.ΔsL
