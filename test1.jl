include("src/interior_point.jl")

n = 3
m = 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]
c!, ∇c!, ∇²cλ! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cλ!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,relax_bnds=true,single_bnds_damping=true,iterative_refinement=true,max_iter=100))
@time solve!(s,verbose=true)
