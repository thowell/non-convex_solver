include("../src/interior_point.jl")

n = 3
m = 3

x0 = rand(n)

xL = zeros(n)

xU = Inf*ones(n)

f_func(x) = x'*x
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = x .- 1.0
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        ϵ_tol=1.0e-5,
                        max_iterative_refinement=10,
                        verbose=true)

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)

eval_step!(s.s)
search_direction!(s.s)
s.s.∇²f
s.s.inertia
s.s.H
rank(s.s.H)


cond(Array(s.s.H_sym))
initialize_restoration_solver!(s.s̄,s.s)
eval_step!(s.s̄)
search_direction_restoration!(s.s̄,s.s)
rank(s.s̄.H)
s.s̄.opts.max_iterative_refinement = 100
restoration!(s.s̄,s.s)
