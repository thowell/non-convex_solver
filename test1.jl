include("src/interior_point.jl")

n = 3
m = 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]
c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}(max_iter=500))
solve!(s,verbose=true)
#
# iterative_refinement_phase1(s.d,s.x,s.zL,s.zU,s.xL,s.xU,s.xL_bool,s.xU_bool,s.Hu,s.H,
#     [s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)],-s.hu,s.n,s.nL,s.nU,s.m,
#     max_iter=s.opts.max_iterative_refinement,ϵ=s.opts.ϵ_iterative_refinement)
#
# iterative_refinement(s.d,s.Hu,
#     [s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)],-s.hu,s.n,s.m,
#     max_iter=s.opts.max_iterative_refinement,ϵ=s.opts.ϵ_iterative_refinement)
