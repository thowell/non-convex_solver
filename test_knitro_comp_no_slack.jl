include("src/interior_point.jl")

n = 8
m = 7

x0 = ones(n)

xL = zeros(n)
xU = Inf*ones(n)


f_func(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
             3*x[1] - x[2] - 3.0 - x[6];
             -x[1] + 0.5*x[2] + 4.0 - x[7];
             -x[1] - x[2] + 7.0 - x[8];
             x[3]*x[6];
             x[4]*x[7];
             x[5]*x[8]]
c!, ∇c!, ∇²cλ! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cλ!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,relax_bnds=false,single_bnds_damping=true,iterative_refinement=true,max_iter=100,ϵ_tol=1.0e-8,nlp_scaling=true))
s.s.ρ = 10.0
@time solve!(s,verbose=true)

s_new = InteriorPointSolver(s.s.x,model,opts=Options{Float64}(kkt_solve=:symmetric,iterative_refinement=true))
s_new.s.λ .= s.s.λ
s_new.s.λ_al .+= s.s.ρ*s.s.c_tmp
s_new.s.ρ = s.s.ρ*10.0
solve!(s_new,verbose=true)
s = s_new

x[3]
x[6]

x[4]
x[7]

x[5]
x[8]
