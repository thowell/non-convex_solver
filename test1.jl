# NOTE: Ipopt fails to converge on this problem. This solver also fail, unless δc ≈ 0.1.

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
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)
opts = Options{Float64}(kkt_solve=:symmetric,
                        relax_bnds=true,
                        nlp_scaling=true,
                        single_bnds_damping=true,
                        iterative_refinement=true,
                        max_iterative_refinement=100,
                        max_iter=100)

s = InteriorPointSolver(x0,model,opts=opts)
# s.s.ρ = 10.
@time solve!(s,verbose=true)
norm(c_func(s.s.x),1)

# s_new = InteriorPointSolver(s.s.x,model,opts=opts)
# s_new.s.y .= s.s.y
# s_new.s.y_al .= s.s.y_al + s.s.ρ*s.s.c
# s_new.s.ρ = s.s.ρ*10.0
# solve!(s_new,verbose=true)
# s = s_new
# norm(c_func(s.s.x),1)

# ######
# using Ipopt, MathOptInterface
# const MOI = MathOptInterface
#
# struct NLPProblem <: MOI.AbstractNLPEvaluator
#     enable_hessian::Bool
# end
#
#
# function MOI.eval_objective(prob::NLPProblem, Z)
#     return f_func(Z)
# end
#
# function MOI.eval_objective_gradient(prob::NLPProblem, grad_f, Z)
#     ∇f!(grad_f,Z)
#     return nothing
# end
#
# function MOI.eval_constraint(prob::NLPProblem, g, Z)
#     c!(g,Z)
# end
#
# function constraint_bounds()
#     c_l = zeros(m)
#     c_u = zeros(m)
#     return c_l, c_u
# end
#
# function MOI.eval_constraint_jacobian(prob::NLPProblem, jac, Z)
#     ∇c!(reshape(jac,m,n),Z)
# end
#
# function row_col!(row,col,r,c)
#     for cc in c
#         for rr in r
#             push!(row,convert(Int,rr))
#             push!(col,convert(Int,cc))
#         end
#     end
#     return row, col
# end
#
# function sparsity()
#     row = []
#     col = []
#
#     r = 1:m
#     c = 1:n
#
#     row_col!(row,col,r,c)
#
#     return collect(zip(row,col))
# end
#
# # sparsity(nlp)
#
# MOI.features_available(prob::NLPProblem) = [:Grad, :Jac]
# MOI.initialize(prob::NLPProblem, features) = nothing
# MOI.jacobian_structure(prob::NLPProblem) = sparsity()
# MOI.hessian_lagrangian_structure(prob::NLPProblem) = []
# MOI.eval_hessian_lagrangian(prob::NLPProblem, H, x, σ, μ) = nothing
#
# function solve(Z0)
#
#     z_l = xL
#     z_u = xU
#     c_l, c_u = constraint_bounds()
#
#     nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
#     block_data = MOI.NLPBlockData(nlp_bounds,NLPProblem(false),true)
#
#     solver = Ipopt.Optimizer()
#     solver.options["max_iter"] = 100000
#     # solver.options["nlp_scaling_method"] = "none"
#     # solver.options["linear_system_scaling"] = "none"
#
#     Z = MOI.add_variables(solver,n)
#
#     for i = 1:n
#         zi = MOI.SingleVariable(Z[i])
#         MOI.add_constraint(solver, zi, MOI.LessThan(z_u[i]))
#         MOI.add_constraint(solver, zi, MOI.GreaterThan(z_l[i]))
#         MOI.set(solver, MOI.VariablePrimalStart(), Z[i], Z0[i])
#     end
#
#     # Solve the problem
#     MOI.set(solver, MOI.NLPBlock(), block_data)
#     MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
#     MOI.optimize!(solver)
#
#     # Get the solution
#     res = MOI.get(solver, MOI.VariablePrimal(), Z)
#
#     return res
# end
#
# sol = solve(x0)
