include("../src/interior_point.jl")

function knitro_comp()
    n = 8
    m = 7

    x0 = rand(n)

    xL = zeros(n)
    xU = Inf*ones(n)

    f_func(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
    c_func(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
                 3*x[1] - x[2] - 3.0 - x[6];
                 -x[1] + 0.5*x[2] + 4.0 - x[7];
                 -x[1] - x[2] + 7.0 - x[8];
                 x[3]*x[6];
                 x[4]*x[7];
                 x[5]*x[8]]

    f, ∇f!, ∇²f! = objective_functions(f_func)
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    cI_idx = zeros(Bool,m)
    cA_idx = ones(Bool,m)
    model = Model(n,m,xL,xU,
                  f,∇f!,∇²f!,
                  c!,∇c!,∇²cy!,
                  cI_idx=cI_idx,cA_idx=cA_idx)

    return x0,model
end

opts = Options{Float64}(kkt_solve=:slack,
                        relax_bnds=true,
                        single_bnds_damping=true,
                        iterative_refinement=true,
                        max_iter=1000,
                        ϵ_tol=1.0e-8,
                        ϵ_al_tol=1.0e-8,
                        nlp_scaling=true,
                        quasi_newton=:none,
                        quasi_newton_approx=:lagrangian,
                        verbose=true)

s = InteriorPointSolver(knitro_comp()...,opts=opts)

@time solve!(s)
s.s.qn.fail_cnt

# x = s.s.x
# x[3]
# x[6]
#
# x[4]
# x[7]
#
# x[5]
# x[8]
