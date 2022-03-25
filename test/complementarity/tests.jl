include("problems.jl")
opts = Options{Float64}(
                        kkt_solve=:symmetric,
                        # relax_bnds=true,
                        # single_bnds_damping=true,
                        # iterative_refinement=true,
                        max_iter=1000,
                        ϵ_tol=1.0e-5,
                        ϵ_al_tol=1.0e-5,
                        # nlp_scaling=true,
                        # quasi_newton=:lbfgs,
                        # quasi_newton_approx=:lagrangian,
                        verbose=true,
                        linear_solver=:QDLDL,
                        # lbfgs_length=6
                        )
                        
@testset "Complementarity problems" begin
    opts.verbose=false
    solver = NCSolver(knitro_comp()...,opts=opts)
    solve!(solver)
    @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
end
