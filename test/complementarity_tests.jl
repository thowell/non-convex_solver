include("complementarity_problems.jl")

opts = Options{Float64}(kkt_solve=:symmetric,
                        max_iter=1000,
                        verbose=true)

@testset "Complementarity problems" begin
    solver = InteriorPointSolver(knitro_comp()...,opts=opts)
    solve!(solver)
    @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
end
