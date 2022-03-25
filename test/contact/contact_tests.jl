include("problems.jl")

opts = Options{Float64}(
                        kkt_solve=:symmetric,
                        # iterative_refinement=true,
                        ϵ_tol=1.0e-5,
                        ϵ_al_tol=1.0e-5,
                        # max_iterative_refinement=10,
                        max_iter=250,
                        verbose=true,
                        # quasi_newton=:bfgs,
                        # quasi_newton_approx=:lagrangian,
                        linear_solver=:QDLDL,
                        # lbfgs_length=6
                        )

@testset "Contact problems" begin
    # particle
	# @info "particle"
	solver = NCSolver(particle()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(
              particle(qpp=[0.,0.,10.],v0=[10.,-7.0, 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = NCSolver(
              particle(qpp=[0.,0.,0.01],v0=[1.,0., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = NCSolver(
              particle(qpp=[0.,0.,0.01],v0=[1.,1., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = NCSolver(
              particle(qpp=[0.,0.,0.01],v0=[0.,0., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = NCSolver(
              particle(qpp=[0.,0.,-1.0],v0=[0.,0., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    # Raibert hopper
	# @info "raibert hopper"
    solver = NCSolver(
              hopper()...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = NCSolver(
              hopper(T=5)...,
              opts=opts)
    solve!(solver)
    @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = NCSolver(
              hopper(T=5,r=0.7,q0=[0.,0.7,0.7,0.,0.],qf =[0.,1.0+0.7,0.7,0.,0.])...,
              opts=opts)
    solve!(solver)
    @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
end
