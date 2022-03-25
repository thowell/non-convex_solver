# TODO: add descriptions to these options
@with_kw mutable struct Options{T}
    verbose::Bool = true
    ϵ_tol::T = 1.0e-8
    ϵ_al_tol::T = 1.0e-8
    max_iter::Int = 100
    zL0::T = 1.0
    zU0::T = 1.0
    μ0::T = 0.1
    κϵ::T = 10.
    κμ::T = 0.2
    θμ::T = 1.5
    τ_min::T = 0.99
    γθ::T = 1.0e-5
    γφ::T = 1.0e-5
    δ::T = 1.0
    γα::T = 0.05
    sθ::T = 1.1
    sφ::T = 2.3
    ηφ::T = 1.0e-4
    κ_soc::T = 0.99
    p_max::Int = 4
    s_max::T = 100.
    κΣ::T = 1.0e10
    bnd_tol::T = 1.0e8
    y_init_ls::Bool = true
    y_max::T = 1.0e3
    z_reset::Bool = true

    δw_min::T = 1.0e-20
    δw0::T = 1.0e-4
    δw_max::T = 1.0e40
    δc::T = 1.0e-8
    κw⁺_::T = 100.0
    κw⁺::T = 8.0
    κw⁻::T = 1.0/3.0
    κc::T = 0.25

    ρ_resto::T = 1000.
    κF::T = 0.999
    κ_resto::T = 0.9

    κ1::T = 1.0e-2
    κ2::T = 1.0e-2

    single_bnds_damping::Bool = true
    κd::T = 1.0e-4

    kkt_solve::Symbol = :symmetric # :symmetric, :fullspace, custom
    linear_solver::Symbol = :MA57

    small_search_direction_max::Int = 2

    iterative_refinement::Bool = true
    max_iterative_refinement::Int = 10
    min_iterative_refinement::Int = 1
    ϵ_iterative_refinement::T = 1.0e-8

    relax_bnds::Bool = true

    g_max::T = 100.
    nlp_scaling::Bool = true

    watch_dog_iters::Int = 2

    max_fail_cnt::Int = 4
    ϵ_mach::T = 1.0e-16

    quasi_newton::Symbol = :none # :none, :bfgs, :lbfgs, :custom
    lbfgs_length::Int = 6
    quasi_newton_approx::Symbol = :lagrangian
    bfgs_max_fail_cnt::Int = 2

    restoration::Bool=true
end
