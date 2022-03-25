Base.@kwdef mutable struct Options{T}
    verbose::Bool = true
    residual_tolerance::T = 1.0e-8
    equality_tolerance::T = 1.0e-8
    max_residual_iterations::Int = 100
    zL0::T = 1.0
    zU0::T = 1.0
    central_path_initial::T = 0.1
    central_path_tolerance::T = 10.
    κcentral_path::T = 0.2
    θcentral_path::T = 1.5
    min_fraction_to_boundary::T = 0.99
    constraint_violation_tolerance::T = 1.0e-5
    γφ::T = 1.0e-5
    regularization::T = 1.0
    step_size_tolerance::T = 0.05
    sθ::T = 1.1
    sφ::T = 2.3
    ηφ::T = 1.0e-4
    κ_soc::T = 0.99
    p_max::Int = 4
    s_max::T = 100.
    κΣ::T = 1.0e10
    bnd_tol::T = 1.0e8

    primal_regularization_min::T = 1.0e-20
    primal_regularization_initial::T = 1.0e-4
    primal_regularization_max::T = 1.0e40
    dual_regularization::T = 1.0e-8
    κw⁺_::T = 100.0
    κw⁺::T = 8.0
    κw⁻::T = 1.0/3.0
    κc::T = 0.25

    κ1::T = 1.0e-2
    κ2::T = 1.0e-2

    κd::T = 1.0e-4

    kkt_solve::Symbol = :symmetric # :symmetric, :fullspace, custom
    linear_solver::Symbol = :QDLDL

    iterative_refinement::Bool = true
    max_iterative_refinement::Int = 10
    min_iterative_refinement::Int = 1
    iterative_refinement_tolerance::T = 1.0e-8

    relax_bnds::Bool = true

    g_max::T = 100.
    nlp_scaling::Bool = true

    machine_tolerance::T = 1.0e-16
end
