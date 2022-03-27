Base.@kwdef mutable struct Options{T}
    verbose::Bool = true
    residual_tolerance::T = 1.0e-8
    equality_tolerance::T = 1.0e-8
    max_residual_iterations::Int = 100
    slack_primal_initial::T=1.0 
    slack_dual_initial::T=1.0
    central_path_initial::T = 0.1
    central_path_tolerance::T = 10.
    scaling_central_path::T = 0.2
    exponent_central_path::T = 1.5
    penalty_initial::T=1.0 
    dual_initial::T=1.0
    min_fraction_to_boundary::T = 0.99
    constraint_violation_tolerance::T = 1.0e-5
    merit_tolerance::T = 1.0e-5
    regularization::T = 1.0
    step_size_tolerance::T = 0.05
    scaling_step_size::T = 0.5
    exponent_constraint_violation::T = 1.1
    exponent_merit::T = 2.3
    armijo_tolerance::T = 1.0e-4
    soc_tolerance::T = 0.99
    max_second_order_correction::Int = 4
    max_bound::T = 1.0e8

    min_regularization::T = 1.0e-20
    primal_regularization_initial::T = 1.0e-4
    max_regularization::T = 1.0e40
    dual_regularization::T = 1.0e-8
    scaling_regularization_initial::T = 100.0
    scaling_regularization::T = 8.0
    scaling_regularization_last::T = 1.0 / 3.0
    exponent_dual_regularization::T = 0.25

    barrier_tolerance::T = 1.0e-4

    linear_solve_type::Symbol = :symmetric # :symmetric, :fullspace, custom
    linear_solver::Symbol = :QDLDL

    iterative_refinement::Bool = true
    max_iterative_refinement::Int = 10
    min_iterative_refinement::Int = 1
    iterative_refinement_tolerance::T = 1.0e-8

    relax_bnds::Bool = true

    scaling::Bool = true
    scaling_tolerance::T = 100.

    machine_tolerance::T = 1.0e-16
end
