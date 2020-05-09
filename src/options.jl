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

    small_search_direction_max::Int = 2

    iterative_refinement::Bool = true
    max_iterative_refinement::Int = 10
    min_iterative_refinement::Int = 1
    ϵ_iterative_refinement::T = 1.0e-8

    relax_bnds::Bool = true

    g_max::T = 100.
    nlp_scaling::Bool = true

    max_fail_cnt::Int = 4
    ϵ_mach::T = 1.0e-16
end

# TODO: do this with a loop
function Base.copy(o::Options{T}) where T
    return Options{T}(
                    verbose=copy(o.verbose),
                    ϵ_tol=copy(o.ϵ_tol),
                    max_iter=copy(o.max_iter),
                    zL0=copy(o.zL0),
                    zU0=copy(o.zU0),
                    μ0=copy(o.μ0),
                    κϵ=copy(o.κϵ),
                    κμ=copy(o.κμ),
                    θμ=copy(o.θμ),
                    τ_min=copy(o.τ_min),
                    γθ=copy(o.γθ),
                    γφ=copy(o.γφ),
                    δ=copy(o.δ),
                    γα=copy(o.γα),
                    sθ=copy(o.sθ),
                    sφ=copy(o.sφ),
                    ηφ=copy(o.ηφ),
                    κ_soc=copy(o.κ_soc),
                    p_max=copy(o.p_max),
                    s_max=copy(o.s_max),
                    κΣ=copy(o.κΣ),
                    bnd_tol=copy(o.bnd_tol),
                    y_init_ls=copy(o.y_init_ls),
                    y_max=copy(o.y_max),
                    z_reset=copy(o.z_reset),
                    δw_min=copy(o.δw_min),
                    δw0=copy(o.δw0),
                    δw_max=copy(o.δw_max),
                    δc=copy(o.δc),
                    κw⁺_=copy(o.κw⁺_),
                    κw⁺=copy(o.κw⁺),
                    κw⁻=copy(o.κw⁻),
                    κc=copy(o.κc),
                    ρ_resto=copy(o.ρ_resto),
                    κF=copy(o.κF),
                    κ_resto=copy(o.κ_resto),
                    κ1=copy(o.κ1),
                    κ2=copy(o.κ2),
                    single_bnds_damping=copy(o.single_bnds_damping),
                    κd=copy(o.κd),
                    kkt_solve=o.kkt_solve,
                    small_search_direction_max=copy(o.small_search_direction_max),
                    iterative_refinement=copy(o.iterative_refinement),
                    max_iterative_refinement=copy(o.max_iterative_refinement),
                    min_iterative_refinement=copy(o.min_iterative_refinement),
                    ϵ_iterative_refinement=copy(o.ϵ_iterative_refinement),
                    relax_bnds=copy(o.relax_bnds),
                    g_max=copy(o.g_max),
                    nlp_scaling=copy(o.nlp_scaling),
                    max_fail_cnt=copy(o.max_fail_cnt),
                    ϵ_mach=copy(o.ϵ_mach))
end
