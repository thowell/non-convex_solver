@with_kw mutable struct Options{T}
    ϵ_tol::T = 1.0e-8
    zl0::T = 1.0
    zu0::T = 1.0
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
    λ_init_ls::Bool = true
    λ_max::T = 1.0e3
    α_sep::Bool = false

    δw_min::T = 1.0e-20
    δw0::T = 1.0e-4
    δw_max::T = 1.0e40
    δc::T = 1.0e-8
    κw⁺_::T = 100.0
    κw⁺::T = 8.0
    κw⁻::T = 1.0/3.0
    κc::T = 0.25

    ρ::T = 1000.
    κF::T = 0.999

    κ1::T = 1.0e-2
    κ2::T = 1.0e-2
end
