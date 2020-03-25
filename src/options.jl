@with_kw mutable struct Options{T}
    ϵ_tol::T = 1.0e-8
    max_iter::Int = 100
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
    κ_resto::T = 0.9

    κ1::T = 1.0e-2
    κ2::T = 1.0e-2
end

function Base.copy(o::Options{T}) where T
    return Options{T}(ϵ_tol=copy(o.ϵ_tol),
                    max_iter=copy(o.max_iter),
                    zl0=copy(o.zl0),
                    zu0=copy(o.zu0),
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
                    λ_init_ls=copy(o.λ_init_ls),
                    λ_max=copy(o.λ_max),
                    α_sep=copy(o.α_sep),
                    δw_min=copy(o.δw_min),
                    δw0=copy(o.δw0),
                    δw_max=copy(o.δw_max),
                    δc=copy(o.δc),
                    κw⁺_=copy(o.κw⁺_),
                    κw⁺=copy(o.κw⁺),
                    κw⁻=copy(o.κw⁻),
                    κc=copy(o.κc),
                    ρ=copy(o.ρ),
                    κF=copy(o.κF),
                    κ_resto=copy(o.κ_resto),
                    κ1=copy(o.κ1),
                    κ2=copy(o.κ2))
end

o = Options{Float64}()
o_ = copy(o)
