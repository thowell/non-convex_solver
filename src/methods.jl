function init_sd(λ,z,n,m,s_max)
    sd = max(s_max,(norm(λ,1) + norm(z,1))/(n+m))/s_max
    return sd
end

function init_sc(z,n,s_max)
    sc = max(s_max,norm(z,1)/n)/s_max
    return sc
end

function eval_Eμ(x,λ,zl,zu,xl,xu,xl_bool,xu_bool,c,∇L,μ,sd,sc)
    Eμ = max(norm(∇L,Inf)/sd,norm(c,Inf),norm((x-xl)[xl_bool].*zl .- μ,Inf)/sc,norm((xu-x)[xu_bool].*zu .- μ,Inf)/sc)
    return Eμ
end

function eval_Fμ(x,λ,zl,zu,xl,xu,xl_bool,xu_bool,c,∇L,μ)
    Fμ = [∇L; c; (x-xl)[xl_bool].*zl .- μ; (xu-x)[xu_bool].*zu .- μ]
    return Fμ
end

function update_μ(μ,κμ,θμ,ϵ_tol)
    μ⁺ = max(ϵ_tol/10.,min(κμ*μ,μ^θμ))
    return μ⁺
end

function update_τ(μ,τ_min)
    τ⁺ = max(τ_min,1.0-μ)
    return τ⁺
end

function fraction_to_boundary(x,d,α,τ)
    return all(x + α*d .>= (1 - τ)*x)
end

function fraction_to_boundary_bnds(x,xl,xu,d,α,τ)
    return all((xu-(x + α*d)) .>= (1 - τ)*(xu-x)) && all(((x + α*d)-xl) .>= (1 - τ)*(x-xl))
end

function reset_z(z,x,μ,κΣ)
    z⁺ = max(min(z,κΣ*μ/x),μ/(κΣ*x))
    return z⁺
end

function armijo_condtion(φ⁺,φ,η,α,∇φ,d)
    return (φ⁺ <= φ + η*α*∇φ'*d)
end

function switching_condition(∇φ,d,α,sφ,δ,θ,sθ)
    return (∇φ'*d < 0. && α*(-∇φ'*d)^sφ > δ*θ^sθ)
end

function sufficient_progress(θ⁺,θ,φ⁺,φ,γθ,γφ)
    return (θ⁺ <= (1-γθ)*θ || φ⁺ <= φ - γφ*θ)
end

function update_α_min(d,θ,∇φ,θ_min,δ,γα,γθ,γφ,sθ,sφ)
    if ∇φ'*d < 0. && θ <= θ_min
        α_min = γα*min(γθ,γφ*θ/(-∇φ'*d),δ*(θ^sθ)/(-∇φ'*d)^sφ)
    elseif ∇φ'*d < 0. && θ > θ_min
        α_min = γα*min(γθ,γφ*θ/(-∇φ'*d))
    else
        α_min = γα*γθ
    end
    return α_min
end

function init_θ_max(θ)
    θ_max = 1.0e4*max(1.0,θ)
    return θ_max
end

function init_θ_min(θ)
    θ_min = 1.0e-4*max(1.0,θ)
    return θ_min
end

function init_x0(x,xl,xu,κ1,κ2)
    pl = min(κ1*max(1.0,abs(xl)),κ2*(xu-xl))
    pu = min(κ1*max(1.0,abs(xu)),κ2*(xu-xl))

    # projection
    if x < xl+pl
        x = xl+pl
    elseif x > xu-pu
        x = xu-pu
    end
    return x
end

function init_λ(zl,zu,∇f,∇c,n,m,xl_bool,xu_bool,λ_max)

    H = [Matrix(I,n,n) ∇c';∇c zeros(m,m)]
    h = zeros(n+m)
    h[1:n] .= ∇f
    h[1:n][xl_bool] .-= zl
    h[1:n][xu_bool] .+= zu

    d = -H\h

    λ = d[n .+ (1:m)]

    if norm(λ,Inf) > λ_max
        @warn "least-squares λ init failure:\n λ_max = $(norm(λ,Inf))"
        return zeros(m)
    else
        return λ
    end
end

function barrier(x,xl,xu,xl_bool,xu_bool,μ,f_func)
    _sum = f_func(x) - μ*sum(log.((xu - x)[xu_bool])) - μ*sum(log.((x - xl)[xl_bool]))
    return _sum
end

function set_DR(DR,xr,n)
    for i = 1:n
        DR[i,i] = min(1.0,1.0/abs(xr[i]))
    end
end

function init_n(c,μ,ρ)
    n = (μ - ρ*c)/(2.0*ρ) + sqrt(((μ-ρ*c)/(2.0*ρ))^2 + (μ*c)/(2.0*ρ))
    return n
end

function init_p(n,c)
    p = c + n
    return p
end

function eval_Eμ_restoration(x,p,n,λ,zl,zu,zp,zn,xl,xu,xl_bool,xu_bool,c,∇L,μ,ρ,sd,sc)
    Eμ = max(norm([∇L;ρ .- zp - λ;ρ .- zn + λ],Inf)/sd,
        norm(c,Inf),
        norm((x-xl)[xl_bool].*zl .- μ,Inf)/sc,
        norm((xu-x)[xu_bool].*zu .- μ,Inf)/sc,
        norm(p.*zp .- μ,Inf),
        norm(n.*zn .- μ,Inf),)
    return Eμ
end
