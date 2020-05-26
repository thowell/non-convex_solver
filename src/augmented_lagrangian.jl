function λ_update!(s::Solver)
    s.λ .+= s.ρ*s.xr
    return nothing
end

function ρ_update!(s::Solver)
    s.ρ = 1.0/s.μ
    return nothing
end

function augmented_lagrangian_update!(s::Solver)
    λ_update!(s)
    ρ_update!(s)
    return nothing
end
