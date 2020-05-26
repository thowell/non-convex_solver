function kkt_error_reduction(s::Solver)
    status = false
    s.t = 0
    s.x_copy .= s.x
    s.y_copy .= s.y
    s.zL_copy .= s.zL
    s.zU_copy .= s.zU

    Fμ_norm = norm(eval_Fμ(s.x,s.y,s.zL,s.zU,s),1)

    β_max!(s)

    while norm(eval_Fμ(s.x+s.β*s.dx,s.y+s.β*s.dy,s.zL+s.β*s.dzL,s.zU+s.β*s.dzU,s),1) <= s.opts.κF*Fμ_norm
        s.x .+= s.β*s.dx
        if s.opts.nlp_scaling
            s.x .= s.Dx*s.x
        end

        s.y .+= s.β*s.dy
        s.zL .+= s.β*s.dzL
        s.zU .+= s.β*s.dzU

        if check_filter(θ(s.x,s),barrier(s.x,s),s.filter)
            status = true
            break
        else
            search_direction!(s)
            β_max!(s)
        end

        s.t += 1
        Fμ_norm = norm(s.Fμ,1)
    end

    if status
        s.α = s.β
        s.αz = s.β
        return true
    else
        s.x .= s.x_copy
        s.y .= s.y_copy
        s.zL .= s.zL_copy
        s.zU .= s.zU_copy
        return false
    end
end

function eval_Fμ(x,y,zL,zU,s)
    eval_∇f!(s.model,x)

    eval_c!(s.model,x)
    get_c_scaled!(s.c,s)

    eval_∇c!(s.model,x)

    s.∇L .= get_∇f(s.model) + get_∇c(s.model)'*y
    s.∇L[s.idx.xL] -= zL
    s.∇L[s.idx.xU] += zU

    s.model.mA > 0 && (s.∇L[s.idx.r] += s.λ + s.ρ*view(x,s.idx.r))

    # damping
    if s.opts.single_bnds_damping
        κd = s.opts.κd
        μ = s.μ
        s.∇L[s.idx.xLs] .+= κd*μ
        s.∇L[s.idx.xUs] .-= κd*μ
    end

    s.ΔxL .= view(x,s.idx.xL) - view(s.model.xL,s.idx.xL)
    s.ΔxU .= view(s.model.xU,s.idx.xU) - view(x,s.idx.xU)

    s.Fμ[s.idx.x] = s.∇L
    s.Fμ[s.idx.y] = s.c
    s.Fμ[s.idx.zL] = zL.*s.ΔxL .- s.μ
    s.Fμ[s.idx.zU] = zU.*s.ΔxU .- s.μ
    return s.Fμ
end

function β_max!(s::Solver)
    α_max!(s)
    αz_max!(s)
    s.β = min(s.α,s.αz)
    return nothing
end
