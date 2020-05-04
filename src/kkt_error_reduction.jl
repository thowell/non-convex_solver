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
    s.model.∇f_func!(s.∇f,x,s.model)
    s.model.c_func!(s.c,x,s.model)
    if s.opts.nlp_scaling
        s.c .= s.Dc*s.c
    end
    s.model.∇c_func!(s.∇c,x,s.model)
    s.∇L .= s.∇f + s.∇c'*y
    s.∇L[s.xL_bool] -= zL
    s.∇L[s.xU_bool] += zU

    # damping
    if s.opts.single_bnds_damping
        κd = s.opts.κd
        μ = s.μ
        s.∇L[s.xLs_bool] .+= κd*μ
        s.∇L[s.xUs_bool] .-= κd*μ
    end

    s.ΔxL .= (s.x - s.xL)[s.xL_bool]
    s.ΔxU .= (s.xU - s.x)[s.xU_bool]

    s.Fμ[s.idx.x] = s.∇L
    s.Fμ[s.idx.y] = s.c
    s.Fμ[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)
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
