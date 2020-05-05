function kkt_error_reduction(s::Solver)
    status = false
    s.t = 0
    s.x_copy .= s.x
    s.y_copy .= s.y
    s.zL_copy .= s.zL
    s.zU_copy .= s.zU

    s.s_copy = s.s
    s.zS_copy = s.zS_copy
    s.r_copy = s.r

    Fμ_norm = norm(eval_Fμ(s.x,s.y,s.zL,s.zU,s.s,s.zS,s.r,s),1)

    β_max!(s)

    while norm(eval_Fμ(s.x+s.β*s.dx,s.y+s.β*s.dy,s.zL+s.β*s.dzL,s.zU+s.β*s.dzU,s.s+s.β*s.ds,s.zS+s.β*s.dzS,s.r+s.β*s.dr,s),1) <= s.opts.κF*Fμ_norm
        s.x .+= s.β*s.dx
        if s.opts.nlp_scaling
            s.x .= s.Dx*s.x
        end

        s.y .+= s.β*s.dy
        s.zL .+= s.β*s.dzL
        s.zU .+= s.β*s.dzU

        s.s .+= s.β*s.ds
        s.zS .+= s.β*s.dzS
        s.r .+= s.β*s.dr

        if check_filter(θ(s.x,s),barrier(s.x,s.s,s.r,s),s.filter)
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

        s.s .= s.s_copy
        s.zS .= s.zS_copy
        s.r .= s.r_copy
        return false
    end
end

function eval_Fμ(x,y,zL,zU,_s,zS,r,s)
    s.model.∇f_func!(s.∇f,x,s.model)
    s.model.c_func!(s.c,x,s.model)
    if s.opts.nlp_scaling
        s.c .= s.Dc*s.c
    end
    s.model.∇c_func!(s.∇c,x,s.model)
    s.∇L[s.idx.x] = s.∇f + s.∇c'*y
    s.∇L[s.idx.xL] -= zL
    s.∇L[s.idx.xU] += zU

    s.mI != 0 && (s.∇L[s.model.n .+ (1:s.mI)] = -y[s.cI_idx] - zS)
    s.mA != 0 && (s.∇L[s.model.n + s.mI .+ (1:s.mA)] = r + 1/s.ρ*(s.λ - y[s.cA_idx]))

    # damping
    if s.opts.single_bnds_damping
        κd = s.opts.κd
        μ = s.μ
        s.∇L[s.idx.xLs] .+= κd*μ
        s.∇L[s.idx.xUs] .-= κd*μ
        s.mI != 0 && (s.∇L[s.model.n .+ (1:s.mI)] .+= κd*μ)
    end

    s.ΔxL .= (s.x - s.xL)[s.xL_bool]
    s.ΔxU .= (s.xU - s.x)[s.xU_bool]
    s.ΔsL .= s.s - s.sL

    s.Fμ[s.idx.primals] = s.∇L
    s.Fμ[s.idx.y] = s.c
    s.Fμ[s.idx.yI] -= s.s
    s.Fμ[s.idx.yA] -= s.r
    s.Fμ[s.idx.zL] = zL.*s.ΔxL .- s.μ
    s.Fμ[s.idx.zU] = zU.*s.ΔxU .- s.μ
    s.Fμ[s.idx.zS] = zS.*s.ΔsL .- s.μ

    return s.Fμ
end

function β_max!(s::Solver)
    α_max!(s)
    αz_max!(s)
    s.β = min(s.α,s.αz)
    return nothing
end
