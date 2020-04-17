function kkt_error_reduction(s::Solver)
    status = false
    s.t = 0
    s.x_copy .= copy(s.x)
    s.λ_copy .= copy(s.λ)
    s.zL_copy .= copy(s.zL)
    s.zU_copy .= copy(s.zU)

    Fμ_norm = norm(eval_Fμ(s.x,s.λ,s.zL,s.zU,s),1)

    β_max!(s)

    while norm(eval_Fμ(s.x+s.β*s.dx,s.λ+s.β*s.dλ,s.zL+s.β*s.dzL,s.zU+s.β*s.dzU,s),1) <= s.opts.κF*Fμ_norm
        s.x .+= s.β*s.dx
        if s.opts.nlp_scaling
            s.x .= s.Dx*s.x
        end

        s.λ .+= s.β*s.dλ
        s.zL .+= s.β*s.dzL
        s.zU .+= s.β*s.dzU



        if check_filter(θ(s.x,s),barrier(s.x,s),s)
            status = true
            break
        else
            search_direction!(s)
            @warn " kkt error reduce refinement!!"
            β_max!(s)
        end

        # if s.t > 10
        #     break
        # end
        s.t += 1
        Fμ_norm = norm(s.Fμ,1)
    end

    if status
        s.α = s.β
        s.αz = s.β
        return true
    else
        s.x .= copy(s.x_copy)
        s.λ .= copy(s.λ_copy)
        s.zL .= copy(s.zL_copy)
        s.zU .= copy(s.zU_copy)
        return false
    end
end

function eval_Fμ(x,λ,zL,zU,s)
    s.model.∇f_func!(s.∇f,x)
    s.model.c_func!(s.c,x)
    if s.opts.nlp_scaling
        s.c .= s.Dc*s.c
    end
    s.model.∇c_func!(s.∇c,x)
    s.∇L .= s.∇f + s.∇c'*λ
    s.∇L[s.xL_bool] -= zL
    s.∇L[s.xU_bool] += zU

    # damping
    if s.opts.single_bnds_damping
        κd = s.opts.κd
        μ = s.μ
        s.∇L[s.xLs_bool] .+= κd*μ
        s.∇L[s.xUs_bool] .-= κd*μ
    end

    s.Fμ[s.idx.x] = s.∇L
    s.Fμ[s.idx.λ] = s.c
    s.Fμ[s.idx.λ[s.c_relax]] .+= 1.0/s.ρ*(s.λ_al - s.λ[s.c_relax])
    s.Fμ[s.idx.zL] = zL.*((x-s.xL)[s.xL_bool]) .- s.μ
    s.Fμ[s.idx.zU] = zU.*((s.xU-x)[s.xU_bool]) .- s.μ
    return s.Fμ
end

function β_max!(s::Solver)
    α_max!(s)
    αz_max!(s)
    s.β = min(s.α,s.αz)
    return nothing
end
