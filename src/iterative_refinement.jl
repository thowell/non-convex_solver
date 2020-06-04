"""
    iterative_refinement(d, s::Solver)

Use iterative refinement on the fullspace KKT system to improve the current step `d`.
"""
function iterative_refinement(d::Vector{T},s::Solver) where T
    s.d_copy .= d
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)
    res_norm_init = copy(res_norm)

    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :fullspace
            s.Δ .= (s.H+Diagonal(s.δ))\s.res
        elseif s.opts.kkt_solve == :symmetric
            s.res_xL .+= s.res_zL./(s.ΔxL .- s.δc)
            s.res_xU .-= s.res_zU./(s.ΔxU .- s.δc)

            s.Δ_xy .= ma57_solve(s.LBL,Array(s.res_xy))
            s.Δ_zL .= -s.σL.*s.Δ_xL + s.res_zL./(s.ΔxL .- s.δc)
            s.Δ_zU .= s.σU.*s.Δ_xU + s.res_zU./(s.ΔxU .- s.δc)
        end

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end
    # println("res: $res_norm")
    @logmsg InnerLoop "res: $(round(res_norm_init, sigdigits=1)) -> $(round(res_norm, sigdigits=1))"

    if res_norm < s.opts.ϵ_iterative_refinement
        return true
    else
        d .= s.d_copy
        return false
    end
end

function iterative_refinement_slack(d::Vector{T},s::Solver) where T
    s.d_copy .= d
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)
    res_norm_init = copy(res_norm)
    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        n = s.model_opt.n
        m = s.model_opt.m
        mI = s.model.mI
        mE = s.model.mE
        mA = s.model.mA
        nL = s.model_opt.nL
        nU = s.model_opt.nU
        idx = s.idx

        ΔxL = view(s.ΔxL,1:nL)
        ΔsL = view(s.ΔxL,nL .+ (1:mI))
        ΔxU = s.ΔxU
        zL = view(s.zL,1:nL)
        zS = view(s.zL,nL .+ (1:mI))
        zU = s.zU

        s.res_xL[1:nL] .+= s.res__zL./(ΔxL .- s.δc)
        s.res_xU[1:nU] .-= s.res_zU./(ΔxU .- s.δc)
        s.res_yI .+= ((ΔsL .- s.δc).*s.res_s + s.res_zs)./zS
        s.res_yA .+= 1.0/(s.ρ + s.δw)*s.res_r

        s.Δ__xy .= ma57_solve(s.LBL_slack,Array(s.res__xy))
        s.Δ_r .= 1.0/(s.ρ + s.δw)*(s.Δ_yA - s.res_r)
        s.Δ__zL .= -(zL.*s.Δ_xL[1:nL] + s.res__zL)./(ΔxL .- s.δc)
        s.Δ_zU .= (zU.*s.Δ_xL[1:nU] - s.res_zU)./(ΔxU .- s.δc)
        s.Δ_zs .= -s.Δ_yI + s.res_s
        s.Δ_s .= -((ΔsL .- s.δc).*s.Δ_zs + s.res_zs)./zS

        # Is = Matrix(I,mI,mI)
        # tmp = [s.δw*Is -Is; Diagonal(zS) Diagonal(ΔsL .- s.δc)]\[-s.res_s + s.Δ_yI; -s.res_zs]
        # s.Δ_s .= tmp[1:mI]
        # s.Δ_zs .= tmp[mI .+ (1:mI)]

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    @logmsg InnerLoop "res: $(round(res_norm_init, sigdigits=1)) -> $(round(res_norm, sigdigits=1))"
    if res_norm < s.opts.ϵ_iterative_refinement
        return true
    else
        d .= s.d_copy
        return false
    end
end
