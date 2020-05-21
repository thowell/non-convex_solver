"""
    iterative_refinement(d, s::Solver)

Use iterative refinement on the fullspace KKT system to improve the current step `d`.
"""
function iterative_refinement(d::Vector{T},s::Solver) where T
    s.d_copy .= d
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)
    println("init res: $res_norm")
    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :fullspace
            s.Δ .= (s.H+Diagonal(s.δ))\s.res
        elseif s.opts.kkt_solve == :symmetric
            s.res_xL .+= s.res_zL./s.ΔxL
            s.res_xU .-= s.res_zU./s.ΔxU

            s.Δ_xy .= ma57_solve(s.LBL,s.res[s.idx.xy])
            s.Δ_zL .= -s.σL.*s.Δ_xL + s.res_zL./s.ΔxL
            s.Δ_zU .= s.σU.*s.Δ_xU + s.res_zU./s.ΔxU
        end

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end
    println("res: $res_norm")
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
    println("init res: $res_norm")
    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        # if s.opts.kkt_solve == :fullspace
        #     s.Δ .= (s.H+Diagonal(s.δ))\s.res
        # elseif s.opts.kkt_solve == :symmetric
        #     s.res_xL .+= s.res_zL./s.ΔxL
        #     s.res_xU .-= s.res_zU./s.ΔxU
        #
        #     s.Δ_xy .= ma57_solve(s.LBL,s.res[s.idx.xy])
        #     s.Δ_zL .= -s.σL.*s.Δ_xL + s.res_zL./s.ΔxL
        #     s.Δ_zU .= s.σU.*s.Δ_xU + s.res_zU./s.ΔxU
        # elseif s.opts.kkt_solve == :slack
        n = s.model_opt.n
        m = s.model_opt.m
        mI = s.model.mI
        mE = s.model.mE
        mA = s.model.mA
        nL = s.model_opt.nL
        nU = s.model_opt.nU
        idx = s.idx

        H = spzeros(n+m,n+m)
        r = zeros(n+m)

        view(H,1:n,1:n) .= s.H[1:n,1:n]
        view(H,CartesianIndex.(idx.xL[1:nL],idx.xL[1:nL])) .+= s.σL[1:nL]
        view(H,CartesianIndex.(idx.xU[1:nU],idx.xU[1:nU])) .+= s.σU[1:nU]

        view(H,1:n,n .+ (1:m)) .= s.H[1:n,idx.y]
        view(H,n .+ (1:m),1:n) .= s.H[idx.y,1:n]

        ΔxL = s.ΔxL[1:nL]
        ΔsL = s.ΔxL[nL .+ (1:mI)]
        ΔxU = s.ΔxU
        zL = s.zL[1:nL]
        zS = s.zL[nL .+ (1:mI)]
        zU = s.zU
        view(H,CartesianIndex.(n .+ (1:mI),n .+ (1:mI))) .= -ΔsL./zS
        view(H,CartesianIndex.(n+mI+mE .+ (1:mA),n+mI+mE .+ (1:mA))) .= -1.0/s.ρ

        hx = s.res[1:n]
        hs = s.res[get_s_idx(s)]
        hr = s.res[get_r_idx(s)]
        hyI = s.res[idx.y[1:mI]]
        hyE = s.res[idx.y[mI .+ (1:mE)]]
        hyA = s.res[idx.y[mI+mE .+ (1:mA)]]
        hzL = s.res[idx.zL[1:nL]]
        hzs = s.res[idx.zL[nL .+ (1:mI)]]
        hzU = s.res[idx.zU]

        r_tmp = zeros(n+m)
        r_tmp[1:n] .= copy(hx)
        r_tmp[idx.xL[1:nL]] .+= hzL./ΔxL
        r_tmp[idx.xU[1:nU]] .-= hzU./ΔxU
        r_tmp[n .+ (1:mI)] .= hyI + (ΔsL.*hs + hzs)./zS
        r_tmp[n+mI .+ (1:mE)] .= copy(hyE)
        r_tmp[n+mI+mE .+ (1:mA)] .= hyA + 1.0/s.ρ*hr

        idx_tmp = [(1:s.model_opt.n)...,s.idx.y...]

        Δ = -H\r_tmp

        Δx = Δ[1:n]
        ΔyI = Δ[n .+ (1:mI)]
        ΔyE = Δ[n+mI .+ (1:mE)]
        ΔyA = Δ[n+mI+mE .+ (1:mA)]

        Δr = 1.0/s.ρ*(ΔyA - hr)
        ΔzL = -(zL.*Δx[s.model_opt.xL_bool] + hzL)./ΔxL
        ΔzU = (zU.*Δx[s.model_opt.xU_bool] - hzU)./ΔxU
        Δzs = -ΔyI + hs
        Δs = -(ΔsL.*Δzs + hzs)./zS

        # end
        s.Δ .= [Δx;Δs;Δr;ΔyI;ΔyE;ΔyA;ΔzL;Δzs;ΔzU]

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end
    println("res: $res_norm")
    if res_norm < s.opts.ϵ_iterative_refinement
        return true
    else
        d .= s.d_copy
        return false
    end
end
