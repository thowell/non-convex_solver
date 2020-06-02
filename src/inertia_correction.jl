"""
    inertia_correction!(s::Solver)

Implementation of Algorithm IC. Increase the regularization of the symmetric KKT system
until it has the correct inertia.
"""
function inertia_correction!(s::Solver; restoration=false)
    s.δw = 0.0
    s.δc = 0.0

    # println("INERTIA CORRECTION")

    # IC-1
    factorize_kkt!(s)
    # println("(n,m,z): ($(s.inertia.n)/$(s.model.n),$(s.inertia.m)/$(s.model.m),$(s.inertia.z)/0)")

    # println(length(s.qn.s) > 0 && s.qn.s[end])
    # println(length(s.qn.y) > 0 && s.qn.y[end])
    # println("first factorization")
    if inertia(s)
        # @logmsg InnerLoop "(n,m,z): ($(s.inertia.n)/$(s.model.n),$(s.inertia.m)/$(s.model.m),$(s.inertia.z)/0)"
        return nothing
    end

    # IC-2
    if s.inertia.z != 0
        s.opts.verbose ? (@warn "$(s.inertia.z) zero eigen values - rank deficient constraints") : nothing
        s.δc = s.opts.δc*s.μ^s.opts.κc
    end

    # IC-3
    if s.δw_last == 0.
        s.δw = s.opts.δw0
    else
        s.δw = max(s.opts.δw_min, s.opts.κw⁻*s.δw_last)
    end

    while !inertia(s)

        # IC-4
        factorize_kkt!(s)

        if inertia(s)
            @logmsg InnerLoop "(n,m,z)+: ($(s.inertia.n)/$(s.model.n),$(s.inertia.m)/$(s.model.m),$(s.inertia.z)/0)"
            break
        else
            # IC-5
            if s.δw_last == 0
                s.δw = s.opts.κw⁺_*s.δw
            else
                s.δw = s.opts.κw⁺*s.δw
            end
        end

        # IC-6
        if s.δw > s.opts.δw_max
            @logmsg InnerLoop "(n,m,z)+: ($(s.inertia.n)/$(s.model.n),$(s.inertia.m)/$(s.model.m),$(s.inertia.z)/0)"
            @logmsg InnerLoop "s.δw: $(s.δw)"
            # skp backtracking line search and go to restoration phase
            # TODO: handle inertia correction failure gracefully
            error("inertia correction failure")
        end
    end

    s.δw_last = s.δw

    return nothing
end

function inertia_correction_slack!(s::Solver; restoration=false)
    # s.δw = 0.0
    # s.δc = 0.0
    #
    # # IC-1
    # factorize_kkt_slack!(s)
    #
    # # s.opts.verbose ? println("inertia-> n: $(s.inertia_slack.n), m: $(s.inertia_slack.m), z: $(s.inertia_slack.z)") : nothing
    #
    # inertia_slack(s) && return nothing
    #
    # # IC-2
    # if s.inertia_slack.z != 0
    #     s.opts.verbose && @warn "$(s.inertia_slack.z) zero eigen values"
    #     s.δc = s.opts.δc*s.μ^s.opts.κc
    # end
    #
    # # IC-3
    # if s.δw_last == 0.
    #     s.δw = s.opts.δw0
    # else
    #     s.δw = max(s.opts.δw_min, s.opts.κw⁻*s.δw_last)
    # end
    #
    # while !inertia_slack(s)
    #
    #     # IC-4
    #     factorize_kkt_slack!(s)
    #
    #     if inertia_slack(s)
    #         @logmsg InnerLoop "(n,m,z)+: ($(s.inertia_slack.n)/$(s.model_opt.n),$(s.inertia_slack.m)/$(s.model.m),$(s.inertia_slack.z)/0)"
    #         break
    #     else
    #         # IC-5
    #         if s.δw_last == 0
    #             s.δw = s.opts.κw⁺_*s.δw
    #         else
    #             s.δw = s.opts.κw⁺*s.δw
    #         end
    #     end
    #
    #     # IC-6
    #     if s.δw > s.opts.δw_max
    #         if s.opts.verbose
    #             @logmsg InnerLoop "(n,m,z)+: ($(s.inertia_slack.n)/$(s.model_opt.n),$(s.inertia_slack.m)/$(s.model.m),$(s.inertia_slack.z)/0)"
    #             @logmsg InnerLoop "s.δw: $(s.δw)"
    #         end
    #         # skp backtracking line search and go to restoration phase
    #         # TODO: handle inertia correction failure gracefully
    #         error("inertia correction failure")
    #     end
    # end
    #
    # s.δw_last = s.δw
    kkt_hessian_symmetric!(s)
    inertia_correction!(s,restoration=s.restoration)
    factorize_kkt_slack!(s)
    return nothing
end

"""
    factorize_kkt!(s::Solver)

Compute the LDL factorization of the symmetric KKT matrix and update the inertia values.
Uses the Ma57 algorithm from HSL.
"""
function factorize_kkt!(s::Solver)
    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.y] .= -s.δc

    s.σL .= s.zL./(s.ΔxL .- s.δc)
    s.σU .= s.zU./(s.ΔxU .- s.δc)

    kkt_hessian_symmetric!(s)

    s.LBL = Ma57(s.H_sym + Diagonal(view(s.δ,s.idx.xy)))
    ma57_factorize(s.LBL)

    s.inertia.m = s.LBL.info.num_negative_eigs
    s.inertia.n = s.LBL.info.rank - s.inertia.m
    s.inertia.z = s.model.n+s.model.m - s.LBL.info.rank

    return nothing
end

function factorize_kkt_slack!(s::Solver)
    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.y] .= -s.δc

    s.σL .= s.zL./(s.ΔxL .- s.δc)
    s.σU .= s.zU./(s.ΔxU .- s.δc)

    kkt_hessian_slack!(s)

    s.LBL_slack = Ma57(s.H_slack + Diagonal(view(s.δ,[(1:s.model_opt.n)...,s.idx.y...])))
    ma57_factorize(s.LBL_slack)

    s.inertia_slack.m = s.LBL_slack.info.num_negative_eigs
    s.inertia_slack.n = s.LBL_slack.info.rank - s.inertia_slack.m
    s.inertia_slack.z = s.model_opt.n+s.model_opt.m - s.LBL_slack.info.rank

    return nothing
end

"""
    inertia(s::Solver)

Check if the inertia of the symmetric KKT system is correct. The inertia is defined as the
    tuple `(n,m,z)` where
- `n` is the number of postive eigenvalues. Should be equal to the number of primal variables.
- `m` is the number of negative eignvalues. Should be equal to the number of dual variables
- `z` is the number of zero eigenvalues. Should be 0.
"""
inertia(s::Solver) = (s.inertia.n == s.model.n
                        && s.inertia.m == s.model.m
                        && s.inertia.z == 0)

inertia_slack(s::Solver) = (s.inertia_slack.n == s.model_opt.n
                        && s.inertia_slack.m == s.model_opt.m
                        && s.inertia_slack.z == 0)
