"""
    inertia_correction!(s::Solver)

Implementation of Algorithm IC. Increase the regularization of the symmetric KKT system
until it has the correct inertia.
"""
function inertia_correction!(s::Solver; restoration=false)
    # s.δw = 0.0
    # s.δc = 0.0
    regularization_init!(s.linear_solver,s)

    # IC-1
    factorize_regularized_kkt!(s)
    # println("(n,m,z): ($(s.inertia.n)/$(s.model.n),$(s.inertia.m)/$(s.model.m),$(s.inertia.z)/0)")

    if inertia(s)
        # @logmsg InnerLoop "(n,m,z): ($(s.inertia.n)/$(s.model.n),$(s.inertia.m)/$(s.model.m),$(s.inertia.z)/0)"
        return nothing
    end

    # IC-2
    if s.linear_solver.inertia.z != 0
        s.opts.verbose ? (@warn "$(s.linear_solver.inertia.z) zero eigen values - rank deficient constraints") : nothing
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
        factorize_regularized_kkt!(s)

        if inertia(s)
            @logmsg InnerLoop "(n,m,z)+: ($(s.linear_solver.n)/$(s.model.n),$(s.linear_solver.m)/$(s.model.m),$(s.linear_solver.z)/0)"
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
            @logmsg InnerLoop "(n,m,z)+: ($(s.linear_solver.n)/$(s.model.n),$(s.linear_solver.m)/$(s.model.m),$(s.linear_solver.z)/0)"
            @logmsg InnerLoop "s.δw: $(s.δw)"
            # TODO: handle inertia correction failure gracefully
            error("inertia correction failure")
        end
    end

    s.δw_last = s.δw

    return nothing
end

"""
    factorize_kkt!(s::Solver)

Compute the LDL factorization of the symmetric KKT matrix and update the inertia values.
Uses the Ma57 algorithm from HSL.
"""
function factorize_regularized_kkt!(s::Solver)
    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.y] .= -s.δc

    s.σL .= s.zL./(s.ΔxL .- s.δc)
    s.σU .= s.zU./(s.ΔxU .- s.δc)

    kkt_hessian_symmetric!(s)

    # s.LBL = Ma57(s.H_sym + Diagonal(view(s.δ,s.idx.xy)))
    # ma57_factorize(s.LBL)
    factorize!(s.linear_solver,s.H_sym + Diagonal(view(s.δ,s.idx.xy)))

    # s.inertia.m = s.LBL.info.num_negative_eigs
    # s.inertia.n = s.LBL.info.rank - s.inertia.m
    # s.inertia.z = s.model.n+s.model.m - s.LBL.info.rank
    compute_inertia!(s.linear_solver,s)

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
inertia(s::Solver) = (s.linear_solver.inertia.n == s.model.n
                        && s.linear_solver.inertia.m == s.model.m
                        && s.linear_solver.inertia.z == 0)
