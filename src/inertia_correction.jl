function inertia_correction_hsl!(H,s::Solver,restoration=false)
    # LDL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
    # ma57_factorize(LDL)
    #
    # m = LDL.info.num_negative_eigs
    # n = LDL.info.rank - m
    # z = s.n+s.m-LDL.info.rank
    #
    # println("n: $n, m: $m, z: $z")
    #
    # if n == s.n && z == 0 && m == s.m
    #     return false
    # end
    #
    # if z != 0 && restoration==false
    #     println("$z zero eigen values")
    #     s.δc = s.opts.δc*s.μ^s.opts.κc
    # end
    #
    # if s.δw_last == 0.
    #     s.δw = s.opts.δw0
    # else
    #     s.δw = max(s.opts.δw_min,s.opts.κw⁻*s.δw_last)
    # end
    #
    # while n != s.n || z != 0 || m != s.m
    #     LDL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
    #     ma57_factorize(LDL)
    #
    #     m = LDL.info.num_negative_eigs
    #     n = LDL.info.rank - m
    #     z = s.n+s.m-LDL.info.rank
    #
    #     if n == s.n && z == 0 && m == s.m
    #         @warn "δw: $(s.δw)"
    #         break
    #     else
    #         if s.δw_last == 0
    #             s.δw = s.opts.κw⁺_*s.δw
    #         else
    #             s.δw = s.opts.κw⁺*s.δw
    #         end
    #     end
    #
    #     if s.δw > s.opts.δw_max
    #         println("n: $n, m: $m, z: $z")
    #         println("s.δw: $(s.δw)")
    #         error("inertia correction failure")
    #     end
    # end
    #
    # s.δw_last = s.δw

    s.δw = 0.1
    s.δc = 0.1

    return true
end
