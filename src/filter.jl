
"""
    check_filter(constraint_violation, merit, f)

Check if the constraint residual `constraint_violation` and the barrier objective `merit` are accepted by the filter.
To be accepted, the pair must be acceptable to each pair stored in the filter.
"""
function check_filter(constraint_violation,merit,f)
    for _f in f
        if !(constraint_violation < _f[1] || merit < _f[2])
            return false
        end
    end
    return true
end

"""
    augment_filter!(constraint_violation,merit,f)

Add the pair `(constraint_violation,merit)` to the filter `f` (a vector of pairs)
"""
function augment_filter!(constraint_violation,merit,f)
    if isempty(f)
        push!(f,(constraint_violation,merit))
        return nothing
    # remove filter points dominated by new point
    elseif check_filter(constraint_violation,merit,f)
        _f = copy(f)
        empty!(f)
        push!(f,(constraint_violation,merit))
        for _p in _f
            if !(_p[1] >= constraint_violation && _p[2] >= merit)
                push!(f,_p)
            end
        end
    end
    return nothing
end

"""
    augment_filter!(s::Solver)

Check current step, and add to the filter if necessary, adding some padding to the points
to ensure sufficient decrease (Eq. 18).
"""
function augment_filter!(s::Solver)
    if !switching_condition(s) || !armijo(s)
        augment_filter!((1.0-s.options.constraint_violation_tolerance)*s.constraint_violation, s.merit - s.options.merit_tolerance*s.constraint_violation, s.filter)
    end
    return nothing
end

