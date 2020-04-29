
"""
    check_filter(θ, φ, s::Solver)

Check if the constraint residual `θ` and the barrier objective `φ` are accepted by the filter.
To be accepted the pair must be less than all the pairs stored in the filter.
"""
function check_filter(θ,φ,s::Solver)
    len = length(s.filter)
    cnt = 0

    # TODO: wouldn't it be more efficient to return false as soon as a point is found that
    # fails the test, so you don't need to check all of them? Especially if there's some loose
    # ordering in the list of pairs?
    for f in s.filter
        if θ < f[1] || φ < f[2]
            cnt += 1
        end
    end

    if cnt == len
        return true
    else
        return false
    end
end

"""
    add_to_filter!(p, f)

Add the pair `p` to the filter `f` (a vector of pairs)
"""
function add_to_filter!(p,f)
    if isempty(f)
        push!(f,p)
        return nothing
    end

    # check that new point is not dominated
    # TODO: replace this will a call to `check_filter`?
    len = length(f)
    for _p in f
        if p[1] >= _p[1] && p[2] >= _p[2]
            len -= 1
        end
    end

    # remove filter's points dominated by new point
    if length(f) == len
        _f = copy(f)
        empty!(f)
        push!(f,p)
        for _p in _f
            if !(_p[1] >= p[1] && _p[2] >= p[2])
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
        add_to_filter!(((1.0-s.opts.γθ)*s.θ, s.φ - s.opts.γφ*s.θ), s.filter)
    end
    return nothing
end
