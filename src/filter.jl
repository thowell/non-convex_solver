
function check_filter(θ,φ,s::Solver)
    len = length(s.filter)
    cnt = 0

    for f in s.filter
        if θ < f[1] || φ < f[2]
        # if θ < (1.0 - s.opts.γθ)*f[1] || φ < f[2] - s.opts.γφ*f[1]
            cnt += 1
        end
    end

    if cnt == len
        return true
    else
        return false
    end
end

function add_to_filter!(p,f)
    if isempty(f)
        push!(f,p)
        return nothing
    end

    # check that new point is not dominated
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

function augment_filter!(s::Solver)
    θ⁺ = θ(s.x⁺,s)
    φ⁺ = barrier(s.x⁺,s)

    if !switching_condition(s) || !armijo(s.x⁺,s)
        add_to_filter!((θ⁺,φ⁺),s.filter)
    end

    return nothing
end
