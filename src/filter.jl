
function check_filter(θ,φ,s::Solver)
    len = length(s.filter)
    cnt = 0
    # println("θ: $θ")
    # println("φ: $φ")
    for f in s.filter
        # println("f(θ,φ): ($(f[1]),$(f[2]))")
        if θ < f[1] || φ < f[2]
        # if θ < (1.0 - s.opts.γθ)*f[1] || φ < f[2] - s.opts.γφ*f[1]
            cnt += 1
        end
    end
    # println("cnt: $cnt")
    # println("len: $len")
    if cnt == len
        return true
    else
        return false
    end
end

function add_to_filter!(p,s::Solver)
    f = s.filter
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
    if s.update == :nominal
        x⁺ = s.x + s.α*s.d[1:s.n]
    elseif s.update == :soc
        x⁺ = s.x + s.α_soc*s.d_soc[1:s.n]
    else
        error("update error in augment filter")
    end
    θ⁺ = θ(x⁺,s)
    φ⁺ = barrier(x⁺,s)

    if !switching_condition(s) || !armijo(x⁺,s)
        add_to_filter!((θ⁺,φ⁺),s)
    end

    return nothing
end
