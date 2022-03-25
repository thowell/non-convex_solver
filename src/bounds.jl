function bounds_mask(xL,xU,max_bound)
    n = length(xL)
    xL_bool = zeros(Bool,n)
    xU_bool = zeros(Bool,n)
    xLs_bool = zeros(Bool,n)
    xUs_bool = zeros(Bool,n)

    for i = 1:n
        # boolean bounds
        if xL[i] < -1.0*max_bound
            xL_bool[i] = 0
        else
            xL_bool[i] = 1
        end

        if xU[i] > max_bound
            xU_bool[i] = 0
        else
            xU_bool[i] = 1
        end

        # single bounds
        if xL_bool[i] == 1 && xU_bool[i] == 0
            xLs_bool[i] = 1
        else
            xLs_bool[i] = 0
        end

        if xL_bool[i] == 0 && xU_bool[i] == 1
            xUs_bool[i] = 1
        else
            xUs_bool[i] = 0
        end
    end
    return xL_bool,xU_bool,xLs_bool,xUs_bool
end

function relax_bounds!(xL,xU,xL_bool,xU_bool,n,ϵ)
   for i in (1:n)[xL_bool]
       xL[i] = relax_bound(xL[i],ϵ,:L)
   end
   for i in (1:n)[xU_bool]
       xU[i] = relax_bound(xU[i],ϵ,:U)
   end
   return nothing
end

"""
    relax_bound(x_bnd, ϵ, bnd_type)

Relax the bound constraint `x_bnd` by ϵ, where `x_bnd` is a scalar. `bnd_type` is either
`:L` for lower bounds or `:U` for upper bounds.
"""
function relax_bound(x_bnd, ϵ, bnd_type)
    if bnd_type == :L
        return x_bnd - ϵ*max(1.0,abs(x_bnd))
    elseif bnd_type == :U
        return x_bnd + ϵ*max(1.0,abs(x_bnd))
    else
        error("bound type error")
    end
end
