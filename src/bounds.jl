function bounds_mask(xL,max_bound)
    n = length(xL)
    xL_bool = zeros(Bool,n)
    xLs_bool = zeros(Bool,n)

    for i = 1:n
        # boolean bounds
        if xL[i] < -1.0*max_bound
            xL_bool[i] = 0
        else
            xL_bool[i] = 1
        end

       

        # single bounds
        if xL_bool[i] == 1 
            xLs_bool[i] = 1
        else
            xLs_bool[i] = 0
        end

    end
    return xL_bool,xLs_bool
end

function relax_bounds!(xL,xL_bool,n,ϵ)
   for i in (1:n)[xL_bool]
       xL[i] = relax_bound(xL[i],ϵ,:L)
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
        return x_bnd - ϵ*max(1.0, abs(x_bnd))
    end
end
