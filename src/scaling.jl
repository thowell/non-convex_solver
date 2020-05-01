"""
    init_sd(y, z, n, m, s_max)

Calculate the scaling parameter for the dual variables
"""
function init_sd(y,z,n,m,s_max)
    sd = max(s_max,(norm(y,1) + norm(z,1))/(n+m))/s_max
    return sd
end

"""
    init_sc(z, n, s+max)

Calculate the scaling parameter for the constraints
"""
function init_sc(z,n,s_max)
    sc = max(s_max,norm(z,1)/n)/s_max
    return sc
end

# QUESTION: why not just use `sparse(I,n,n)?`
# QUESTION: where are these used?
function init_Dx!(Dx,n)
    for i = 1:n
        Dx[i,i] = 1.0
    end
    return nothing
end

function init_Dx(n)
    Dx = spzeros(n,n)
    init_Dx!(Dx,n)
    return Dx
end


init_df(g_max,∇f) = min(1.0,g_max/norm(∇f,Inf))

function init_Dc!(Dc,g_max,∇c,m)
    for j = 1:m
        Dc[j,j] = min(1.0,g_max/norm(∇c[j,:],Inf))
    end
end

function init_Dc(g_max,∇c,m)
    Dc = spzeros(m,m)
    init_Dc!(Dc,g_max,∇c,m)
    return Dc
end
