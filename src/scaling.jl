objective_gradient_scaling(g_max,∇f) = min(1.0,g_max/norm(∇f,Inf))

function constraint_scaling!(Dc,g_max,∇c,m)
    for j = 1:m
        Dc[j,j] = min(1.0,g_max/norm(∇c[j,:],Inf))
    end
end

function constraint_scaling(g_max,∇c,m)
    Dc = spzeros(m,m)
    constraint_scaling!(Dc,g_max,∇c,m)
    return Dc
end
