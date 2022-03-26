# TODO: add reference
objective_gradient_scaling(scaling_tolerance, ∇f) = min(1.0,scaling_tolerance/norm(∇f,Inf))

function constraint_scaling(scaling_tolerance, ∇c, m)
    Dc = spzeros(m, m)
    for j = 1:m
        Dc[j, j] = min(1.0, scaling_tolerance / norm(∇c[j,:], Inf))
    end
    return Dc
end
