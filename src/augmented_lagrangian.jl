function dual_update!(s::Solver)
    s.dual .+= s.penalty * s.xr
    return
end

function penalty_update!(s::Solver)
    s.penalty = 1.0 / s.central_path
    return 
end

function augmented_lagrangian_update!(s::Solver)
    dual_update!(s)
    penalty_update!(s)
    return
end
