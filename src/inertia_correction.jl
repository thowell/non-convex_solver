function inertia_correction!(s::Solver)
    s.δw = 0.1
    s.δc = 0.1
    return true
end
