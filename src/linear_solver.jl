abstract type LinearSolver end

mutable struct Inertia
    n::Int  # number of positve eigenvalues
    m::Int  # number of negative eigenvalues
    z::Int  # number of zero eigenvalues
end

mutable struct MA57Solver{T} <: LinearSolver
    LBL::Ma57{T}
    inertia::Inertia
end

function factorize!(ls::MA57Solver,H)
    ls.LBL = Ma57(H)
    ma57_factorize(ls.LBL)
    return nothing
end

function compute_inertia!(ls::MA57Solver,s)
    ls.inertia.m = ls.LBL.info.num_negative_eigs
    ls.inertia.n = ls.LBL.info.rank - ls.inertia.m
    ls.inertia.z = s.model.n+s.model.m - ls.LBL.info.rank
    return nothing
end

function regularization_init!(::MA57Solver,s)
    s.δw = 0.0
    s.δc = 0.0
    return nothing
end

function solve!(ls::MA57Solver,d,h)
    d .= ma57_solve(ls.LBL,h)
    return nothing
end

mutable struct QDLDLSolver <: LinearSolver
    F
    inertia::Inertia
end
