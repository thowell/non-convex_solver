abstract type LinearSolver end

mutable struct Inertia
    n::Int  # number of positve eigenvalues
    m::Int  # number of negative eigenvalues
    z::Int  # number of zero eigenvalues
end

# HSL MA57
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

function regularization_init(::MA57Solver)
    δw = 0.0
    δc = 0.0
    return δw, δc
end

function solve!(ls::MA57Solver,d,h)
    d .= ma57_solve(ls.LBL,h)
    return nothing
end

# QDLDL
mutable struct QDLDLSolver <: LinearSolver
    F
    inertia::Inertia
end

function factorize!(ls::QDLDLSolver,H)
    try
        ls.F = qdldl(H)
    catch
        @warn "QDLDL factorization failure"
    end
    return nothing
end

function compute_inertia!(ls::QDLDLSolver,s)
    ls.inertia.m = count(ls.F.workspace.D .<= 0.0)
    ls.inertia.n = ls.F.workspace.positive_inertia.x
    ls.inertia.z = count(ls.F.workspace.D .== 0.0)
    return nothing
end

function regularization_init!(::QDLDLSolver,s)
    s.δw = 1.0e-7
    s.δc = 1.0e-7
    return nothing
end

function regularization_init(::QDLDLSolver)
    δw = 1.0e-7
    δc = 1.0e-7
    return δw, δc
end

function solve!(ls::QDLDLSolver,d,h)
    d .= QDLDL.solve(ls.F,h)
    return nothing
end

# PARDISO
mutable struct PARDISOSolver <: LinearSolver
    ps
    A_pardiso
    inertia::Inertia
end

function factorize!(ls::PARDISOSolver,H)
    ls.A_pardiso .= copy(H)
    return nothing
end

function compute_inertia!(ls::PARDISOSolver,s)
    e = eigen(Array(ls.A_pardiso))
    ls.inertia.m = count(e.values .< 0.0)
    ls.inertia.n = count(e.values .> 0.0)
    ls.inertia.z = count(e.values .== 0.0) # TODO get rank info
    return nothing
end

function regularization_init!(::PARDISOSolver,s)
    s.δw = 0.0
    s.δc = 0.0
    return nothing
end

function regularization_init(::PARDISOSolver)
    δw = 0.0
    δc = 0.0
    return δw, δc
end

function solve!(ls::PARDISOSolver,d,h)
    pardiso(ls.ps,d,ls.A_pardiso,h)
    return nothing
end
