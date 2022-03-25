abstract type LinearSolver end

mutable struct Inertia
    n::Int  # number of positve eigenvalues
    m::Int  # number of negative eigenvalues
    z::Int  # number of zero eigenvalues
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