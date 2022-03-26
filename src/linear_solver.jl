abstract type LinearSolver end

mutable struct Inertia
    positive::Int  # number of positve eigenvalues
    negative::Int  # number of negative eigenvalues
    zero::Int  # number of zero eigenvalues
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
    ls.inertia.positive = count(ls.F.workspace.D .<= 0.0)
    ls.inertia.negative = ls.F.workspace.positive_inertia.x
    ls.inertia.zero = count(ls.F.workspace.D .== 0.0)
    return nothing
end

function initialize_regularization!(::QDLDLSolver,s)
    s.primal_regularization = 1.0e-7
    s.dual_regularization = 1.0e-7
    return nothing
end

function solve!(ls::QDLDLSolver,d,h)
    d .= QDLDL.solve(ls.F,h)
    return nothing
end