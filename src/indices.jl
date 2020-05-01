"""
    Indices

Stores useful indices into the primal-dual vector `[x; y]` where `x` is the vector
of `n` primal variables and `y` is the vector of `m` dual variables.
"""
struct Indices
    x::UnitRange{Int}    # primal variables
    xL::Vector{Int}      # set of lower bounds on primals
    xU::Vector{Int}      # set of upper bounds on primals
    xLs::Vector{Int}     # set of lower bounds on slacks
    xUs::Vector{Int}     # set of upper bounds on slacks
    y::UnitRange{Int}    # dual variables
    y_al::Vector{Int}    # augmented Lagrangian dual variable estimates
    zL::UnitRange{Int}   # duals for slack lower bounds?
    zU::UnitRange{Int}   # duals for slack upper bounds?
    xy::UnitRange{Int}   # entire primal-dual vector
end

function indices(n,m,nL,nU,xL_bool,xU_bool,xLs_bool,xUs_bool,c_al_idx)
    x = 1:n
    xL = x[xL_bool]
    xU = x[xU_bool]
    xLs = x[xLs_bool]
    xUs = x[xUs_bool]
    y = n .+ (1:m)
    y_al = y[c_al_idx]
    zL = n + m .+ (1:nL)
    zU = n + m + nL .+ (1:nU)
    xy = 1:(n+m)

    Indices(x,xL,xU,xLs,xUs,y,y_al,zL,zU,xy)
end

struct RestorationIndices
    p::UnitRange{Int}
    n::UnitRange{Int}
    zL::UnitRange{Int}
    zp::UnitRange{Int}
    zn::UnitRange{Int}
    zU::UnitRange{Int}
    xy::Vector{Int}
end

function restoration_indices()
    p = 0:0
    n = 0:0
    zL = 0:0
    zp = 0:0
    zn = 0:0
    zU = 0:0
    xy = [0]

    RestorationIndices(p,n,zL,zp,zn,zU,xy)
end
