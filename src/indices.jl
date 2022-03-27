"""
    Indices

Stores useful indices into the primal-dual vector `[x; y]` where `x` is the vector
of `n` primal variables and `y` is the vector of `m` dual variables.
"""
struct Indices
    x::UnitRange{Int}    # primal variables
    s::UnitRange{Int}
    r::UnitRange{Int}
    xL::Vector{Int}      # set of lower bounds on primals
    xLs::Vector{Int}     # set of lower bounds on slacks
    y::UnitRange{Int}    # dual variables
    yI::Vector{Int}
    yE::Vector{Int}
    yA::Vector{Int}
    zL::UnitRange{Int}   # duals for slack lower bounds?
    xy::UnitRange{Int}   # entire primal-dual vector
end

function indices(model,model_opt)
    x = 1:model.n
    s = model_opt.n .+ (1:model_opt.mI)
    r = model_opt.n + model_opt.mI .+ (1:model_opt.mA)
    xL = x[model.xL_bool]
    xLs = x[model.xLs_bool]
    y = model.n .+ (1:model.m)
    yI = y[model.cI_idx]
    yE = y[model.cE_idx]
    yA = y[model.cA_idx]
    zL = model.n + model.m .+ (1:model.nL)
    xy = 1:(model.n+model.m)
    Indices(x,s,r,xL,xLs,y,yI,yE,yA,zL,xy)
end