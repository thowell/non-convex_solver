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
    xU::Vector{Int}      # set of upper bounds on primals
    xLs::Vector{Int}     # set of lower bounds on slacks
    xUs::Vector{Int}     # set of upper bounds on slacks
    y::UnitRange{Int}    # dual variables
    yI::Vector{Int}
    yE::Vector{Int}
    yA::Vector{Int}
    zL::UnitRange{Int}   # duals for slack lower bounds?
    zU::UnitRange{Int}   # duals for slack upper bounds?
    xy::UnitRange{Int}   # entire primal-dual vector
end

function indices(model,model_opt)
    x = 1:model.n
    s = model_opt.n .+ (1:model_opt.mI)
    r = model_opt.n + model_opt.mI .+ (1:model_opt.mA)
    xL = x[model.xL_bool]
    xU = x[model.xU_bool]
    xLs = x[model.xLs_bool]
    xUs = x[model.xUs_bool]
    y = model.n .+ (1:model.m)
    yI = y[model.cI_idx]
    yE = y[model.cE_idx]
    yA = y[model.cA_idx]
    zL = model.n + model.m .+ (1:model.nL)
    zU = model.n + model.m + model.nL .+ (1:model.nU)
    xy = 1:(model.n+model.m)
    Indices(x,s,r,xL,xU,xLs,xUs,y,yI,yE,yA,zL,zU,xy)
end