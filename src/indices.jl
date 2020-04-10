struct Indices
    x
    xL
    xU
    xLs
    xUs
    λ
    zL
    zU
    xλ
end

function indices(n,m,nL,nU,xL_bool,xU_bool,xLs_bool,xUs_bool)
    x = 1:n
    xL = x[xL_bool]
    xU = x[xU_bool]
    xLs = x[xLs_bool]
    xUs = x[xUs_bool]
    λ = n .+ (1:m)
    zL = n + m .+ (1:nL)
    zU = n + m + nL .+ (1:nU)
    xλ = 1:(n+m)

    Indices(x,xL,xU,xLs,xUs,λ,zL,zU,xλ)
end
