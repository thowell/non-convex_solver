using LinearAlgebra
using ForwardDiff
using Plots

nx = 3
m = 2
nl = 2

x0 = [-2.;3.;1.]
xl = [-Inf;0.;0.]
xu = [Inf; Inf; Inf]

ρ = [1000.;1000.;1000.;1000.]
# ρ = 1000.0*rand(2m)
DR = Diagonal(1.0./abs.(x0))
#
f(x,p,n,μ) = ρ'*[p;n] + 0.5*sqrt(μ)*norm(DR*(x-x0))^2
barrier(x,p,n,μ) = f(x,p,n,μ) - μ*sum(log.(x[2:3])) - μ*sum(log.(p)) - μ*sum(log.(n))
∇barrier(x,p,n,μ) = [sqrt(μ)*DR'*DR*(x-x0) - μ*[0.;1.0./x[2:3]];
                     ρ[1:m] - μ./p;
                     ρ[m .+ (1:m)] - μ./n]
# ∇barrier(x0,p0,n0,μ0)
# ∇f(x,p,n,μ) = [sqrt(μ)*DR'*DR*(x-x0);ρ*ones(2m)]

c(x) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
∇c(x) = ForwardDiff.jacobian(c,x)

c_slack(x,p,n) = c(x) - p + n

function kkt(x,p,n,λ,zl,zp,zn,μ)
    [sqrt(μ)*DR'*DR*(x-x0) + ∇c(x)'*λ - [0.;zl];
     ρ[1:m] .- zp - λ;
     ρ[m .+ (1:m)] .- zn + λ;
     c(x) - p + n;
     x[2:3].*zl .- μ;
     p.*zp .- μ;
     n.*zn .- μ]
end

p0 = ones(m)
n0 = ones(m)
λ0 = zeros(m)
zl0 = zeros(2)
zp0 = ones(m)
zn0 = ones(m)
μ0 = 1.0
_kkt = kkt(x0,p0,n0,λ0,zl0,zp0,zn0,μ0)

function dkkt(x,p,n,λ,zl,zp,zn,μ)
    ∇²cλ(x) = ∇c(x)'*λ

    [
     (sqrt(μ)*DR'*DR + ForwardDiff.jacobian(∇²cλ,x)) zeros(nx,2m) ∇c(x)' [zeros(1,nl); -1.0*Matrix(I,nl,nl)] zeros(nx,2m);
     zeros(2m,nx+2m) [-1.0*Matrix(I,m,m); Matrix(I,m,m)] zeros(2m,nl) -1.0*Matrix(I,2m,2m);
     ∇c(x) -1.0*Matrix(I,m,m) 1.0*Matrix(I,m,m) zeros(m,m+nl+2m);
     zeros(nl,1) Diagonal(zl) zeros(nl,2m+m) Diagonal(x[2:3]) zeros(nl,2m);
     zeros(m,nx) Diagonal(zp) zeros(m,m+m+nl) Diagonal(p) zeros(m,m);
     zeros(m,nx+m) Diagonal(zn) zeros(m,m+nl+m) Diagonal(n);
     ]
end

_dkkt = dkkt(x0,p0,n0,λ0,zl0,zp0,zn0,μ0)
rank(_dkkt)

function check_filter(θ,φ,fil)
    len = length(fil)
    cnt = 0

    for f in fil
        if θ < f[1] || φ < f[2]
            cnt += 1
        end
    end

    if cnt == len
        return true
    else
        return false
    end
end

function add_to_filter!(p,fil)
    f = fil
    if isempty(f)
        push!(f,p)
        return nothing
    end

    # check that new point is not dominated
    len = length(f)
    for _p in f
        if p[1] >= _p[1] && p[2] >= _p[2]
            len -= 1
        end
    end

    # remove filter's points dominated by new point
    if length(f) == len
        # _f = copy(f)
        # empty!(f)
        push!(f,p)
        # for _p in _f
        #     if !(_p[1] >= p[1] && _p[2] >= p[2])
        #         push!(f,_p)
        #     end
        # end
    end
    return nothing
end

function init_x0(x,xl,xu,κ1,κ2)
    pl = min(κ1*max(1.0,abs(xl)),κ2*(xu-xl))
    pu = min(κ1*max(1.0,abs(xu)),κ2*(xu-xl))

    # projection
    if x < xl+pl
        x = xl+pl
    elseif x > xu-pu
        x = xu-pu
    end
    return x
end

function primaldual(x0;verbose=false,max_iter=35)
    iter = 0
    iter_outer = 0

    γθ = 1.0e-5
    γφ = 1.0e-5
    sθ = 1.1
    sφ = 2.3
    ηφ = 1.0e-4
    τ_min = 0.99

    x = copy(x0)
    for i = 1:nx
        x[i] = init_x0(x0[i],xl[i],xu[i],1.0e-2,1.0e-2)
    end

    _c = c(x)

    μ = 0.1
    τ = max(τ_min,1.0-μ)

    n = zeros(m)
    p = zeros(m)
    for i = 1:m
        n[i] = (μ - ρ[i]*_c[i])/(2.0*ρ[i]) + sqrt(((μ-ρ[i]*_c[i])/(2.0*ρ[i]))^2 + (μ*_c[i])/(2.0*ρ[i]))
        p[i] = _c[i] + n[i]
    end
    λ = zeros(m)
    zl = ones(nl)
    zp = sqrt(μ)./p
    zn = sqrt(μ)./n

    r = 0.
    θ = norm(c_slack(x,p,n),Inf)
    φ = barrier(x,p,n,μ)
    ∇φ = ∇barrier(x,p,n,μ)

    θ_min = 1.0e-4*max(1.0,θ)
    θ_max = 1.0e4*max(1.0,θ)

    fil = Tuple[]
    push!(fil,(θ_max,Inf))

    while μ > 1e-8
        r = kkt(x,p,n,λ,zl,zp,zn,μ)
        while norm(r,Inf) > 1e-8
            δ = -dkkt(x,p,n,λ,zl,zp,zn,μ)\r
            α = 1.
            αz = 1.
            while any(zl + αz*δ[(nx + m + m + m) .+ (1:nl)] .< (1-τ)*zl) ||
                    any(zp + αz*δ[(nx + m + m + m + nl) .+ (1:m)] .< (1-τ)*zp) ||
                    any(zn + αz*δ[(nx + m + m + m + nl + m) .+ (1:m)] .< (1-τ)*zn)
                αz = αz/2
                iter += 1

                if iter > max_iter
                    error("feasibility failed")
                end
                iter += 1

                if iter > max_iter
                    error("feasibility failed")
                end
            end

            while any(x[2:3] + α*δ[2:3] .< (1-τ)*x[2:3]) ||
                  any(p + α*δ[nx .+ (1:m)] .< (1-τ)*p) ||
                  any(n + α*δ[(nx + m) .+ (1:m)] .< (1-τ)*n)

                α = α/2
                iter += 1

                if iter > max_iter
                    error("feasibility failed")
                end
            end

            if θ < θ_min && ∇φ'*δ[1:(nx+2m)] < 0 && α*(-∇φ'*δ[1:(nx+2m)])^sφ > θ^sθ
                @warn "θ_min threshold"
                while !check_filter(norm(c_slack(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)]),1),barrier(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)],μ),fil) && barrier(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)],μ) > φ + ηφ*α*∇φ'*δ[1:(nx+2m)]
                    α = α/2

                    iter += 1

                    if iter > max_iter
                        error("line search failed")
                    end
                end
                # error("huh?")
            else
                while !check_filter(norm(c_slack(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)]),1),barrier(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)],μ),fil) && norm(c_slack(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)]),1) > (1-γθ)*θ && barrier(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)],μ) > φ - γφ*θ
                    α = α/2
                    iter += 1

                    if iter > max_iter
                        error("line search failed")
                    end
                end
                add_to_filter!((norm(c_slack(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)]),1),barrier(x + α*δ[1:3],p + α*δ[nx .+ (1:m)],n + α*δ[(nx + m) .+ (1:m)],μ)),fil)
            end

            iter_outer += 1

            x .+= α*δ[1:nx]
            p .+= α*δ[nx .+ (1:m)]
            n .+= α*δ[(nx + m) .+ (1:m)]
            λ .+= α*δ[(nx + m + m) .+ (1:m)]
            zl .+= αz*δ[(nx + m + m + m) .+ (1:nl)]
            zp .+= αz*δ[(nx + m + m + m + nl) .+ (1:m)]
            zn .+= αz*δ[(nx + m + m + m + nl + m) .+ (1:m)]

            r = kkt(x,p,n,λ,zl,zp,zn,μ)
            θ = norm(c_slack(x,p,n),1)
            φ = barrier(x,p,n,μ)
            ∇φ = ∇barrier(x,p,n,μ)

            # add_to_filter!((θ,φ),fil)

            println("iter: $iter_outer")
            println("α = $(α)")
            println("res = $(norm(r,Inf))")
        end
        μ = 0.1*μ
        τ = max(τ_min,1.0-μ)
        println("μ: $μ")
    end

    return x, p, n, fil
end

@time x1, p1, n1, fil1 = primaldual(x0,verbose=false,max_iter=100)

c_slack(x1,p1,n1)
println("x: $x1")
println("p: $p1")
println("n: $n1")

@time x2, p2, n2, fil2 = primaldual(x1,verbose=false,max_iter=100)

c_slack(x2,p2,n2)
println("x: $x2")
println("p: $p2")
println("n: $n2")
