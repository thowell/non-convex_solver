abstract type QuasiNewton end

mutable struct BFGS <: QuasiNewton
    B
    s
    y

    x_prev
    ∇f_prev
    ∇c_prev

    status
end

function BFGS(;n=0,m=0)
    BFGS(sparse(1.0*I,n,n),[],[],zeros(n),zeros(n),zeros(m,n),:init)
end

function update_bfgs!(bfgs,x,y,zL,zU,idx_xL,idx_xU,∇f,∇c; init=false, update=:lagrangian, x_update=false, ∇L_update=false)

    x_update && push!(bfgs.s,x - bfgs.x_prev)
    if ∇L_update
        if update == :lagrangian
            ∇L⁺ = ∇f + ∇c'*y
            ∇L⁺[idx_xL] -= zL
            ∇L⁺[idx_xU] += zU

            ∇L = bfgs.∇f_prev + bfgs.∇c_prev'*y
            ∇L[idx_xL] -= zL
            ∇L[idx_xU] += zU
            push!(bfgs.y,∇L⁺ - ∇L)
        end
    end
    bfgs.x_prev.= copy(x)
    bfgs.∇f_prev .= copy(∇f)
    bfgs.∇c_prev .= copy(∇c)

    return nothing
end

function get_B(bfgs::BFGS)

    if length(bfgs.s) > 0
        s = bfgs.s[end]
        y = bfgs.y[end]
        ρ = s'*y

        if isfinite(ρ) && ρ > 0.

            #damped BFGS
            if ρ >= 0.2*s'*bfgs.B*s
                θ = 1.0
            else
                θ = (0.8*s'*bfgs.B*s)/(s'*bfgs.B*s - s'*y)
            end
            θ = 1.0
            r = θ*y + (1.0 - θ)*bfgs.B*s
            if length(bfgs.s) == 1
                bfgs.B .= (r'*r)/(s'*r)*bfgs.B - (bfgs.B*s*s'*bfgs.B)/(s'*bfgs.B*s) + (r*r')/(s'*r)
            else
                bfgs.B .= bfgs.B - (bfgs.B*s*s'*bfgs.B)/(s'*bfgs.B*s) + (r*r')/(s'*r)
            end
        end
    end
    # println("B pos. def. check: $(isposdef(Hermitian(bfgs.B)))")
    return bfgs.B
end

function reset_bfgs!(bfgs)
    bfgs.B .= sparse(1.0*I,size(bfgs.B))
    bfgs.s .= [bfgs.s[end]]
    bfgs.y .= [bfgs.y[end]]
    return nothing
end
