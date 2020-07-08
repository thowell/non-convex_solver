abstract type QuasiNewton end

function reset_quasi_newton!(qn::QuasiNewton)
    @warn "quasi newton reset"
    qn.B .= sparse(1.0*I,size(qn.B))
    empty!(qn.s)
    empty!(qn.y)
    qn.fail_cnt = 0
    return nothing
end

mutable struct BFGS <: QuasiNewton
    s
    y

    x_prev
    ∇f_prev
    ∇c_prev
    ∇L_prev

    ∇f
    ∇L

    B

    fail_cnt
end

function BFGS(;n=0,m=0)
    BFGS([],[],zeros(n),zeros(n),zeros(m,n),zeros(n),zeros(n),zeros(n),sparse(1.0*I,n,n),0)
end

function update_quasi_newton!(qn::BFGS,x; max_reset_cnt=100)

    # s update
    s = x - qn.x_prev
    push!(qn.s,s)
    qn.x_prev .= copy(x)

    # y update (damped)
    y = qn.∇L - qn.∇L_prev

    ρ = s'*y
    if ρ >= 0.2*s'*qn.B*s
        θ = 1.0
        qn.fail_cnt = 0
    else
        θ = (0.8*s'*qn.B*s)/(s'*qn.B*s - ρ)
        qn.fail_cnt += 1
    end
    y = θ*y + (1.0 - θ)*qn.B*s

    push!(qn.y,y)

    if qn.fail_cnt > max_reset_cnt
        reset_quasi_newton!(qn)
    end

    return nothing
end

function get_B(qn::BFGS)
    if length(qn.s) > 0
        s = qn.s[end]
        y = qn.y[end]

        δ = length(qn.s) == 1 ? (y'*y)/(s'*y) : 1.0

        qn.B .= δ*qn.B - (qn.B*s*s'*qn.B)/(s'*qn.B*s) + (y*y')/(s'*y)
    end

    return qn.B
end

# L-BFGS
mutable struct LBFGS <: QuasiNewton
    s
    y
    δ
    k

    A1
    A2
    L
    D

    x_prev
    ∇f_prev
    ∇c_prev
    ∇L_prev

    ∇f
    ∇L

    B

    fail_cnt
end

function LBFGS(;n=0,m=0,k=0)
    LBFGS([],[],0,k,zeros(n,2k),zeros(2k,2k),zeros(k,k),zeros(k,k),
        zeros(n),zeros(n),zeros(m,n),zeros(n),zeros(n),zeros(n),sparse(1.0*I,n,n),0)
end

# qn = LBFGS(n=100,m=10,k=5)

function update_quasi_newton!(qn::LBFGS,x; max_reset_cnt=2)

    # s update
    s = x - qn.x_prev
    push!(qn.s,s)
    qn.x_prev .= copy(x)

    # y update (damped)
    y = qn.∇L - qn.∇L_prev

    ρ = s'*y
    flag = false
    if ρ >= 0.0#0.2*s'*qn.B*s
        # θ = 1.0
        qn.fail_cnt = 0
    else
        # θ = (0.8*s'*qn.B*s)/(s'*qn.B*s - ρ)
        @warn "L-BFGS update skip"
        qn.fail_cnt += 1
        flag = true
    end
    # y = θ*y + (1.0 - θ)*qn.B*s

    push!(qn.y,y)

    if flag
        pop!(qn.s)
        pop!(qn.y)
    end

    if qn.fail_cnt > max_reset_cnt
        reset_quasi_newton!(qn)
    end

    return nothing
end

function get_B(qn::LBFGS)
    if length(qn.s) > 0
        n = size(qn.B,1)
        # δ = (qn.y[end]'*qn.y[end])/(qn.s[end]'*qn.y[end])
        δ = (qn.s[end]'*qn.y[end])/(qn.s[end]'*qn.s[end])

        In = sparse(1.0*I,n,n)

        ls = length(qn.s)
        _k = min(ls,qn.k)
        S = [qn.s[ls-_k+i] for i = 1:_k]
        Y = [qn.y[ls-_k+i] for i = 1:_k]

        qn.A1 .= 0.0
        qn.A2 .= 0.0
        qn.D .= 0.0
        qn.L .= 0.0

        for i = 1:_k
            qn.A1[:,i] = δ*S[i]
            qn.A1[:,i+_k] = Y[i]

            for j = 1:_k
                qn.L[i,j] = (i > j ? S[i]'*Y[j] : 0.)
            end
            qn.D[i] = S[i]'*Y[i]
        end
        qn.A2[1:_k,1:_k] = δ*hcat(S...)'*hcat(S...)
        qn.A2[1:_k,_k .+ (1:_k)] = qn.L[1:_k,1:_k]
        qn.A2[_k .+ (1:_k),1:_k] = qn.L[1:_k,1:_k]'
        qn.A2[CartesianIndex.(_k .+ (1:_k),_k .+ (1:_k))] = -1.0*qn.D[1:_k]

        qn.B = δ*In - qn.A1[1:n,1:2*_k]*inv(qn.A2[1:2*_k,1:2*_k])*qn.A1[1:n,1:2*_k]'
    end
    return qn.B
end

function reset_quasi_newton!(qn::LBFGS)
    @warn "L-BFGS reset"
    qn.B .= sparse(1.0*I,size(qn.B))
    empty!(qn.s)
    empty!(qn.y)
    qn.fail_cnt = 0
    return nothing
end
