abstract type QuasiNewton end

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

function update_quasi_newton!(qn::BFGS,x; x_update=false, ∇L_update=false, max_reset_cnt=100)

    x_update && push!(qn.s,x - qn.x_prev)

    if ∇L_update && length(qn.s) > 0
        y = qn.∇L - qn.∇L_prev

        # damped
        s = qn.s[end]
        ρ = s'*y

        if ρ >= 0.2*s'*qn.B*s
            θ = 1.0
            qn.fail_cnt = 0
        else
            θ = (0.8*s'*qn.B*s)/(s'*qn.B*s - ρ)
            qn.fail_cnt += 1
        end
        y = θ*y + (1.0 - θ)*qn.B*s

        if x_update || isempty(qn.y)
            push!(qn.y,y)
        else
            qn.y[end] = y
        end
    end

    if qn.fail_cnt >= max_reset_cnt
        reset_quasi_newton!(qn)
    else
        qn.x_prev .= x
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

function reset_quasi_newton!(qn::BFGS)
    @warn "quasi newton reset"
    qn.B .= sparse(1.0*I,size(qn.B))
    empty!(qn.s)
    empty!(qn.y)
    qn.fail_cnt = 0
    return nothing
end

# L-BFGS
#
# n = 100
# m = 2
# δ = 1.0
# In = sparse(1.0*I,n,n)
#
# S = [i < 3 ? rand(n) : zeros(n) for i = 1:m]
# Y = [i < 3 ? rand(n) : zeros(n) for i = 1:m]
#
# hcat(S...)
# tmp1 = zeros(n,2m)
# tmp2 = zeros(2m,2m)
# L = spzeros(m,m)
# D = zeros(m)
#
# for i = 1:m
#     tmp1[:,i] = δ*S[i]
#     tmp1[:,i+m] = Y[i]
#
#     for j = 1:m
#         L[i,j] = (i > j ? S[i]'*Y[j] : 0.)
#     end
#     D[i] = S[i]'*Y[i]
# end
#
# D
#
# tmp2[1:m,1:m] = δ*hcat(S...)'*hcat(S...)
# tmp2[1:m,m .+ (1:m)] = L
# tmp2[m .+ (1:m),1:m] = L'
# tmp2[CartesianIndex.(m .+ (1:m),m .+ (1:m))] = -1.0*D
#
# Bk = δ*In - tmp1*(tmp2\tmp1')
#
# rank(Bk)

#
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

qn = LBFGS(n=100,m=10,k=5)

function update_quasi_newton!(qn::LBFGS,x; x_update=false, ∇L_update=false, max_reset_cnt=2)

    x_update && push!(qn.s,x - qn.x_prev)

    if ∇L_update && length(qn.s) > 0
        y = qn.∇L - qn.∇L_prev

        # damped
        s = qn.s[end]
        ρ = s'*y

        if ρ >= 0.2*s'*qn.B*s
            θ = 1.0
            qn.fail_cnt = 0
        else
            θ = (0.8*s'*qn.B*s)/(s'*qn.B*s - ρ)
            qn.fail_cnt += 1
        end
        y = θ*y + (1.0 - θ)*qn.B*s

        if x_update || isempty(qn.y)
            push!(qn.y,y)
        else
            qn.y[end] = y
        end
    end

    if qn.fail_cnt >= max_reset_cnt
        reset_quasi_newton!(qn)
    else
        qn.x_prev .= x
    end

    return nothing
end

function get_B(qn::LBFGS)
    if length(qn.s) > 0
        shift = 0
        if qn.s[end]'*qn.y[end] <= 0.
            qn.fail_cnt += 1
            shift = 1
        end
        ls = length(qn.s)-shift
        ly = length(qn.y)-shift

        (ls < 1 || ly < 1) && return qn.B

        qn.fail_cnt = 0
        n = size(qn.B,1)
        δ = (qn.y[end-shift]'*qn.y[end-shift])/(qn.s[end-shift]'*qn.y[end-shift])

        In = sparse(1.0*I,n,n)

        _k = min(ls,qn.k)
        S = [qn.s[ls-_k+i] for i = 1:_k]
        Y = [qn.y[ly-_k+i] for i = 1:_k]

        # tmp1 = zeros(n,2*_k)
        # tmp2 = zeros(2*_k,2*_k)
        # L = spzeros(_k,_k)
        # D = zeros(_k)

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

        try
            qn.B = δ*In - qn.A1[1:n,1:2*_k]*(qn.A2[1:2*_k,1:2*_k]\qn.A1[1:n,1:2*_k]')
        catch
            qn.B = δ*In
        end

    end
    return qn.B
end

function reset_quasi_newton!(qn::LBFGS)
    @warn "L-BFGS reset"
    # qn.B .= sparse(1.0*I,size(qn.B))
    empty!(qn.s)
    empty!(qn.y)
    qn.fail_cnt = 0
    return nothing
end
