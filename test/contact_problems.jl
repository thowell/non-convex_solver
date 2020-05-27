function particle(;qpp=[0.,0.,1.],v0=[0.,0.,0.])
    nc = 1
    nf = 2
    nq = 3
    nu = 2
    nβ = nc*nf

    nx = nq+nu+nc+nβ+nc
    np = nq+nβ+4nc

    dt = 0.1

    M(q) = 1.0*Matrix(I,nq,nq)
    B(q) = [1. 0. 0.;0. 1. 0.]
    P(q) = [1. 0. 0.;0. 1. 0.]

    G(q) = [0; 0; 9.8]

    N(q) = [0; 0; 1]

    v1 = v0 - G(qpp)*dt
    qp = qpp + 0.5*dt*(v0 + v1)

    v2 = v1 - G(qp)*dt
    q1 = qp + 0.5*dt*(v1 + v2)

    qf = [0.; 0.; 0.]
    uf = [0.; 0.]

    W = 10.0*Matrix(I,nq,nq)
    w = -W*qf
    R = 1.0e-1*Matrix(I,nu,nu)
    r = -R*uf
    obj_c = 0.5*qf'*W*qf + 0.5*uf'*R*uf

    function unpack(x)
        q = x[1:nq]
        u = x[nq .+ (1:nu)]
        y = x[nq+nu+nc]
        β = x[nq+nu+nc .+ (1:nβ)]
        ψ = x[nq+nu+nc+nβ+nc]

        return q,u,y,β,ψ
    end

    function f_func(x)
        q,u,y,β,ψ = unpack(x)
        return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c
    end

    function c_func(x)
        q,u,y,β,ψ = unpack(x)
        [(N(q)'*q);
         (((0.5*y)^2 - β'*β));
         (M(q)*(2*qp - qpp - q)/dt - G(q)*dt + B(q)'*u + P(q)'*β + N(q)*y);
         (P(q)*(q-qp)/dt + 2.0*β*ψ);
         y*(N(q)'*q);
         ψ*((0.5*y)^2 - β'*β)]
    end

    f, ∇f!, ∇²f! = objective_functions(f_func)
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    n = nx
    m = np
    xL = zeros(nx)
    xL[1:(nq+nu)] .= -Inf
    xL[nq+nu+nc .+ (1:nβ)] .= -Inf
    xU = Inf*ones(nx)

    cI_idx = zeros(Bool,m)
    cI_idx[1:nc+nc] .= 1

    cA_idx = ones(Bool,m)
    cA_idx[1:nc+nc] .= 0

    nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)

    q0 = q1
    u0 = 1.0e-3*ones(nu)
    y0 = 1.0
    β0 = 1.0*ones(nβ)
    ψ0 = 1.0
    x0 = [q0;u0;y0;β0;ψ0]

    return x0, nlp_model
end

mutable struct Hopper{T,S} <: AbstractModel
    mb::T
    ml::T
    Jb::T
    Jl::T
    r::T
    μ::T
    g::T
    k::S
    Δt::T
end

function hopper(;T=1,r=0.7,q0=[0.,r,r,0.,0.],qf =[1.,r,r,0.,0.])
    # Dimensions
    nq = 5 # configuration dim
    nu = 2 # control dim
    nc = 1 # number of contact points
    nf = 1 # number of faces for friction cone pyramid
    nβ = nc*nf

    nx = nq+nu+nc+nβ+nc
    np = nq+nβ+4nc

    # Parameters
    g = 9.81 # gravity
    Δt = 0.1 # time step
    μ = 0.5  # coefficient of friction
    mb = 10. # body mass
    ml = 1.  # leg mass
    Jb = 2.5 # body inertia
    Jl = 0.25 # leg inertia

    # Kinematics
    p1(q) = [q[1] + q[3]*sin(q[5]), q[2] - q[3]*cos(q[5])]

    # Methods
    M(h::Hopper,q) = Diagonal([h.mb+h.ml, h.mb+h.ml, h.ml, h.Jb, h.Jl])
    ∇V(h::Hopper,q) = [0., (h.mb+h.ml)*h.g, 0., 0., 0.]

    C(h::Hopper,qk,qn) = zeros(nq)

    function ϕ(::Hopper,q)
        q[2] - q[3]*cos(q[5])
    end

    N(::Hopper,q) = ([0., 1., -cos(q[5]), 0., q[3]*sin(q[5])])'

    function P(::Hopper,q)
            ([1., 0., sin(q[5]), 0., q[3]*cos(q[5])])'
    end

    B(::Hopper,q) = [0. 0. 0. 1. -1.;
                     0. 0. 1. 0. 0.]

    model = Hopper(mb,ml,Jb,Jl,r,μ,g,p1,Δt)

    function unpack(x)
        q = x[1:nq]
        u = x[nq .+ (1:nu)]
        y = nc == 1 ? x[nq+nu+nc] : x[nq+nu .+ (1:nc)]
        β = nβ == 1 ? x[nq+nu+nc+nβ] : x[nq+nu+nc .+ (1:nβ)]
        ψ = nc == 1 ? x[nq+nu+nc+nβ+nc] : x[nq+nu+nc+nβ .+ (1:nc)]

        return q,u,y,β,ψ
    end

    W = Diagonal([1e-3,1e-3,1e-3,1e-3,1e-3])
    R = Diagonal([1.0e-1,1.0e-3])
    Wf = Diagonal(1.0*ones(nq))

    uf = zeros(nu)
    w = -W*qf
    wf = -Wf*qf
    rr = -R*uf
    obj_c = 0.5*(qf'*W*qf + uf'*R*uf)
    obj_cf = 0.5*(qf'*Wf*qf + uf'*R*uf)

    function linear_interp(x0,xf,T)
        n = length(x0)
        X = [copy(Array(x0)) for t = 1:T]

        for t = 1:T
            for i = 1:n
                X[t][i] = (xf[i]-x0[i])/(T-1)*(t-1) + x0[i]
            end
        end

        return X
    end

    Q0 = linear_interp(q0,qf,T+2)

    qpp = Q0[2]
    qp = Q0[2]

    function f_func(z)
        _sum = 0.
        for t = 1:T
            q,u,y,β,ψ = unpack(z[(t-1)*nx .+ (1:nx)])

            if t != T
                _sum += 0.5*q'*W*q + w'*q + 0.5*u'*R*u + rr'*u + obj_c
            else
                _sum += 0.5*q'*Wf*q + wf'*q + 0.5*u'*R*u + rr'*u + obj_cf
            end
        end
        return _sum
    end

    function c_func(z)
        c = zeros(eltype(z),np*T)

        for t = 1:T
            q,u,y,β,ψ = unpack(z[(t-1)*nx .+ (1:nx)])

            if t == 1
                _qpp = qpp
                _qp = qp
            elseif t == 2
                _qpp = qp
                _qp = z[(t-2)*nx .+ (1:nq)]
            else
                _qpp = z[(t-3)*nx .+ (1:nq)]
                _qp = z[(t-2)*nx .+ (1:nq)]
            end

            c[(t-1)*np .+ (1:np)] .= [(ϕ(model,q));
                                      ((model.μ*y)^2 - β'*β);
                                      (1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*∇V(model,_qp) + B(model,q)'*u +  N(model,q)'*y + P(model,q)'*β);
                                      (P(model,q)*(q-_qp)/model.Δt + 2.0*β*ψ);
                                      ϕ(model,q)*y;
                                      ((model.μ*y)^2 - β'*β)*ψ]
         end
         return c
    end

    f, ∇f!, ∇²f! = objective_functions(f_func)
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    n = T*nx
    m = T*np
    xL = zeros(T*nx)
    xU = Inf*ones(T*nx)

    for t = 1:T
        xL[(t-1)*nx .+ (1:nq+nu)] .= -Inf
        xL[(t-1)*nx + 3] = model.r/2.
        xU[(t-1)*nx + 3] = model.r

        xL[(t-1)*nx+nq+nu+nc .+ (1:nβ)] .= -Inf
    end

    cI_idx_t = zeros(Bool,np)
    cI_idx_t[1:nc+nc] .= 1
    cI_idx = zeros(Bool,m)

    for t = 1:T
        cI_idx[(t-1)*np .+ (1:np)] .= cI_idx_t
    end

    cA_idx_t = ones(Bool,np)
    cA_idx_t[1:nc+nc] .= 0
    cA_idx = ones(Bool,m)

    for t = 1:T
        cA_idx[(t-1)*np .+ (1:np)] .= cA_idx_t
    end

    nlp_model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)

    u0 = 1.0e-3*ones(nu)
    y0 = 1.0e-3
    β0 = 1.0e-3*ones(nβ)[1]
    ψ0 = 1.0e-3

    x0 = zeros(T*nx)
    for t = 1:T
        x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];u0;y0;β0;ψ0]
    end

    return x0, nlp_model
end
