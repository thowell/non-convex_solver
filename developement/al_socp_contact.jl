using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"Positive-orthant projection"
function Πp(x)
	max.(x,0.0)
end

"Second-order cone projection"
function Πsoc(v,s)
	if norm(v) <= -s
		# @warn "below cone"
		return zero(v), 0.0
	elseif norm(v) <= s
		# @warn "in cone"
		return v, s
	elseif norm(v) > abs(s)
		# @warn "outside cone"
		a = 0.5*(1.0 + s/norm(v))
		return a*v, a*norm(v)
	else
		@warn "soc projection error"
		return zero(v), 0.0
	end
end


"Particle drop"
nc = 1
nf = 2
nq = 3
nu = 2
nβ = nc*nf

nx = nq+nu+nc+nβ
np = 8

dt = 0.1

M(q) = Diagonal(1.0*ones(nq))
B(q) = [1. 0. 0.;0. 1. 0.]
P(q) = [1. 0. 0.;0. 1. 0.]

G(q) = [0; 0; 9.8]

N(q) = [0; 0; 1]

qpp = [0.,0.,10.]
v0 = [10.,-7.0, 0.]
v1 = v0 - G(qpp)*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G(qp)*dt
q1 = qp + 0.5*dt*(v1 + v2)

qf = [0.; 0.; 0.]
uf = [0.; 0.]

W = Diagonal(10.0*ones(nq))
w = -W*qf
R = Diagonal(1.0e-1*ones(nu))
r = -R*uf
obj_c = 0.5*qf'*W*qf + 0.5*uf'*R*uf

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = x[nq+nu+nc]
    β = x[nq+nu+nc .+ (1:nβ)]

    return q,u,y,β
end

function f(x)
    q,u,y,β = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c
end

function cone_dif(z)
	v = z[1:nβ]
	s = z[nβ+1]
	[v;s] - vcat(Πsoc(v,s)...)
end

function c(x,λ_i,ρ_i)
    q,u,y,β = unpack(x)
    [q[3]-Πp(q[3]);
     y - Πp(y);
	 y*q[3];
	 P(q)*(q-qp)/dt + ForwardDiff.jacobian(cone_dif,[β;y])[:,1:nβ]'*(λ_i + ρ_i*cone_dif([β;y]));
     (M(q)*(2*qp - qpp - q)/dt - G(q)*dt + B(q)'*u + P(q)'*β + N(q)*y);
    ]
end

function c_hist(x,λ_i,ρ_i)
    q,u,y,β = unpack(x)
    [q[3];
     y;
	 y*q[3];
	 P(q)*(q-qp)/dt + ForwardDiff.jacobian(cone_dif,[β;y])[:,1:nβ]'*(λ_i + ρ_i*[β;y]);
     (M(q)*(2*qp - qpp - q)/dt - G(q)*dt + B(q)'*u + P(q)'*β + N(q)*y);
    ]
end

"Augmented Lagrangian"
L(x,λ,ρ,λ_i,ρ_i) = f(x) + λ'*c(x,λ_i,ρ_i) + 0.5*ρ*c(x,λ_i,ρ_i)'*c(x,λ_i,ρ_i)

function solve(x)
	x = copy(x)

	λ = zeros(np)
	ρ = 1.0

	λ_i = zeros(nβ+1)
	ρ_i = 1.0

	k = 1
	while k < 10
		i = 1
		while i < 25
			_L(z) = L(z,λ,ρ,λ_i,ρ_i)
			f = _L(x)
			∇L = ForwardDiff.gradient(_L,x)
			norm(∇L) < 1.0e-5 && break
			∇²L = ForwardDiff.hessian(_L,x)
			Δx = -(∇²L + 1.0e-5*I)\∇L

			α = 1.0

			j = 1
			while j < 25
				if (_L(x + α*Δx) <= f + 1.0e-4*α*Δx'*∇L) && (-Δx'*ForwardDiff.gradient(_L,x+α*Δx) <= -0.9*Δx'*∇L)
					break
				else
					α *= 0.5
					j += 1
				end
			end
			j == 10 && @warn "line search failed"

			x .+= α*Δx
			i += 1
		end
		# return x,λ,ρ
		norm(c(x,λ_i,ρ_i)) < 1.0e-3 && break

		# λ += λ + ρ*c_hist(x,λ_i,ρ_i)
		# λ[1:2] = Πp(λ[1:2])
		q,u,y,β = unpack(x)
		tmp = λ_i + ρ_i*[β;y]
		λ_i += vcat(Πsoc(tmp[1:end-1],tmp[end])...)
		# ρ *= 2.0
		ρ_i *= 10.0

		k += 1
	end
	q,u,y,β = unpack(x)
	println("cone_dif: $(cone_dif([β;y]))")
	return x, λ, ρ
end

x0 = randn(nx)
x_sol, λ_sol, ρ_sol = solve(x0)
@show x_sol


q,u,y,β = unpack(x_sol)
println("cone_dif: $(cone_dif([β;y]))")
