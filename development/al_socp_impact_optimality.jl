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

nx = nq+nu+nc
np = 3nc + nq

dt = 0.1

M(q) = Diagonal(1.0*ones(nq))
B(q) = [1. 0.; 0. 1.; 0. 0.]
P(q) = [1. 0. 0.; 0. 1. 0.]

G(q) = [0; 0; 9.8]

ϕ(q) = q[3]
N(q) = [0; 0; 1]

qpp = [0.,0.,0.125]
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

    return q,u,y
end

function f(x)
    q,u,y = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c
end

function ci(x)
    q,u,y = unpack(x)
    [
	q[3];
	y
    ]
end

function c̄i(x)
	ci(x) - Πp(ci(x))
end

function ce(x)
    q,u,y = unpack(x)
    [
	M(q)*(2*qp - qpp - q)/dt - G(q)*dt + B(q)*u + N(q)*y;
	q[3]*y
    ]
end

function c(x)
	[c̄i(x);ce(x)]
end

"Augmented Lagrangian"
L(x,λ,ρ) = f(x) + λ'*c(x) + 0.5*ρ*c(x)'*c(x)

function solve(x)
	x = copy(x)

	λi = zeros(2)
	λe = zeros(4)
	ρ = 1.0

	k = 1
	while k < 10
		i = 1
		while i < 25
			L(z) = f(z) + [λi;λe]'*c(z) + 0.5*ρ*c(z)'*c(z)
			_L = L(x)

			res(z) = ForwardDiff.gradient(f,z) + ForwardDiff.jacobian(c,z)'*([λi;λe] + ρ*c(z))#ForwardDiff.gradient(L,z)
			_res = res(x)

			norm(_res) < 1.0e-5 && break

			∇res = ForwardDiff.jacobian(res,x)#ForwardDiff.hessian(L,x)

			Δx = -(∇res + 1.0e-5*I)\_res

			α = 1.0
			j = 1
			while j < 25
				if norm(res(x + α*Δx),1)/length(x) <= (1.0-α*0.1)*norm(_res,1)/length(x)
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
		# return x,[λi;λe],ρ
		norm(c(x)) < 1.0e-6 && break

		λi += ρ*ci(x)
		λi = Πp(λi)

		λe += ρ*ce(x)

		ρ *= 10.0

		k += 1
	end
	k == 10 && (@warn "solve failed")
	return x, [λi;λe], ρ
end

x0 = [q1;zeros(nu);0.0]
x_sol, λ_sol, ρ_sol = solve(x0)
@show x_sol

x_sol[1:nq]
norm(c(x_sol))
