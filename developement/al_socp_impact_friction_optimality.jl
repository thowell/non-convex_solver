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
nb = nc*nf

nx = nq+nu+nc
np = 3nc + nq

dt = 0.1

M(q) = Diagonal(1.0*ones(nq))
B(q) = [1. 0.; 0. 1.; 0. 0.]
P(q) = [1. 0. 0.; 0. 1. 0.]

G(q) = [0; 0; 9.8]

ϕ(q) = q[3]
N(q) = [0; 0; 1]

qpp = [0.,0.,0.1]
v0 = [10.0,-20.0, 0.]
v1 = v0 - G(qpp)*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G(qp)*dt
q1 = qp + 0.5*dt*(v1 + v2)

qf = [0.; 0.; 0.]
uf = [0.; 0.]

W = Diagonal(10.0*ones(nq))
w = -W*qf
R = Diagonal(10000.0*ones(nu))
r = -R*uf
obj_c = 0.5*qf'*W*qf + 0.5*uf'*R*uf

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = x[nq+nu+nc]
	b = x[nq+nu+nc .+ (1:nb)]

    return q,u,y,b
end

function f_impact(x)
    q,u,y,b = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c
end

function f_friction(x)
    q,u,y,b = unpack(x)
    return (q-qp)'P(q)'*b/dt
end

function ci(x)
    q,u,y,b = unpack(x)
    [
	q[3];
	y
    ]
end

function c̄i(x)
	ci(x) - Πp(ci(x))
end

function ce(x)
    q,u,y,b = unpack(x)
    [
	M(q)*(2*qp - qpp - q)/dt - G(qp)*dt + B(q)*u + N(q)*y + P(q)'*b;
	q[3]*y
    ]
end

function c_impact(x)
	[c̄i(x);ce(x)]
end

function c_friction(x)
	q,u,y,b = unpack(x)
	[b;y] - vcat(Πsoc(b,y)...)
end

function c_friction_alt(x)
	q,u,y,b = unpack(x)
	[b;y] - vcat(Πsoc(b,y)...)
end

function solve(x)
	x = copy(x)

	λi_impact = zeros(2nc)
	λe_impact = zeros(nc+nq)
	λ_friction = zeros(nb+nc)

	ρ = 1.0

	k = 1
	while k < 10
		i = 1
		while i < 25


			res(z) = [ForwardDiff.gradient(f_impact,z)[1:(nq+nu+nc)] + ForwardDiff.jacobian(c_impact,z)[:,1:(nq+nu+nc)]'*([λi_impact;λe_impact] + ρ*c_impact(z));
					  ForwardDiff.gradient(f_friction,z)[nq+nu+nc .+ (1:nb)] + ForwardDiff.jacobian(c_friction,z)[:,nq+nu+nc .+ (1:nb)]'*(λ_friction + ρ*c_friction(z))]
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
		norm([c_impact(x);c_friction(x)]) < 1.0e-6 && break

		q,u,y,b = unpack(x)

		λi_impact += ρ*ci(x)
		λi_impact = Πp(λi_impact)

		λe_impact += ρ*ce(x)

		λ_friction += ρ*[b;y]
		λ_friction = vcat(Πsoc(λ_friction[1:nb],λ_friction[nb+1])...)

		ρ *= 10.0

		k += 1
	end
	k == 10 && (@warn "solve failed")
	return x, [λi_impact;λe_impact;λ_friction], ρ
end

x0 = [q1;0.0*rand(nu);0.0;0.0*rand(nb)]
x_sol, λ_sol, ρ_sol = solve(x0)
@show x_sol

norm([c_impact(x_sol);c_friction(x_sol)])

q_sol,u_sol,y_sol,b_sol = unpack(x_sol)
norm(b_sol) - y_sol
@show b_sol
@show y_sol


u_sol
