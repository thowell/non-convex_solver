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

nx = nβ
np = nβ+nc

dt = 0.1

M(q) = Diagonal(1.0*ones(nq))
B(q) = [1. 0. 0.;0. 1. 0.]
P(q) = [1. 0. 0.;0. 1. 0.]

G(q) = [0; 0; 9.8]

N(q) = [0; 0; 1]

qpp = [0.,0.,0.1]
v0 = [1.0,0.0, 0.]
v1 = v0 - G(qpp)*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G(qp)*dt
q1 = qp + 0.5*dt*(v1 + v2)

qf = [0.; 0.; 0.]
uf = [0.; 0.]

W = Diagonal(1.0*ones(nq))
w = -W*qf
R = Diagonal(1.0e-1*ones(nu))
r = -R*uf
obj_c = 0.5*qf'*W*qf + 0.5*uf'*R*uf

y = 1.0
function unpack(x)
    β = x[1:nβ]

    return β
end

function f(x)
    β = unpack(x)
	return v0'*P(qpp)'*β/dt
end

function cone_dif(z)
	v = z[1:nβ]
	s = z[nβ+1]
	[v;s] - vcat(Πsoc(v,s)...)
end

function cone_dif_alt(z)
	v = z[1:nβ]
	s = z[nβ+1]
	[v;s] #- vcat(Πsoc(v,s)...)
end

function c(x,λ_i,ρ_i)
    β = unpack(x)
	cone_dif([β;y])
end

function c_hist(x,λ_i,ρ_i)
    β = unpack(x)

	[β;y]
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
			println("res: $(norm(∇L))")
			norm(∇L) < 1.0e-5 && break
			∇²L = ForwardDiff.hessian(_L,x)
			Δx = -(∇²L + 1.0e-5*I)\∇L

			println("cond: $(cond(∇²L))")
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
		# norm(c(x,λ_i,ρ_i)) < 1.0e-3 && break



		λ += ρ*c_hist(x,λ_i,ρ_i)
		λ[(1:nβ+1)] = vcat(Πsoc(λ[(1:nβ)],λ[nβ+1])...)

		ρ *= 10.0

		# λ_i += ρ_i*[β;y]
		# λ_i = vcat(Πsoc(λ_i[1:end-1],λ_i[end])...)
		#
		# ρ_i *= 10.0
		#
		# λ_i = λ[3nc+nq .+ (1:nβ+1)]
		# ρ_i = ρ

		k += 1

		println("outer loop: $ρ")

		(norm(c(x,λ_i,ρ_i)) < 1.0e-5 && k > 3) && break

	end
	k == 10 && (@warn "solve failed")
	return x, λ, ρ, λ_i, ρ_i
end

x0 = 1.0*rand(nβ)
x_sol, λ_sol, ρ_sol, λ_i_sol, ρ_i_sol = solve(copy(x0))
@show x_sol[1:nβ]
norm(c(x_sol,λ_i_sol,ρ_i_sol))
β = unpack(x_sol)
norm(β) - y
y
λ_i_sol
λ_sol
ρ_sol
ρ_i_sol
(q1-qp)'*P(q)'
