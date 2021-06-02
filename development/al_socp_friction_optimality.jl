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

P = [1. 0. 0.; 0. 1. 0.]

v = [1.;5.;0.]
n = 1.0

function f(x)
    v'*P'*x
end

function c(x)
	[x;n] - vcat(Πsoc(x,n)...)
end

"Augmented Lagrangian"
L(x,λ,ρ) = f(x) + λ'*c(x) + 0.5*ρ*c(x)'*c(x)

function solve(x)
	x = copy(x)

	λ = zeros(3)
	ρ = 1.0

	k = 1
	while k < 10
		i = 1
		while i < 25

			res(z) = ForwardDiff.gradient(f,z) + ForwardDiff.jacobian(c,z)'*(λ + ρ*c(z)) #ForwardDiff.gradient(L,z)
			_res = res(x)

			norm(_res) < 1.0e-5 && break

			∇res = ForwardDiff.jacobian(res,x) # ForwardDiff.hessian(L,x)

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

		norm(c(x)) < 1.0e-6 && break

		λ += ρ*[x;n]
		λ = vcat(Πsoc(λ[1:2],λ[3])...)

		ρ *= 10.0

		k += 1
	end
	k == 10 && (@warn "solve failed")
	return x, λ, ρ
end

x0 = zeros(2)
x_sol, λ_sol, ρ_sol = solve(x0)
@show x_sol

x_sol
norm(c(x_sol))

norm(x_sol) - n

x_sol
