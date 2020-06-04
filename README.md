## Non-convex solver

This is an infeasible-start, primal-dual augmented-Lagrangian interior-point solver for non-convex optimization problems. The solver implements key algorithms from [Ipopt](https://link.springer.com/content/pdf/10.1007/s10107-004-0559-y.pdf) to handle inequality constraints and a novel primal-dual augmented Lagrangian is employed for equality constraints. Conic constraints, including second-order and positive semidefinite cones, are coming soon.

Problems of the following form,
```
minimize        f(x)
   x
subject to      cI(x) >= 0
                c(x)   = 0
                cA(x)  = 0
                xL <= x <= xU
```

can be solved.

## Features
Augmented Lagrangian:
- [ ] penalty feedback update

Cones:
- [ ] second-order cone
- [ ] positive semidefinite cone

The following algorithms from Ipopt are implemented:
- [X] barrier update (eq. 7)
- [X] fraction to boundary (eqs. 8, 15)
- [X] \alpha, \alpha_z update (eq. 14)
- [X] z reset (eq. 16)
- [X] smaller, symmetric linear solve (eq. 13)
- [X] line-search filter (Algorithm A)
- [X] filter update (eq. 22)
- [X] \alpha min (eq. 23)
- [X] second-order corrections
- [X] inertia correction
- [X] acceleration heuristics (sec. 3.2)
  - [X] Case 1:
  - [X] Case 2: watchdog
- [X] feasibility restoration phase
- [X] KKT error reduction
- [X] relaxed bounds (eq. 35)
  -[X] implemented for restoration mode
- [X] primal initialization (sec. 3.6)
- [X] equality constraint multiplier initialization (eq. 36)
- [X] single bounds damping
  -[X] implemented for restoration mode
- [X] automatic scaling of problem statement
- [X] small search directions
- [X] iterative refinement on fullspace system
  - [ ] tolerance
- [X] round-off error acceptance criteria relaxation
- [X] MA57
  - [ ] default pivot tolerance

TODO:
- [ ] replace MA57 solver
- [X] restoration mode cleanup
  - [X] indices/views
  - [X] single bound damping
- [ ] iterative refinement rejection
- [ ] restoration-free version
- [ ] Quasi-Newton
 - [ ] BFGS
 - [ ] L-BFGS
 - [ ] block-wise BFGS
- [ ] Schur-complement linear-system solve

## Install
-Git clone this repository
-Install [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)

## Example
Wachter problem
```
minimize        x1
x1,x2,x3
subject to      x1^2 - x2 - 1.0 = 0
                x1   - x3 - 0.5 = 0
                x2, x3 >= 0
```

```julia
# problem dimensions
n = 3 # decision variables
m = 2 # constraints

# initial guess
x0 = [-2.0;3.0;1.0]

# bounds
xL = -Inf*ones(n) # lower bounds
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n) # upper bounds

# objective
f_func(x) = x[1]

# constraints
c_func(x) = [x[1]^2 - x[2] - 1.0;
           x[1] - x[3] - 0.5]

# model
model = Model(n,m,xL,xU,f_func,c_func)

# options
opts = Options{Float64}()

# solver
s = NonConvexSolver(x0,model,opts=opts)

# solve
solve!(s)

# solution
x = get_solution(s) # x* = [1, 0, 0.5]
```
