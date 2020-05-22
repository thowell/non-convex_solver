# CALIPSO: a Conic primal-dual Augmented-Lagrangian Interior-Point SOlver

This is an infeasible-start, conic primal-dual augmented-Lagrangian interior-point solver for non-convex optimization problems. The solver implements key algorithms from [Ipopt](https://link.springer.com/content/pdf/10.1007/s10107-004-0559-y.pdf) to handle inequality constraints and a novel primal-dual augmented Lagrangian is employed for equality constraints. Conic constraints, including second-order and positive semidefinite cones, are coming soon.

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

Cones:
- [ ] second-order cone
- [ ] positive semidefinite cone

Augmented Lagrangian:
- [ ] penalty feedback update

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
- [X] restoration mode cleanup
  - [ ] indices/views
  - [X] single bound damping
- [ ] iterative refinement rejection
- [ ] restoration-free version
- [ ] Quasi-Newton
 - [ ] L-BFGS
 - [ ] block-wise BFGS
- [ ] Schur complement linear system solve
