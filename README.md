# Interior-point solver

This is an infeasible-start, interior-point solver for non-convex optimization problems. The solver re-implements key algorithms from [Ipopt](https://link.springer.com/content/pdf/10.1007/s10107-004-0559-y.pdf).

Problems of the following form,
```
minimize        f(x)

subject to      xL <= x <= xU
                c(x) = 0
```

can be solved.

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
  - modified: \delta_c = \mu
- [ ] acceleration heuristics (sec. 3.2)
  - [X] Case 1
  - [ ] Case 2: watchdog
- [X] feasibility restoration phase
- [X] KKT error reduction
- [X] relaxed bounds (eq. 35)
  -[ ] implemented for restoration mode
- [X] primal initialization (sec. 3.6)
- [X] equality constraint multiplier initialization (eq. 36)
- [X] single bounds damping
- [X] automatic scaling of problem statement
- [X] small search directions
- [X] iterative refinement on unreduced system
  - [ ] tolerance
- [X] round-off error acceptance criteria relaxation
- [X] MA57
  - [ ] default pivot tolerance

TODO
-[ ] restoration mode cleanup
  -[ ] indices/views
  -[ ] single bound damping
-[ ] iterative refinement rejection
-[ ] watchdog
-[ ] restoration-free version
