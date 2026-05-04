# B2 port design

## Files

- `src/lik_specs/lik_grad_h_kernel.h` — declarations of the 7 per-family
  `void grad_h_<family>(params, data, layout, grad, log_post_out)` entry
  points. Matches `tulpa::FullGradFn` signature exactly.
- `src/lik_grad_h_kernel.cpp` — single-file implementation. Contains:
  * The math header (closed-form per-obs residuals + extra-param priors).
  * `assert_no_latent_structure(layout)`: bails on any spatial/temporal/
    RE/ZI/OI/SVC/TVC/ST/latent flag — keeps the kernel honest about its
    B2 scope.
  * `apply_beta_prior(...)`: shared Gaussian beta-prior contribution.
  * `compute_eta_matrix(...)`: matvec per process, plus optional offset.
  * `template <class Fn> double run_obs_loop(...)`: shared scaffold that
    asks each family's functor for `resid[k]` and `extra[]`, scatters
    `X_k^T r` into beta gradients, and accumulates extra-param gradients.
  * 7 per-family functor structs (`BinomialFn`, `PoissonGammaFn`, ...)
    with `operator()` returning `log L_i` and writing the per-obs
    derivatives.
  * 7 public entry points that wire the right functor + extra-parameter
    prior gradients (Gamma on log_phi / log_shape, Half-Cauchy on
    log_sigma) and the fused log-posterior.

## Wiring

Each `build_<family>_spec` (in `lik_<family>.cpp`) sets
`spec.gradient_fn = &grad_h_<family>` after the `ll_double` /
`ll_arena` / `ll_fwd` registration. Tulpa's gradient dispatcher
(`hmc_gradient_dispatch.h`) prefers `spec.gradient_fn` over any AD path
when the user requests anything other than NUMERICAL — so the spec path
at `gradient_mode = "H"` (and `"auto"`) calls our kernel.

## Bridge changes

`tulpa_bridge.cpp` now also writes `data.phi_prior_shape`,
`data.phi_prior_rate`, and `data.sigma_re_scale` from the bridge args.
The AD path was reading these from a per-family `static cfg`; the
H-kernel reads them from `ModelData` so the entire prior config sits in
one well-known place.

## No copy-paste rule

The 7 family functors share:
* `apply_beta_prior` — Gaussian on betas (single source of truth).
* `compute_eta_matrix` — eta = X @ beta + offset (single source of truth).
* `run_obs_loop<PerObsFn>` — X^T r scatter + extra accumulation (single
  source of truth, instantiated per family by the compiler).
* Helper math (Gamma / Half-Cauchy log-prior + grad) — single source of
  truth.

The only per-family code is the 6-30 line `operator()` of each functor:
the per-obs `resid[k]` and `extra[e]` formulas. No scaffolding,
allocation, or accumulation logic is copy-pasted across families.
