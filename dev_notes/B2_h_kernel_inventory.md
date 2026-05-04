# B2 H-kernel inventory (legacy ratio paths in tulpa)

Single-file inventory of every place tulpa's source code reads ratio-specific
fields (`data.legacy.*`, `model_type`) inside the gradient pipeline.
Confirmed by reading each file in full (not grep snippets).

| File | Function / fragment | Families covered | Notes |
|------|---------------------|-------------------|-------|
| `src/hmc_gradient_analytical_lik_scalar.h` (872 LOC) | obs-loop body inside `compute_gradient_analytical` | All 7 (BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA, NEGBIN_GAMMA, GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL) plus ZI/OI variants | Reaches: `data.legacy.y_num`, `y_denom`, `y_num_cont`, `y_denom_cont`, `X_num_flat`, `X_denom_flat`, `model_type`. ZI/OI variants are out of B2 scope. |
| `src/hmc_gradient_analytical_lik_vec.h` (190 LOC) | vectorised obs-loop fast path | Same 7 families | Activated when `used_vectorized = true`; uses the same legacy fields. |
| `src/hmc_gradient_analytical_priors_basic.h` (213 LOC) | beta + RE + phi prior gradients | All families (priors are family-agnostic) | Beta prior uses `data.legacy.p_num/p_denom`; phi prior uses `legacy.has_phi_*`. |
| `src/hmc_gradient_analytical_priors_misc.h` (~100 LOC) | spatial/temporal/SVC prior gradients | All families | Reads `data.legacy.model_type` only inside ZI gating branches. |
| `src/hmc_gradient_analytical_post.h` (147 LOC) | post-loop assembly + spatial prior + log-post | All families | Reads `legacy.model_type` to decide whether to share residuals across processes. |
| `src/hmc_gradient_helpers_impl.h` (1500+ LOC) | shared helpers (`beta_gradient_prior`, `phi_gradient_prior`, `compute_obs_residuals`, `scatter_beta_gradients`, `accumulate_phi_likelihood_grad`, ...) | All 7 (incl. ZI) | Heavy ratio-specific surface. The `compute_obs_residuals` switch alone duplicates per-family math from lik_scalar.h. |
| `src/hmc_gradient_vectorized_*.h` (~70K LOC) | composite + vectorised gradient | All families incl. RE-slopes | `phase3_loop` has per-family residual code that mirrors lik_scalar.h. |
| `src/hmc_gradient_dispatch.h` | `resolve_gradient_fn` | All families | Already routes spec-path (`n_processes>0` + `gradient_fn != nullptr`) to `spec.gradient_fn` regardless of mode (B2 hooks into THIS dispatch slot — no edit needed). |

## Per-family analytical gradient (recorded for B2 port)

All formulas verified against `tulpa/src/hmc_gradient_analytical_lik_scalar.h`
and the math headers already in tulpaRatio's `lik_*.cpp` (B1b).

| Family | resid (eta-space) | extra-grad (d/dphi or d/dlog_sigma) | Process count |
|--------|-------------------|--------------------------------------|---------------|
| binomial | `y - n*p`, `p = sigmoid(eta)` | n/a | 1 |
| poisson_gamma | num: `y_n - mu_n`; denom: `shape*(1 - y_d/mu_d)` (skip y_d ≤ 0) | `d/dshape = log(rate) + 1 + log(y_d) − ψ(shape) − rate*y_d/shape` | 2 |
| negbin_gamma | num: `y_n − mu_n*(y_n+phi_n)/(mu_n+phi_n)`; denom: gamma resid | num: NegBin phi-grad; denom: gamma shape-grad | 2 |
| negbin_negbin | both: NegBin scalar resid | both: NegBin phi-grads | 2 |
| gamma_gamma | both: `phi*(y/mu − 1)` | both: gamma shape-grads | 2 |
| lognormal | `(log y − mu) / sigma^2` (eta IS mu_log) | `d/dlog_sigma = z^2 − 1`, `z = (log y − mu)/sigma` | 2 |
| beta_binomial | `dLL/dp * p*(1-p)` with `dLL/dp = phi*(ψ(y+α)−ψ(n−y+β)−ψ(α)+ψ(β))` | beta-binomial digamma sum | 1 |

## Important reality check

The B2 spec path sees only `n_processes > 0` data. None of:
- ZI / OI variants
- Spatial (ICAR / BYM2 / GP / HSGP)
- Temporal (AR1 / GP / multi-scale)
- RE (single / crossed / slopes / correlated)
- SVC / TVC / latent / spatiotemporal

are reachable in B2. Those are gated by `specs_eligible` in
`tulpaRatio/R/backend_hmc.R` and stay on the legacy backend until later
phases (B1c / B1d). The tulpaRatio H-kernel asserts via `Rcpp::stop` on
any of those flags so a mistake gets caught immediately, not silently.

## Deletion impact (NOT executed in B2)

Tulpa's `cpp_hmc_fit` C++ entry point still routes `n_processes == 0`
(legacy ratio mode) through `compute_gradient_analytical`, which would
reach the (now-empty) per-family branches if the kernel were deleted.
That path is exercised by tulpa-side tests:

* `tulpa/tests/testthat/test-hmc-modeldata-builders.R`: 2 tests at
  `model_type_str = "binomial"` with `gradient_mode_str = "auto"` —
  the dispatcher picks H since `can_use_analytical_gradient(...)` returns
  true for these inputs.
* `tulpa/tests/testthat/test-spatial-car-proper.R`: gradient check at
  `model_type_str = "binomial"` with `gradient_mode_str = "H"`.

Per the user's plan, these tests must NOT regress after deletion. Until
those entry points are themselves removed (Phase D) or rewired to AD
fallback, deleting the legacy ratio H-kernel breaks them. Deletion is
therefore deferred — see report.
