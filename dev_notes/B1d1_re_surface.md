# B1d-1 RE surface inventory

## Legacy RE feature surface

| Feature                                | Legacy (HMC `cpp_hmc_fit`) | B1c spec path | B1d-1 scope    |
|----------------------------------------|----------------------------|---------------|----------------|
| Single-term, intercept-only            | yes                        | yes           | yes (already)  |
| Single-term, intercept + uncorr slopes | yes                        | rejected      | **yes (new)**  |
| Single-term, intercept + corr slopes   | yes (LKJ)                  | rejected      | no (B1d-2+)    |
| Multi-term (crossed grouping factors)  | yes                        | rejected      | no (B1d-2)     |
| Nested grouping factors                | flattened to crossed       | rejected      | no (B1d-2)     |

## Where the legacy plumbing lives

- **Formula parsing**: `R/formula.R` resolves `(1 | g)` and `(1 + x | g)` /
  `(1 + x || g)` into per-term lists `{group_var, group_idx, n_groups,
  has_intercept, slope_vars, correlated}`.
- **`extract_re_for_hmc()` (`R/backend_hmc.R:1183`)** rolls those terms into:
  `re_terms[t] = {n_groups, n_coefs, n_sigma, n_chol, ..., re_offset, ...}`,
  `total_re_params`, `total_sigma_params`, `total_chol_params`,
  `has_slopes`, `has_correlated_slopes`, plus a per-term `slope_matrix`
  (built in `extract_hmc_data()` at `R/backend_hmc.R:1107`).
- **q_init layout (legacy)** (`R/backend_hmc.R:2286`): for slopes
  `[ ... | sigmas (total_sigma_params) | re (total_re_params)
   | chol (total_chol_params, only if correlated) | ... ]`.
  Matches `hmc_sampler.cpp::compute_param_layout` exactly.

## Engine-side RE math

`tulpa/src/tulpa_priors_re.h::compute_re_prior<T>` is the single source of
truth. It supports all three regimes with the same template:

  - Non-centered uncorrelated: `z ~ N(0, I)`, `re = sigma * z`,
    `log p(z) = -0.5 sum z^2`, `log p(log_sigma) = log_HC(0, sigma_re_scale)`.
  - Non-centered correlated: tanh-Cholesky of the correlation matrix +
    LKJ(2) prior + Cholesky -> correlation Jacobian; `re = diag(sigma) * L * z`.
  - Centered: `re ~ N(0, sigma^2)`, `log p(re) = -0.5 (re/sigma)^2 - log sigma`.

Default for both legacy and spec is non-centered (`re_parameterization = 1`).

`tulpa/src/log_post_generic_impl.h::generic_re_effect<T>` adds `Z * u` to
`eta` per observation. Walks `re_group_multi_flat` (1-based,
`obs * n_terms + term`) and `re_slope_matrices[t][i*n_slopes + s]` for
slope coefficients. Falls back to legacy `re_group[i]` for the
`n_terms == 0` intercept-only path.

`add_to_shared_processes(eta, data.sharing.re, ...)` decides which
processes get the random effect added; `data.sharing.re` defaults to
`true` for every process (`SharingSpec::init`).

## What the spec path needs to set on `ModelData` for B1d-1

For the **single-term, uncorrelated, intercept + slopes** case:

- `data.n_re_terms        = 1`
- `data.n_re_groups       = n_groups` (legacy single-term mirror)
- `data.re_group          = ...`      (1-based, length N — legacy mirror)
- `data.re_group_multi_flat = ...`    (length N, same content as re_group)
- `data.re_n_groups_multi = {n_groups}`
- `data.re_offsets        = {0}`
- `data.has_re_slopes     = true`     (only when n_coefs > 1)
- `data.has_re_correlated_slopes = false`   (B1d-1 gates correlated out)
- `data.re_n_coefs        = {n_coefs}`           (intercept + slope_vars)
- `data.re_n_slopes       = {n_coefs - 1}`
- `data.re_correlated     = {false}`
- `data.re_n_chol         = {0}`
- `data.re_slope_matrices[0] = flat N x n_slopes row-major`
- `data.total_re_groups   = n_groups`
- `data.total_re_params   = n_groups * n_coefs`
- `data.total_sigma_params = n_coefs`
- `data.total_chol_params = 0`
- `data.re_parameterization = 1`     (NC; matches legacy default)
- `data.sigma_re_scale    = sigma_prior_scale`

The bridge does NOT need to call any layout helper — `tulpa::compute_layout`
reads these fields and builds the right slice. tulpaRatio's
`hmc_sampler::compute_param_layout` is the same algorithm, so legacy and
spec layouts agree slot-by-slot.

## Out of scope for B1d-1 (kept on legacy-only path)

- Multi-term RE (`n_terms > 1`): bridge keeps the existing rejection.
- Correlated slopes (`re_correlated[t] = true`): R-side `re_supported`
  predicate keeps these on legacy.
- Centered parameterization (`re_parameterization = 0`): not exercised
  by tulpaRatio's R-side defaults. Falls through if a user explicitly
  sets `re_param = "centered"`; the engine's centered branch is the same
  math, but parity tests stay on the default NC path.

## In-repo bug review

`R/backend_hmc.R:354-361` and the legacy q_init builder both treat
`n_re_terms == 1` as the "single term" path, which the bridge currently
maps to `data.n_re_terms = 0` (lines 432 of `tulpa_bridge.cpp`) so the
engine routes through the legacy `re_group` branch. That's deliberate
and matches how tulpa's `compute_param_layout` selects the
intercept-only single-term slot. No bug. For the slopes case we **must**
set `data.n_re_terms = 1` so the slopes branch fires.

No other RE-related issues surfaced.
