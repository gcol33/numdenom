# tulpaRatio (development)

## 1.4.0

* `tratio()` replaces `ratiod()` as the package's front door, joining the
  `tulpa*` family of fitting verbs alongside `tulpa::tulpa()` and
  `tulpaObs::tobs()`. The `ratiod_*` family constructors, the `ratiod_fit`
  class, and every other `ratiod_`-prefixed name are unchanged.

* `tratio()` carries statistical arguments in its signature and every perf,
  numerical, and tuning knob in a single `control = list()`, matching the
  convention the engine and tulpaObs already follow. Moved into `control`:
  `chains`, `iter`, `warmup`, `thin`, `cores`, `seed`, `verbose`,
  `adapt_delta`, `max_treedepth`, `metric`, `riemannian`, `gradient_mode`,
  `re_param`, `vi_variant`, and the stochastic-gradient knobs (`batch_size`,
  `epsilon`, `alpha`, `schedule_*`, `use_schedule`) that previously arrived
  through `...`. An unrecognised knob is now an error rather than a silent
  no-op, and a `warmup` that would leave no post-warmup draws is rejected.

* `refresh` is removed. It was a documented argument that no backend ever
  read, so passing it changed nothing.

## 1.3.1

* `diagnostics()` replaces `mcmc_diagnostics()`, following the engine rename in
  tulpa 0.0.95. It delegates to `tulpa::diagnostics()`, which selects chain or
  approximation diagnostics from the fit's draws provenance rather than from the
  function's name. `mcmc_diagnostics()` is deprecated and still returns the same
  value.

## Bug fixes

* A fit no longer decides how many threads later fits get (#16). The Laplace
  and Polya-Gamma backends take a per-fit `cores`, which they applied by moving
  the process-wide OpenMP thread count and leaving it moved, and it defaults to
  one thread. Every region in the session that read that value afterwards was
  therefore sized by whichever of those fits ran last, in fits it knew nothing
  about. Each backend now restores the previous value when it returns, on the
  interrupt path as well as the normal one.

* Successive fits in one session no longer corrupt the heap on Windows (#8).
  The OpenMP team for chain-parallel work was sized per fit from `cores`, so a
  fit asking for fewer cores than the one before it shrank the team; libgomp
  then destroyed the surplus workers and ran their `thread_local` destructors
  while later work was still in flight, and the damage surfaced as
  `STATUS_HEAP_CORRUPTION` at an unrelated `free()`. Four chains followed by
  two was enough to trigger it. The team now only ever grows, sized by the
  widest core budget any fit has asked for, and `cores` bounds how many chains
  are in flight instead, so it keeps its meaning. Growing a team was always
  safe; only shrinking faulted. Its size never comes from
  `omp_get_max_threads()`: the Laplace and Polya-Gamma backends move that value
  through `omp_set_num_threads()` and leave it moved, so reading it let a
  Laplace fit ahead of the first chain fit pin the team at one thread and
  serialize the chains of every fit after it.

* The Gibbs spatial backend runs its chains in parallel under OpenMP (#7). It
  looped over chains in R, so a 4-chain fit cost close to four times a 1-chain
  fit; on a 40-site binomial ICAR model that ratio drops from 3.7x to 0.94x.
  `cores` now reaches this backend, bounded by the chain count and the machine.
  Chain seeds are still derived in R, so a fit with a given seed returns the
  same draws as before, and the parallel and serial paths agree bitwise.

* The handcoded gradient no longer disagrees with its own log posterior when a
  temporal effect is not shared between numerator and denominator. The
  vectorized and fused kernels added the temporal effect to the denominator's
  linear predictor and fed the denominator residual back into the temporal
  gradient, both unconditionally, while `compute_log_post()` applied the effect
  to the numerator alone. NUTS therefore sampled a density whose gradient
  pointed elsewhere, silently, for every family with a real denominator
  (`poisson_gamma`, `negbin_negbin`, `negbin_gamma`, `gamma_gamma`,
  `lognormal`). Binomial models were unaffected. The gradient checks now cover
  a continuous-denominator family with `shared = FALSE`, which is the only
  configuration that can see this.

* `ModelData`'s scalar members are initialised at declaration. The constructor
  set only `unique_id`, so roughly thirty members held indeterminate values
  until assigned; the GP entry point never assigned `re_parameterization` and
  read stack residue to choose between the centred and non-centred
  parameterisation.

* The scalar gradient fallback reduces its per-thread partial sums in
  thread-index order rather than in a critical section, whose summation order
  varies from run to run.

* `mcmc_diagnostics()` now reports per-parameter Rhat and ESS (#4). It handed a
  multi-parameter draws array to the single-variable `posterior::rhat()` /
  `ess_bulk()` / `ess_tail()`, collapsing every parameter to one scalar that was
  then recycled across the returned rows: a well-mixed four-chain fit reported
  `rhat = 2.124`, `ess_bulk = 1`. Diagnostics now delegate to
  `tulpa::mcmc_diagnostics()`, the engine's per-parameter implementation, which
  removes the local re-derivation (`compute_diagnostics_basic()`,
  `compute_split_rhat()`, `compute_ess_basic()`). `summary()`, `plot_rhat()`,
  `plot_ess()`, `diagnostic_summary()` and `check_diagnostics()` all inherit the
  correction, and a converged fit no longer prints a false non-convergence
  warning.

* `print()` on a `ratiod_diagnostic_summary` no longer errors when a fit has a
  parameter with Rhat > 1.01 or ESS < 400. The worst-Rhat and worst-ESS tables
  are column extracts of the diagnostics table and dispatched on
  `print.ratiod_diagnostics()`, which rounded columns the extract does not
  carry.

## Internal

* B1a PoC: plain binomial fits can route through tulpa's `LikelihoodSpec`
  path (`tulpa::get_nuts_fn()`) behind a feature flag
  (`options(tulpaRatio.use_specs = TRUE)` or `TULPARATIO_USE_SPECS=1`).
  Off by default; legacy backend remains the production path until full
  parity ships in B5. Posterior means agree with the legacy backend within
  Monte Carlo noise.

* B1b: the same feature flag now covers all 7 ratio families (binomial,
  poisson_gamma, negbin_gamma, negbin_negbin, gamma_gamma, lognormal,
  beta_binomial) for the simplest non-ZI / no-spatial / no-RE / single-chain
  configuration. Autodiff only — H-kernel port is deferred to B2.

* B1c: zero-inflation, hurdle, and one-inflation variants now route through
  the `LikelihoodSpec` path. Each family's templated likelihood dispatches on
  `data.zi_type` and calls a shared mixture helper in
  `src/lik_specs/lik_helpers.h`; per-family allowlist lives in
  `src/lik_dispatch.cpp` and `R/backend_hmc.R::SPEC_ZI_COMPAT`. Posterior
  parity with the legacy backend matches within Monte Carlo noise for every
  binomial × {ZI, hurdle, OI, ZOIB} combination; for count families
  (poisson_gamma, negbin_*) the legacy A_r path was already silently
  ignoring ZI, so the spec path here gives the mathematically correct
  posterior — recorded as a behaviour change, not a parity mismatch (see
  `dev_notes/B1c_zi_surface.md`). Autodiff only; ZI variants for the
  hand-coded H-kernel remain deferred.

* The tulpa ABI test no longer pins a hardcoded version number. It compared
  `cpp_tulpa_abi_version()` against a literal `32L`, a second copy of a
  constant tulpa already owns, so every tulpa ABI bump broke the test with a
  stale-literal failure that said nothing about compatibility. Two unguarded
  accessors (`cpp_tulpa_compiled_abi_version()`, the value baked in from the
  `LinkingTo` headers, and `cpp_tulpa_runtime_abi_version()`, the value in the
  loaded tulpa DLL) let the test assert the invariant that matters — that the
  two agree — and report both numbers when they do not. tulpa remains the
  single source of truth for the version.

* Documentation is regenerated under roxygen2 8.0.0, matching the rest of the
  packages that link against tulpa. `RoxygenNote` migrates to
  `Config/roxygen2/version`. Two external links now resolve to a topic alias
  rather than an Rd filename (`posterior::as_draws_df()`,
  `loo::stacking_weights()`), which clears the "Non-topic package-anchored
  link(s)" note from `R CMD check`; `INFERENCE_TIERS` drops the `\docType{data}`
  and `\format{}` entries roxygen2 8.0.0 no longer emits for documented values;
  and the package page lists authors alongside the maintainer.

# numdenom 1.3.0

## New Inference Backends

* **Tiered inference mode system**: Inference backends are now organized into
  three tiers based on epistemic guarantees:
  - Tier 1 (Exact): HMC, ESS, Polya-Gamma - diagnosable convergence
  - Tier 2 (Structured): Laplace - controlled approximation
  - Tier 3 (Optimized): VI, SGHMC, SGLD - no convergence guarantee (explicit opt-in)

* **Elliptical Slice Sampling (ESS)**: New Tier 1 backend for models with
  Gaussian priors. Use `mode = "ess"`.

* **Variational Inference (VI)**: New Tier 3 backend with mean-field, full-rank,
  and low-rank variants. Use `mode = "vi"`.

* **Stochastic Gradient MCMC**: New Tier 3 backends for very large data:
  - SGHMC: `mode = "sghmc"`
  - SGLD: `mode = "sgld"`

## Performance Improvements

* **Thread-safe autodiff**: A_t (tape autodiff) mode now uses RAII-based
  TapeScope for thread isolation. Parallel chains work correctly with all
  gradient modes.

* **Hand-coded gradients**: Added O(N) analytical gradients for:
  - GP spatial models (5.4x speedup)
  - Spatially varying coefficients (SVC)
  - Time-varying coefficients (TVC)
  - Correlated/uncorrelated random slopes
  - Crossed random effects
  - ZI/Hurdle binomial families
  - Latent factor models

* **Spatial prediction**: New `predict()` method for `ratiod_fit` objects
  supports prediction at new spatial locations.

## Internal Changes

* Forward-mode autodiff (`fwd::Dual`) for thread-safe gradient computation
* GPU backend stubs (preparation for future CUDA support)
* CRT-based negative binomial Polya-Gamma sampler

---

# numdenom 1.2.0

## New Features

### Spatiotemporal Interaction

* **Spatiotemporal interaction effects**: New `spatiotemporal()` function specifies
  space-time interactions using Knorr-Held (2000) Type I-IV models:
  - Type I: Unstructured (IID) interaction
  - Type II: Structured time at each location
  - Type III: Structured space at each time point
  - Type IV: Fully structured (Kronecker product)
  - Separable: For GP-based spatial/temporal effects

* **Non-separable spatiotemporal GP**: New `spatiotemporal_gp()` function for continuous
  space-time GP models with Gneiting or Cressie-Huang non-separable covariance.

* **Spatiotemporal effect extraction**: New `spatiotemporal_effects()` extracts posterior
  distributions in array, long, or summary format with visualization support.

### Zero-Inflation Variants

* **Zero-inflated binomial**: New `ratiod_zibinomial()` for proportions with excess zeros.

* **One-inflated binomial**: New `ratiod_oibinomial()` for proportions with excess ones
  (100% success/detection).

* **Zero-and-one inflated binomial (ZOIB)**: New `ratiod_zoibinomial()` for proportions
  with excess at both boundaries.

* **Hurdle binomial**: New `ratiod_hurdle_binomial()` for binomial data with hurdle
  at zero.

## Documentation

* New `vignette("spatial-temporal")` with examples of space-time modeling.
* New `vignette("zero-inflation")` covering ZI, OI, ZOIB, and hurdle binomial models.

---

# numdenom 1.1.0

## New Features

* **Latent factors for unmeasured confounders**: New `latent_factor()` function allows
  specifying latent factors that capture shared unmeasured variation between numerator
  and denominator processes. Latent factors are observation-level random effects with
  sum-to-zero constraints for identifiability. Use `latent = latent_factor(n_factors = 1)`
  in the `ratiod()` call.

* **Latent factor extraction**: New `latent_factors()` function extracts posterior
  summaries or full draws for latent factor scores from fitted models.

## Documentation

* Updated `vignette("random-effects")` with section on latent factors.

---

# numdenom 1.0.0

Initial release with:

* Three model families: `ratiod_negbin_negbin()`, `ratiod_binomial()`, `ratiod_poisson_gamma()`
* Native HMC/NUTS backend (no Stan dependency)
* Shared random effects (default) between numerator and denominator
* Spatial structure: CAR, BYM2, GP/NNGP, RSR
* Temporal structure: AR(1), RW1, cyclic
* Random slopes (correlated and uncorrelated)
* Nested and crossed random effects
* LOO/WAIC model comparison
* Zero-inflation support
