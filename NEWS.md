# tulpaRatio (development)

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
