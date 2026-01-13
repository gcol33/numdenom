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
