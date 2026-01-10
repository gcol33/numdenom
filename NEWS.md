# ratiod 1.1.0

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

# ratiod 1.0.0

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
