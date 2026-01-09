# ratiod

Bayesian hierarchical modelling of ratios, rates, and proportions in R.

## Philosophy

**Ratios are not data. They are derived quantities.**

All inference is performed on the latent processes generating the numerator and denominator, never on their quotient. Ratios, rates, and proportions are computed post hoc with full uncertainty propagation.

## Installation

```r
pak::pak("gcol33/ratiod")
```

## Quick Start

```r
library(ratiod)

# Fit a model for count ratios (e.g., relative abundance)
fit <- ratiod(
  species_count | total_count ~ habitat + (1 | site),
  data = df,
  family = ratiod_negbin_negbin()
)

# Extract ratio posteriors
ratio(fit)

# Compare conditions
ratio_contrast(fit, "habitat")
```

## Features

- **Three model families**: `ratiod_negbin_negbin()`, `ratiod_binomial()`, `ratiod_poisson_gamma()`
- **Shared random effects** by default - prevents spurious ratio effects
- **Spatial priors**: CAR and BYM2 for areal data
- **Temporal effects**: AR(1) and random walk
- **Random slopes**: correlated `(1 + x | site)` and uncorrelated `(1 + x || site)`
- **Nested RE**: `(1 | site/plot)` with automatic expansion
- **Native HMC/NUTS backend** - no external dependencies
- **Model comparison**: LOO and WAIC

## Documentation

- [Quick Start](https://gillescolling.com/ratiod/articles/getting-started.html) - Installation and basic usage
- [Why Ratios Are Not Data](https://gillescolling.com/ratiod/articles/philosophy.html) - Statistical motivation
- [Complete Workflows](https://gillescolling.com/ratiod/articles/workflows.html) - Real-world analysis examples
- [Spatial and Temporal Models](https://gillescolling.com/ratiod/articles/spatial-temporal.html) - CAR, BYM2, AR(1), random walk
- [Random Effects](https://gillescolling.com/ratiod/articles/random-effects.html) - Random slopes, nested/crossed RE

## License

MIT
