# numdenom

[![R-CMD-check](https://github.com/gcol33/numdenom/actions/workflows/R-CMD-check.yml/badge.svg)](https://github.com/gcol33/numdenom/actions/workflows/R-CMD-check.yml)
[![Codecov test coverage](https://codecov.io/gh/gcol33/numdenom/graph/badge.svg)](https://app.codecov.io/gh/gcol33/numdenom)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bayesian Hierarchical Modelling of Ratios, Rates, and Proportions**

The `numdenom` package provides a principled framework for modelling ratios, rates, and proportions in ecological and related data. Unlike traditional approaches that model ratios directly, `numdenom` performs inference on the latent processes generating the numerator and denominator, computing ratios post hoc with full uncertainty propagation.

## Philosophy

**Ratios are not data. They are derived quantities.**

All inference is performed on the latent processes generating the numerator and denominator, never on their quotient. This approach:

- Prevents spurious ratio effects from unmeasured confounders
- Naturally handles shared structure between numerator and denominator
- Propagates uncertainty correctly through the ratio computation
- Avoids the statistical pathologies of ratio regression

## Quick Start

```r
library(numdenom)

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

## Statement of Need

Ratios, rates, and proportions are ubiquitous in ecological and biological research: relative abundance, catch-per-unit-effort (CPUE), survival rates, proportional cover. Yet standard statistical approaches—modelling ratios directly or using offsets—can produce biased inference when numerator and denominator share unmeasured drivers.

`numdenom` addresses this by:

1. **Modelling both components jointly** with shared random effects by default
2. **Computing ratios post hoc** from the full posterior, preserving uncertainty
3. **Providing spatial and temporal extensions** for structured data
4. **Using native HMC/NUTS sampling** with no external dependencies

This makes the package useful for:

- Relative abundance modelling in community ecology
- CPUE standardization in fisheries science
- Proportional cover in vegetation surveys
- Any application where the ratio's components share latent structure

## Features

### Model Families

| Family | Numerator | Denominator | Use Case |
|--------|-----------|-------------|----------|
| `ratiod_negbin_negbin()` | NegBin | NegBin | Count/count ratios (relative abundance) |
| `ratiod_binomial()` | Binomial | Fixed trials | Success/trial proportions |
| `ratiod_poisson_gamma()` | Poisson | Gamma | CPUE (count/continuous effort) |

### Hierarchical Structure

- **Shared random effects** by default — prevents spurious ratio effects
- **Random slopes**: correlated `(1 + x | site)` and uncorrelated `(1 + x || site)`
- **Nested RE**: `(1 | site/plot)` with automatic expansion
- **Crossed RE**: `(1 | site) + (1 | year)` for multi-way structure

### Spatial and Temporal

- **CAR priors**: `spatial_car()` for areal data
- **BYM2 priors**: `spatial_bym2()` for combined spatial + IID effects
- **Temporal AR(1)**: `temporal_ar1()` for autocorrelated time series
- **Random walk**: `temporal_rw()` for flexible temporal trends

### Inference

- **Native HMC/NUTS backend** — no Stan or JAGS required
- **Laplace approximation** for very large datasets (N > 50k)
- **Pólya-Gamma Gibbs** for binomial models (experimental)
- **LOO and WAIC** for model comparison

## Installation

```r
# Install development version from GitHub
# install.packages("pak")
pak::pak("gcol33/numdenom")
```

## Usage Examples

### Basic: Count Ratios

```r
library(numdenom)

# Relative abundance of a focal species
fit <- ratiod(
  focal_count | total_count ~ treatment + (1 | site),
  data = abundance_data,
  family = ratiod_negbin_negbin()
)

# Summary of model fit
summary(fit)

# Extract ratio posteriors with credible intervals
ratio(fit)

# Compare treatment levels
ratio_contrast(fit, "treatment")
```

### CPUE with Continuous Effort

```r
# Catch-per-unit-effort with varying tow duration
fit <- ratiod(
  catch | tow_hours ~ depth + season + (1 | vessel),
  data = fisheries_data,
  family = ratiod_poisson_gamma()
)

# Posterior predictive check
pp_check(fit)
```

### Binomial Proportions

```r
# Survival rates
fit <- ratiod(
  survivors | n_released ~ treatment + (1 | cohort),
  data = survival_data,
  family = ratiod_binomial()
)
```

### Spatial Models

```r
# Relative abundance with spatial structure
fit <- ratiod(
  species_count | total_count ~ habitat,
  data = grid_data,
  family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_matrix, group_var = "grid_id")
)
```

### Shared vs Independent Structure

```r
# Default: shared random effects (recommended)
fit_shared <- ratiod(
  num | denom ~ x + (1 | group),
  data = df,
  family = ratiod_negbin_negbin()
)

# Explicit independence (triggers warning)
fit_indep <- ratiod(
  num | denom ~ x + (1 | group),
  data = df,
  family = ratiod_negbin_negbin(),
  shared = ~ 0
)
```

### Process-Specific Predictors

```r
# Different predictors for numerator and denominator
fit <- ratiod(
  species_count | total_count ~ (1 | site),
  formula_num = ~ depth + temperature,
  formula_denom = ~ sampling_effort,
  data = df,
  family = ratiod_negbin_negbin()
)
```

## Comparison with Alternative Approaches

| Approach | Shared Structure | Uncertainty | Flexibility |
|----------|-----------------|-------------|-------------|
| Ratio regression | ✗ | ✗ | Low |
| GLM with offset | ✗ | Partial | Medium |
| Separate models | ✗ | ✗ | High |
| **numdenom** | ✓ (default) | ✓ | High |

## Documentation

- [Quick Start](https://gillescolling.com/numdenom/articles/getting-started.html) — Installation and basic usage
- [Why Ratios Are Not Data](https://gillescolling.com/numdenom/articles/philosophy.html) — Statistical motivation
- [Complete Workflows](https://gillescolling.com/numdenom/articles/workflows.html) — Real-world analysis examples
- [Spatial and Temporal Models](https://gillescolling.com/numdenom/articles/spatial-temporal.html) — CAR, BYM2, AR(1), random walk
- [Random Effects](https://gillescolling.com/numdenom/articles/random-effects.html) — Random slopes, nested/crossed RE

## Support

> "Software is like sex: it's better when it's free." — Linus Torvalds

I'm a PhD student who builds R packages in my free time because I believe good tools should be free and open. I started these projects for my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to say thanks. It helps with my coffee addiction.

[![Buy Me A Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT (see the LICENSE.md file)

## Citation

```bibtex
@software{numdenom,
  author = {Colling, Gilles},
  title = {numdenom: Bayesian Hierarchical Modelling of Ratios, Rates, and Proportions},
  year = {2025},
  url = {https://github.com/gcol33/numdenom}
}
```
