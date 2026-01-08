# quotr Roadmap

## Current Status: v0.8.0

### Implemented Features
- **Three model families:** `quotr_negbin_negbin()`, `quotr_binomial()`, `quotr_poisson_gamma()`
- **Zero-inflated families:** `quotr_zinegbin()`, `quotr_zipois()`, `quotr_hurdle_negbin()`, `quotr_hurdle_pois()`
- **Inference backends:**
  - HMC/NUTS (default) - custom C++ implementation, no external dependencies
  - Pólya-Gamma Gibbs sampler - fast for binomial models
  - Laplace approximation - very fast approximate inference
- **Hierarchical structure:**
  - Shared random effects (default behavior)
  - Group-level random intercepts
- **Spatial models:**
  - ICAR (Intrinsic CAR)
  - BYM2 (Besag-York-Mollié 2)
- **Temporal models:**
  - RW1 (first-order random walk)
  - RW2 (second-order random walk)
  - AR(1) (autoregressive)
  - Cyclic variants
- **Zero-inflation:**
  - Zero-inflated Poisson/NegBin
  - Hurdle Poisson/NegBin
  - Separate ZI regression formula
- **Performance optimizations:**
  - OpenMP parallelization (likelihood + gradient)
  - SIMD-friendly loop unrolling
  - Optimized linear algebra routines
  - Parallel multi-chain sampling
- **Model comparison:** LOO-CV, WAIC via `loo` package
- **Diagnostics:** `pp_check()`, `mcmc_diagnostics()`, `check_diagnostics()`
- **Inference:** `ratio()`, `ratio_contrast()`, `fitted()`, `predict()`
- **Benchmarking:** `quotr_benchmark()`, `quotr_benchmark_compare()`, `quotr_threads()`

---

## v0.6.0: Temporal Structure

### Goal
Add temporal random effects for time-series and panel data.

### Features

#### 1. AR(1) Temporal Effects
```r
quotr(

  count | effort ~ x + temporal_ar1(time, group = site),
  data = df,
  family = quotr_poisson_gamma()
)
```

- Autoregressive structure: `φ[t] = ρ * φ[t-1] + ε[t]`
- Shared across numerator/denominator (default) or separate
- Prior on ρ: `ρ ~ Uniform(-1, 1)` or `Beta` transformed

#### 2. Random Walk (RW1/RW2)
```r
quotr(
  count | effort ~ x + temporal_rw1(year),
  data = df,
  family = quotr_poisson_gamma()
)
```

- RW1: First-order random walk `φ[t] - φ[t-1] ~ N(0, σ²)`
- RW2: Second-order (smoothing) `φ[t] - 2φ[t-1] + φ[t-2] ~ N(0, σ²)`
- Intrinsic (improper) or proper versions

#### 3. Seasonal Effects
```r
quotr(
  count | effort ~ x + temporal_seasonal(month, period = 12),
  data = df,
  family = quotr_poisson_gamma()
)
```

### Implementation

**New files:**
```
R/
├── temporal.R           # temporal_ar1(), temporal_rw1(), temporal_rw2()
├── temporal_seasonal.R  # temporal_seasonal()

src/
├── temporal_ar1.cpp     # AR(1) precision matrix and sampling
├── temporal_rw.cpp      # Random walk precision matrices
```

**Key algorithms:**
- AR(1): Tridiagonal precision matrix, efficient Cholesky
- RW1/RW2: Sparse band matrices, sum-to-zero constraint
- All backends (HMC, PG, Laplace) need updates

### API Design
```r
# AR(1) with estimated autocorrelation
temporal_ar1(time_var, group = NULL, rho_prior = NULL)

# Random walk
temporal_rw1(time_var, cyclic = FALSE)
temporal_rw2(time_var, cyclic = FALSE)

# Seasonal
temporal_seasonal(time_var, period, type = c("rw1", "ar1"))
```

---

## v0.7.0: Zero-Inflation

### Goal
Handle excess zeros in count data, common in ecological surveys.

### Features

#### 1. Zero-Inflated Families
```r
quotr(
  count | effort ~ x + (1 | site),
  data = df,
  family = quotr_zinegbin()  # Zero-inflated negative binomial
)
```

**New families:**
- `quotr_zipois()` - Zero-inflated Poisson (numerator)
- `quotr_zinegbin()` - Zero-inflated negative binomial
- `quotr_zinegbin_negbin()` - ZI numerator, regular denominator

#### 2. Hurdle Models
```r
quotr(
  count | effort ~ x + (1 | site),
  data = df,
  family = quotr_hurdle_negbin()
)
```

- Separate processes for zero vs positive
- More interpretable for "structural" zeros

#### 3. Zero-Inflation Covariates
```r
quotr(
  count | effort ~ x + (1 | site),
  zi = ~ habitat_type,  # Predictors for P(zero)
  data = df,
  family = quotr_zinegbin()
)
```

### Implementation

**Model structure:**
```
P(Y = 0) = π + (1 - π) * P(Y = 0 | count process)
P(Y = y) = (1 - π) * P(Y = y | count process), y > 0

logit(π) = W * γ  # Zero-inflation linear predictor
```

**New files:**
```
R/
├── family_zi.R          # Zero-inflated family definitions

src/
├── zi_likelihood.cpp    # ZI likelihood contributions
├── zi_hmc.cpp          # HMC updates for ZI parameters
```

**Challenges:**
- Identifiability: Need informative priors or constraints
- Mixing: ZI models can have multimodal posteriors
- Ratio interpretation: What does ratio mean with ZI?

### API Design
```r
# Zero-inflated negative binomial
quotr_zinegbin(zi_link = "logit")

# With ZI covariates
quotr(
  y | n ~ x,
  zi = ~ z,  # Zero-inflation formula
  family = quotr_zinegbin()
)

# Hurdle model
quotr_hurdle_negbin()
```

---

## v0.8.0: Performance & Scalability

### Goal
GPU acceleration and optimizations for large datasets (100k+ observations).

### Features

#### 1. GPU-Accelerated HMC
```r
quotr(
  ...,
  backend = "hmc",
  control = list(gpu = TRUE)
)
```

- CUDA/OpenCL for gradient computation
- Batch likelihood evaluation
- Target: 10-50x speedup for N > 50k

#### 2. Sparse Matrix Optimizations
- Exploit sparsity in spatial precision matrices
- Sparse Cholesky for Laplace backend
- Memory-efficient storage for large CAR models

#### 3. Parallel Chains
```r
quotr(
  ...,
  chains = 4,
  cores = 4,  # Parallel chain execution
  backend = "hmc"
)
```

- Already implemented for HMC spatial
- Extend to all backends

#### 4. Variational Inference (Optional)
```r
quotr(
  ...,
  backend = "vi"  # Mean-field variational
)
```

- Fast approximate posterior
- Useful for model selection/screening

### Implementation

**Dependencies:**
```
Suggests:
    torch,      # For GPU via torch
    gpuR        # Alternative GPU backend
```

**New files:**
```
src/
├── gpu_likelihood.cu    # CUDA kernels
├── gpu_gradient.cu
├── sparse_cholesky.cpp  # Optimized sparse operations
```

---

## v0.9.0: Extended Families & Polish

### Goal
Additional model families and API refinements before v1.0.

### Features

#### 1. Additional Families
- `quotr_beta_binomial()` - Overdispersed proportions
- `quotr_gamma_gamma()` - Continuous ratio data
- `quotr_lognormal()` - Log-normal responses

#### 2. Prior Specification Refinements
```r
quotr_priors(
  beta = prior_normal(0, 2.5),
  sigma = prior_half_cauchy(2.5),
  phi = prior_gamma(2, 0.1),
  rho_temporal = prior_beta(2, 2)  # NEW
)
```

#### 3. Model Averaging
```r
quotr_average(fit1, fit2, fit3, weights = "loo")
```

#### 4. Comprehensive Documentation
- Full package vignette
- Case study vignettes (ecology, epidemiology)
- Function reference improvements

---

## v1.0.0: Stable Release

### Requirements for v1.0
1. **All features from v0.6-v0.9 implemented and tested**
2. **API stability guarantee** - no breaking changes after v1.0
3. **Comprehensive test coverage** (>90%)
4. **Full documentation** with vignettes
5. **CRAN submission ready**
6. **Performance benchmarks** published

### Release Checklist
- [ ] All backends working: HMC, PG, Laplace
- [ ] Temporal: AR(1), RW1, RW2, seasonal
- [ ] Zero-inflation: ZI-Poisson, ZI-NegBin, hurdle
- [ ] Spatial: ICAR, BYM2
- [ ] GPU support (optional dependency)
- [ ] `fitted()`, `predict()`, `ratio()`, `ratio_contrast()`
- [ ] `pp_check()`, `loo()`, `waic()`
- [ ] Vignettes: getting-started, philosophy, temporal, spatial, zero-inflation
- [ ] pkgdown site
- [ ] CRAN checks pass

---

## Future (v1.1+)

### Potential Features
- **Continuous spatial:** Gaussian Process / Matérn / SPDE
- **Multivariate responses:** >2 linked processes
- **Missing data:** Multiple imputation integration
- **Prediction intervals:** For new observations
- **Bayesian model selection:** Bayes factors, posterior model probabilities

---

## Version Summary

| Version | Focus | Key Features | Status |
|---------|-------|--------------|--------|
| v0.5.0 | Foundation | Core models, 3 backends, spatial | ✅ Complete |
| v0.6.0 | Temporal | AR(1), RW1/RW2, cyclic | ✅ Complete |
| v0.7.0 | Zero-inflation | ZI-Poisson, ZI-NegBin, hurdle | ✅ Complete |
| v0.8.0 | Performance | SIMD, OpenMP, benchmarking | ✅ Complete |
| v0.9.0 | Polish | Extra families, priors, docs | Planned |
| v1.0.0 | Stable | CRAN release | Planned |

## Priority Order

For ecological applications (the primary use case), the recommended implementation order is:

1. **v0.6.0 Temporal** - Most ecological data has temporal structure
2. **v0.7.0 Zero-inflation** - Common in species occurrence/abundance
3. **v0.8.0 Performance** - Enables analysis of large monitoring datasets
4. **v0.9.0 Polish** - Refinements based on user feedback
