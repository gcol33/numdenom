# ratiod Roadmap

## Current Status: v0.9.3

### Implemented Features
- **Core model families:** `ratiod_negbin_negbin()`, `ratiod_binomial()`, `ratiod_poisson_gamma()`
- **Extended families (v0.9):** `ratiod_beta_binomial()`, `ratiod_gamma_gamma()`, `ratiod_lognormal()`
- **Zero-inflated families:** `ratiod_zinegbin()`, `ratiod_zipois()`, `ratiod_hurdle_negbin()`, `ratiod_hurdle_pois()`
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
  - **Spatially-Varying Coefficients (v0.9.3)** - NNGP-based SVCs via `spatial_svc()`
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
- **Model averaging (v0.9):** `ratiod_average()` with stacking/pseudo-BMA weights
- **Prior specification (v0.9):** Enhanced `ratiod_priors()` with `prior_*()` helper functions
- **Prior helpers (v0.9.2):** `priors_default()` for inspecting defaults, `sim_ratiod()` for simulation
- **tidybayes compatibility (v0.9.2):** `as_draws()`, `spread_draws()`, `gather_draws()`, `point_interval()`
- **Diagnostics:** `pp_check()`, `mcmc_diagnostics()`, `check_diagnostics()`
- **Inference:** `ratio()`, `ratio_contrast()`, `fitted()`, `predict()`
- **Benchmarking:** `ratiod_benchmark()`, `ratiod_benchmark_compare()`, `ratiod_threads()`

---

## v0.6.0: Temporal Structure

### Goal
Add temporal random effects for time-series and panel data.

### Features

#### 1. AR(1) Temporal Effects
```r
ratiod(

  count | effort ~ x + temporal_ar1(time, group = site),
  data = df,
  family = ratiod_poisson_gamma()
)
```

- Autoregressive structure: `φ[t] = ρ * φ[t-1] + ε[t]`
- Shared across numerator/denominator (default) or separate
- Prior on ρ: `ρ ~ Uniform(-1, 1)` or `Beta` transformed

#### 2. Random Walk (RW1/RW2)
```r
ratiod(
  count | effort ~ x + temporal_rw1(year),
  data = df,
  family = ratiod_poisson_gamma()
)
```

- RW1: First-order random walk `φ[t] - φ[t-1] ~ N(0, σ²)`
- RW2: Second-order (smoothing) `φ[t] - 2φ[t-1] + φ[t-2] ~ N(0, σ²)`
- Intrinsic (improper) or proper versions

#### 3. Seasonal Effects
```r
ratiod(
  count | effort ~ x + temporal_seasonal(month, period = 12),
  data = df,
  family = ratiod_poisson_gamma()
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
ratiod(
  count | effort ~ x + (1 | site),
  data = df,
  family = ratiod_zinegbin()  # Zero-inflated negative binomial
)
```

**New families:**
- `ratiod_zipois()` - Zero-inflated Poisson (numerator)
- `ratiod_zinegbin()` - Zero-inflated negative binomial
- `ratiod_zinegbin_negbin()` - ZI numerator, regular denominator

#### 2. Hurdle Models
```r
ratiod(
  count | effort ~ x + (1 | site),
  data = df,
  family = ratiod_hurdle_negbin()
)
```

- Separate processes for zero vs positive
- More interpretable for "structural" zeros

#### 3. Zero-Inflation Covariates
```r
ratiod(
  count | effort ~ x + (1 | site),
  zi = ~ habitat_type,  # Predictors for P(zero)
  data = df,
  family = ratiod_zinegbin()
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
ratiod_zinegbin(zi_link = "logit")

# With ZI covariates
ratiod(
  y | n ~ x,
  zi = ~ z,  # Zero-inflation formula
  family = ratiod_zinegbin()
)

# Hurdle model
ratiod_hurdle_negbin()
```

---

## v0.8.0: Performance & Scalability

### Goal
GPU acceleration and optimizations for large datasets (100k+ observations).

### Features

#### 1. GPU-Accelerated HMC
```r
ratiod(
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
ratiod(
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
ratiod(
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
- `ratiod_beta_binomial()` - Overdispersed proportions
- `ratiod_gamma_gamma()` - Continuous ratio data
- `ratiod_lognormal()` - Log-normal responses

#### 2. Prior Specification Refinements
```r
ratiod_priors(
  beta = prior_normal(0, 2.5),
  sigma = prior_half_cauchy(2.5),
  phi = prior_gamma(2, 0.1),
  rho_temporal = prior_beta(2, 2)  # NEW
)
```

#### 3. Model Averaging
```r
ratiod_average(fit1, fit2, fit3, weights = "loo")
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

## v0.9.3: Spatially-Varying Coefficients

### Goal
Allow regression coefficients to vary smoothly across space, capturing local effects.

### Features

#### 1. SVC Specification
```r
ratiod(
  count | effort ~ depth + temp + (1 | site),
  data = df,
  family = ratiod_poisson_gamma(),
  svc = spatial_svc(~ lon + lat, terms = c("(Intercept)", "depth"))
)
```

- NNGP (Nearest Neighbor Gaussian Process) for computational efficiency
- Multiple terms can vary spatially
- Covariance functions: exponential, Matérn, Gaussian, spherical

#### 2. SVC Extraction and Visualization
```r
# Extract SVC posteriors
svc_post <- svc(fit)
summary(svc_post)

# Plot spatial surface
plot(svc_post, "depth", type = "mean")
plot(svc_post, "depth", type = "sd")
```

### Implementation
- R layer: `spatial_svc()`, `validate_svc()`, `compute_nngp_neighbors()`
- C++ layer: `hmc_svc.h` with NNGP likelihood computation
- Extraction: `svc()` generic with plotting and summary methods

### Status: ✅ Complete

---

## v0.9.4: Continuous Spatial GP & Multi-Scale

**Goal:** Full Gaussian Process spatial effects and multi-scale decomposition.

**Status:** ✅ R layer complete (C++ backend integration pending)

### Features

#### 1. GP Specification
```r
ratiod(
  count | effort ~ x + (1 | site),
  data = df,
  spatial = spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
)
```

- Matérn covariance with estimated range (phi) and variance (sigma2)
- Smoothness parameter nu: 0.5 (exponential), 1.5, 2.5 (common choices)
- NNGP approximation for scalability (builds on SVC infrastructure)
- Shared between numerator/denominator by default

#### 2. Covariance Options
- `"exponential"` - Matérn with nu = 0.5 (default, rough)
- `"matern"` - Matérn with configurable nu (1.5 = once differentiable)
- `"gaussian"` - Squared exponential (very smooth)
- `"spherical"` - Compact support, zero beyond range

#### 3. Prediction at New Locations
```r
# Predict spatial effect at unobserved locations
predict(fit, newdata = grid, type = "spatial")
```

### Implementation
- R layer: `spatial_gp()` specification function
- C++ layer: Extend `hmc_svc.h` for single spatial field (simpler than SVC)
- Reuse NNGP neighbor computation from SVC
- ~5-10% overhead vs ICAR for N < 10k; critical for N > 100k

---

## v0.9.5: Multi-Scale Spatial

**Goal:** Separate local (plot-to-plot) and regional (landscape) spatial effects.

### Features

#### 1. Multi-Scale Specification
```r
ratiod(
  count | effort ~ x,
  data = df,
  spatial = spatial_multiscale(
    ~ lon + lat,
    scales = c("local", "regional"),
    range_local = c(0.1, 5),      # km, prior range
    range_regional = c(10, 100)   # km, prior range
  )
)
```

- Two independent spatial fields at different scales
- PC priors on variance components favor simpler (single-scale) models
- Range parameters constrained to prevent scale overlap

#### 2. Additive Decomposition
```
η(s) = η_local(s) + η_regional(s)

η_local ~ GP(0, C_local)      # Fine-scale variation
η_regional ~ GP(0, C_regional) # Broad-scale gradients
```

#### 3. Scale-Specific Extraction
```r
# Extract each scale separately
spatial_effects <- spatial(fit)
plot(spatial_effects, scale = "local")
plot(spatial_effects, scale = "regional")
```

### Implementation
- Two NNGP fields with different neighbor counts (k_local < k_regional)
- Separate range/variance parameters per scale
- PC priors: P(sigma_local > sigma_regional) constrains hierarchy
- Computational: ~30% overhead vs single-scale GP

### Design Considerations
With 3M observations:
- Identifiability is strong - scales are clearly separable
- Vecchia approximation essential (O(n·k²) not O(n³))
- Local scale: k = 10-15 neighbors
- Regional scale: k = 30-50 neighbors (larger range)

---

## v0.9.6: Multi-Scale Temporal

**Goal:** Separate short-term fluctuations from long-term trends.

### Features

#### 1. Temporal Decomposition
```r
ratiod(
  count | effort ~ x,
  data = df,
  temporal = temporal_multiscale(
    time_var = year,
    trend = "rw2",           # Long-term smooth trend
    seasonal = 12,           # Monthly seasonality (if applicable
    short_term = "ar1"       # Year-to-year deviations
  )
)
```

#### 2. Component Structure
```
η(t) = trend(t) + seasonal(t) + short(t)

trend ~ RW2              # Smooth long-term change
seasonal ~ cyclic_RW1    # Repeating pattern
short ~ AR(1) or IID     # Residual temporal correlation
```

#### 3. Component Extraction
```r
temporal_effects <- temporal(fit)
plot(temporal_effects, component = "trend")
plot(temporal_effects, component = "seasonal")
```

### Implementation
- Stack multiple temporal precision matrices
- Sum-to-zero constraints on each component
- Variance allocation priors prevent over-fitting

---

## v0.9.7: Spatial Confounding Mitigation

**Goal:** Prevent spatial effects from absorbing covariate information.

### Features

#### 1. Restricted Spatial Regression (RSR)
```r
ratiod(
  count | effort ~ depth + temp + (1 | site),
  data = df,
  spatial = spatial_gp(~ lon + lat, restrict_to = ~ depth + temp)
)
```

- Orthogonalizes spatial field to covariate space
- Preserves covariate coefficient interpretation
- Spatial effect captures only residual spatial structure

#### 2. Spatial+ Approach
```r
ratiod(
  count | effort ~ depth + temp,
  data = df,
  spatial = spatial_gp(~ lon + lat, type = "spatial+")
)
```

- Decomposes covariates into spatial + non-spatial components
- Explicitly models spatial confounding
- More parameters but clearer interpretation

### Why This Matters
- Without RSR, spatial random effects can absorb covariate effects
- Leads to attenuated (biased toward zero) coefficient estimates
- Critical for causal interpretation in observational studies

---

## Future (v1.1+)

### Other Potential Features
- **Spatiotemporal interaction:** Space × time varying effects
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
| v0.9.0 | Polish | Extra families, priors, model averaging | ✅ Complete |
| v0.9.2 | Pro features | priors_default(), sim_ratiod(), tidybayes | ✅ Complete |
| v0.9.3 | SVC | Spatially-varying coefficients with NNGP | ✅ Complete |
| v0.9.4 | Multi-scale | GP spatial, multi-scale spatial/temporal | ✅ R layer |
| v1.0.0 | Stable | CRAN release | Planned |

## Priority Order

For ecological applications (the primary use case), the recommended implementation order is:

1. **v0.6.0 Temporal** - Most ecological data has temporal structure
2. **v0.7.0 Zero-inflation** - Common in species occurrence/abundance
3. **v0.8.0 Performance** - Enables analysis of large monitoring datasets
4. **v0.9.0 Polish** - Refinements based on user feedback
