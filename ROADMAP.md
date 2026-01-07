# quotr Roadmap

## v1.0 (Current)
- Stan backend via cmdstanr
- Three families: negbin_negbin, binomial, poisson_gamma
- Shared random effects (default)
- Spatial CAR/BYM2
- LOO/WAIC model comparison

---

## v1.1: Pólya-Gamma Backend for Binomial

### Goal
Fast Gibbs sampler for `quotr_binomial()` using Pólya-Gamma data augmentation.
Targets 5-10x speedup over Stan for binomial models.

### Background
Pólya-Gamma augmentation (Polson et al., 2013) enables efficient Gibbs sampling
for logistic/binomial models by introducing auxiliary variables that yield
conjugate normal updates for regression coefficients.

### Architecture

```
quotr(..., family = quotr_binomial(), backend = "pg")
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   pg_sampler.cpp        │
                              │   (Rcpp implementation) │
                              └─────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            ┌──────────────┐      ┌──────────────────┐     ┌──────────────┐
            │ sample_omega │      │ sample_beta      │     │ sample_re    │
            │ (PG draws)   │      │ (conjugate)      │     │ (conjugate)  │
            └──────────────┘      └──────────────────┘     └──────────────┘
```

### Key Components

#### 1. Pólya-Gamma Sampler (`src/pg_sampler.cpp`)
```cpp
// Core PG(1, z) sampler using Devroye method for small z
// and saddle-point approximation for large z
double rpg1(double z);

// Vectorized version
NumericVector rpg(int n, NumericVector z);
```

#### 2. Gibbs Sampler for Binomial Model (`src/pg_binomial.cpp`)
```cpp
// One Gibbs iteration
List pg_binomial_step(
  IntegerVector y,      // successes
  IntegerVector n,      // trials
  NumericMatrix X,      // design matrix
  NumericVector beta,   // current coefficients
  NumericVector omega,  // current PG auxiliary vars
  // ... random effects, spatial
);

// Full sampler
List pg_binomial_fit(
  IntegerVector y,
  IntegerVector n,
  NumericMatrix X,
  List re_structure,
  List spatial,
  List priors,
  int n_iter,
  int n_warmup,
  int thin
);
```

#### 3. R Interface (`R/backend_pg.R`)
```r
# Dispatch to PG backend
quotr_fit_pg <- function(formula_spec, family, data, spatial, priors, ...) {

  # Prepare data
  stan_data <- make_standata(formula_spec, family, data, spatial, priors)

  # Call C++ sampler
  samples <- pg_binomial_fit(
    y = stan_data$y_num,
    n = stan_data$y_denom,
    X = stan_data$X,
    re_structure = list(
      idx = stan_data$re_idx,
      n_groups = stan_data$n_re_groups,
      shared = stan_data$re_shared
    ),
    spatial = if (stan_data$use_spatial) list(
      node1 = stan_data$node1,
      node2 = stan_data$node2
    ) else NULL,
    priors = list(
      beta_sd = priors$beta_sd,
      sigma_U = priors$sigma_U,
      sigma_alpha = priors$sigma_alpha
    ),
    n_iter = iter,
    n_warmup = warmup,
    thin = thin
  )

  # Convert to posterior::draws format
  as_quotr_fit(samples, formula_spec, family, data)
}
```

### Algorithm Details

For binomial model: Y_i ~ Binomial(n_i, p_i) with logit(p_i) = η_i

**Augmented model:**
1. Introduce ω_i ~ PG(n_i, η_i)
2. Define κ_i = (y_i - n_i/2)
3. Conditional on ω, the model is Gaussian:
   - κ_i = ω_i × η_i + ε_i where ε_i ~ N(0, 1/ω_i)

**Gibbs updates:**
1. Sample ω_i | β, re ~ PG(n_i, η_i)
2. Sample β | ω, re ~ N(μ_β, Σ_β) [conjugate update]
3. Sample re | ω, β ~ N(μ_re, Σ_re) [conjugate update]
4. Sample σ² | re ~ InverseGamma [conjugate]

### Random Effects with PG

Shared random effects:
```
η_num[i] = X[i,] β + b[group[i]]
η_denom[i] = ... (fixed for binomial, denominator is known)

p(b | σ²) = N(0, σ² I)
p(b | ω, y, β, σ²) = N(μ_b, Σ_b)

Σ_b = (Z'ΩZ + I/σ²)^{-1}
μ_b = Σ_b Z'κ
```

### Spatial CAR with PG

For CAR spatial effects φ:
```
p(φ | τ²) ∝ exp(-φ'Qφ / 2τ²)

p(φ | ω, y, β, τ²) = N(μ_φ, Σ_φ)
Σ_φ = (D'ΩD + Q/τ²)^{-1}
```

Where Q is the CAR precision matrix (sparse).

### Files to Create

```
src/
├── pg_rng.cpp           # Pólya-Gamma random number generator
├── pg_rng.h
├── pg_binomial.cpp      # Binomial Gibbs sampler
├── pg_binomial.h
├── sparse_utils.cpp     # Sparse matrix operations for CAR
└── RcppExports.cpp

R/
├── backend_pg.R         # PG backend dispatcher
├── pg_diagnostics.R     # Convergence diagnostics for Gibbs
└── zzz.R               # Update to register C++ routines
```

### Testing Plan

1. **Unit tests for PG sampler**
   - Compare PG(1, z) samples to known distribution
   - Test moment matching

2. **Comparison with Stan**
   - Same model, same data, same priors
   - Compare posterior means ± 2 MCMC SE
   - Verify coverage of credible intervals

3. **Performance benchmarks**
   - N = 1000, 10000, 100000 observations
   - Measure ESS/second vs Stan

### API Changes

```r
# New backend argument
quotr(
  y | n ~ x + (1 | site),
  data = df,
  family = quotr_binomial(),
  backend = "pg"  # NEW: "stan" (default) or "pg"
)
```

### Dependencies

Add to DESCRIPTION:
```
LinkingTo: Rcpp, RcppEigen
```

### Timeline Estimate
- PG sampler core: 2 days
- Binomial Gibbs with RE: 3 days
- CAR spatial integration: 2 days
- Testing & benchmarks: 2 days
- Documentation: 1 day
- **Total: ~2 weeks**

---

## v1.2: Laplace Backend for Large Areal Data

### Goal
Fast approximate inference for large datasets (10k-100k+ observations) using
nested Laplace approximation. Focus on areal models (CAR/BYM2) only,
no continuous spatial (SPDE).

### Why Custom Implementation?
- No CRAN dependency issues (R-INLA not on CRAN)
- Tailored for quotr's bivariate structure
- Lighter weight (no SPDE mesh machinery)
- Full control over approximation quality

### Architecture

```
quotr(..., backend = "laplace")
            │
            ▼
┌───────────────────────────────────────────────────────┐
│                   Laplace Engine                       │
├───────────────────────────────────────────────────────┤
│  1. Hyperparameter grid/optimization                   │
│  2. For each θ: Laplace approx for p(x|y,θ)           │
│  3. Numerical integration over θ                       │
│  4. Marginal posteriors via nested Laplace            │
└───────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────┐
│              Sparse Linear Algebra                     │
│  • Cholesky factorization of precision Q              │
│  • Solve Q x = b via back-substitution                │
│  • Log-determinant from Cholesky diagonal             │
└───────────────────────────────────────────────────────┘
```

### Mathematical Foundation

**Latent Gaussian Model:**
```
y | x, θ ~ ∏ p(y_i | η_i, θ)           [Likelihood]
x | θ ~ N(0, Q(θ)^{-1})                 [Latent field]
θ ~ p(θ)                                 [Hyperpriors]
```

For quotr bivariate:
```
x = (β_num, β_denom, b_shared, b_num, b_denom, φ_spatial)
θ = (σ_shared, σ_num, σ_denom, σ_spatial, φ_overdispersion, ...)
```

**INLA Approximation:**

1. **Laplace for hyperparameters:**
   ```
   p̃(θ | y) ∝ p(y, x, θ) / p̃_G(x | θ, y) |_{x=x*(θ)}
   ```
   where x*(θ) is the mode of p(x | y, θ)

2. **Nested Laplace for latent field:**
   ```
   p̃(x_i | y) = ∫ p̃_G(x_i | θ, y) p̃(θ | y) dθ
   ```

3. **Numerical integration over θ:**
   - Grid-based for low-dim θ (≤4)
   - CCD (Central Composite Design) for higher dim

### Key Components

#### 1. Precision Matrix Builder (`R/inla_precision.R`)
```r
# Build full precision matrix Q(θ) for latent field
build_precision <- function(formula_spec, spatial, theta) {
  # Block structure:
  # Q = | Q_beta    0         0      |
  #     | 0         Q_re      0      |
  #     | 0         0         Q_sp   |

  n_beta <- ncol(formula_spec$numerator$X) + ncol(formula_spec$denominator$X)
  n_re <- sum(sapply(formula_spec$shared$random_effects, function(r) r$n_groups))
  n_sp <- if (!is.null(spatial)) spatial$n_spatial else 0

  # Fixed effects: vague prior
  Q_beta <- Diagonal(n_beta, 1e-6)


  # Random effects: precision = 1/sigma^2
  Q_re <- build_re_precision(formula_spec, theta)

  # Spatial: CAR or BYM2 precision
  Q_sp <- if (!is.null(spatial)) {
    build_spatial_precision(spatial, theta)
  } else {
    Matrix(0, 0, 0)
  }

  bdiag(Q_beta, Q_re, Q_sp)
}
```

#### 2. Laplace Approximation (`src/inla_laplace.cpp`)
```cpp
// Find mode x*(θ) via Newton-Raphson
// Returns: mode, Hessian at mode, log-determinant
List laplace_mode(
  NumericVector y_num,
  NumericVector y_denom,
  NumericMatrix X_num,
  NumericMatrix X_denom,
  S4 Q,                    // Sparse precision matrix
  NumericVector theta,
  String family
);

// Laplace approximation to p(θ|y)
double log_posterior_theta(
  NumericVector theta,
  NumericVector y_num,
  NumericVector y_denom,
  // ... model specification
);
```

#### 3. Hyperparameter Integration (`R/inla_integrate.R`)
```r
# Grid-based integration for low-dim theta
inla_grid_integrate <- function(log_post_fn, theta_init, n_points = 20) {
  # Find mode
  opt <- optim(theta_init, function(t) -log_post_fn(t),
               method = "BFGS", hessian = TRUE)

  # Build grid around mode
  theta_sd <- sqrt(diag(solve(opt$hessian)))
  grid <- expand.grid(lapply(seq_along(theta_init), function(i) {
    seq(opt$par[i] - 3*theta_sd[i], opt$par[i] + 3*theta_sd[i],
        length.out = n_points)
  }))

  # Evaluate and normalize
  log_weights <- apply(grid, 1, log_post_fn)
  weights <- exp(log_weights - max(log_weights))
  weights <- weights / sum(weights)

  list(grid = grid, weights = weights)
}

# CCD integration for higher dim
inla_ccd_integrate <- function(log_post_fn, theta_init) {
  # Central Composite Design points
  # ...
}
```

#### 4. Marginal Posteriors (`R/inla_marginals.R`)
```r
# Compute marginal posterior for each element of latent field
inla_marginals <- function(mode, hessian, theta_grid, theta_weights) {
  n_x <- length(mode)
  marginals <- vector("list", n_x)

  for (i in seq_len(n_x)) {
    # For each theta configuration
    marginal_i <- numeric(0)
    for (j in seq_along(theta_weights)) {
      # Gaussian approximation at this theta
      mu_i <- mode[i]  # Would need to recompute for each theta
      sd_i <- sqrt(solve(hessian)[i, i])

      # Add to mixture
      marginal_i <- c(marginal_i,
                      theta_weights[j] * dnorm(x_grid, mu_i, sd_i))
    }
    marginals[[i]] <- marginal_i
  }

  marginals
}
```

### Bivariate Extension

For quotr's joint numerator-denominator model:

```
η_num = X_num β_num + Z b_shared + φ_spatial
η_denom = X_denom β_denom + Z b_shared + φ_spatial

y_num | η_num ~ f_num(...)
y_denom | η_denom ~ f_denom(...)
```

Key insight: The shared structure means the precision matrix has off-diagonal
blocks connecting numerator and denominator latent effects.

```
Q_joint = | Q_num    Q_cross |
          | Q_cross' Q_denom |
```

For shared random effects:
```
Q_cross[i,j] = -1/σ²_shared if obs i and j share group
```

### Likelihood Contributions

For different families:

**negbin_negbin:**
```cpp
double log_lik_negbin(int y, double eta, double phi) {
  double mu = exp(eta);
  return lgamma(y + phi) - lgamma(phi) - lgamma(y + 1) +
         phi * log(phi / (mu + phi)) + y * log(mu / (mu + phi));
}
```

**binomial:**
```cpp
double log_lik_binomial(int y, int n, double eta) {
  double p = 1.0 / (1.0 + exp(-eta));
  return y * log(p) + (n - y) * log(1 - p) + lchoose(n, y);
}
```

**poisson_gamma:**
```cpp
// Poisson for numerator
double log_lik_poisson(int y, double eta) {
  return y * eta - exp(eta) - lgamma(y + 1);
}

// Gamma for denominator
double log_lik_gamma(double y, double eta, double shape) {
  double rate = shape / exp(eta);
  return shape * log(rate) - lgamma(shape) + (shape - 1) * log(y) - rate * y;
}
```

### Files to Create

```
src/
├── inla_laplace.cpp       # Mode finding, Hessian computation
├── inla_laplace.h
├── inla_sparse.cpp        # Sparse Cholesky, log-det
├── inla_sparse.h
├── inla_likelihood.cpp    # Family-specific likelihoods
└── inla_likelihood.h

R/
├── backend_inla.R         # Main INLA dispatcher
├── inla_precision.R       # Precision matrix construction
├── inla_integrate.R       # Hyperparameter integration
├── inla_marginals.R       # Marginal posterior extraction
└── inla_utils.R           # Helper functions
```

### Approximation Quality Controls

```r
quotr(
  ...,
  backend = "inla",
  inla_control = list(
    strategy = "gaussian",     # "gaussian", "laplace", "simplified.laplace"
    int.strategy = "grid",     # "grid", "ccd", "eb" (empirical Bayes)
    n.grid = 20,               # Grid points per hyperparameter
    tolerance = 1e-6           # Optimization tolerance
  )
)
```

### Comparison with R-INLA

| Feature | quotr Laplace | R-INLA |
|---------|---------------|--------|
| Areal models (CAR/BYM2) | ✓ | ✓ |
| Matérn/SPDE | ✗ | ✓ |
| Bivariate joint | ✓ (native) | Requires workarounds |
| Installation | CRAN-ready | Separate repo |
| Speed (areal) | Comparable | Comparable |
| Approximation quality | Good | Excellent |

### Testing Plan

1. **Unit tests for components**
   - Precision matrix structure
   - Laplace mode finding
   - Log-determinant computation

2. **Comparison with Stan**
   - Same model, compare posteriors
   - Should match within ±0.1 posterior SD

3. **Comparison with R-INLA** (if available)
   - Standard CAR model
   - BYM2 model

4. **Scalability benchmarks**
   - N = 1k, 10k, 50k, 100k
   - Measure time and memory

### API

```r
quotr(
  y | n ~ x + (1 | region),
  data = large_df,
  family = quotr_binomial(),
  spatial = spatial_car(adj),
  backend = "laplace"        # Fast approximate inference
)
```

### Dependencies

```
LinkingTo: Rcpp, RcppEigen
Imports: Matrix
```

### Timeline Estimate
- Precision matrix builder: 2 days
- Laplace approximation (C++): 4 days
- Hyperparameter integration: 3 days
- Marginal posteriors: 2 days
- Bivariate joint structure: 3 days
- Testing & validation: 4 days
- Documentation: 2 days
- **Total: ~4 weeks**

---

## Version Summary

| Version | Backend | Best For | Speed |
|---------|---------|----------|-------|
| v1.0 | Stan | All families, exact inference | Baseline |
| v1.1 | Pólya-Gamma | Binomial, N < 50k | 5-10x faster |
| v1.2 | Laplace | Areal spatial, N > 10k | 10-100x faster |

## Future (v2.0+)

- SPDE/Matérn spatial (would require significant Laplace backend expansion)
- Temporal AR(1) / random walk
- Zero-inflation variants
- GPU acceleration for very large datasets
