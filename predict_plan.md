# predict.ratiod_fit() Completion Plan

## Current State

`predict.ratiod_fit()` in `R/ratio.R:736` supports:
- ✅ Fixed effects at new covariate values
- ✅ Random intercepts (existing levels)
- ✅ New RE levels with `allow_new_levels = TRUE` (uses population mean)
- ✅ HMC backend only
- ❌ Spatial effects at new coordinates
- ❌ Temporal effects at new time points
- ❌ Other backends (Laplace, PG, VI, ESS)
- ❌ Latent factors
- ❌ Random slopes

## Priority 1: Spatial Prediction (Critical)

This is what makes ecological models useful — predicting at unsampled locations.

### 1.1 GP/NNGP Spatial Prediction

For models with `spatial_gp()` or `spatial_hsgp()`:

**API**:
```r
predict(fit, newdata = grid_df, coords.0 = coords_matrix)
# or
predict(fit, newdata = grid_df)  # if coords columns are in newdata
```

**Algorithm** (Kriging):

Given fitted GP with posterior samples of:
- `w` (spatial effects at training locations)
- `sigma2_gp` (marginal variance)
- `phi` (range parameter)

For new location `s0`:
```
C(s0, S) = covariance between s0 and training locations S
C(S, S) = covariance matrix at training locations (already computed)

w(s0) | w(S), θ ~ N(C(s0,S) C(S,S)^{-1} w(S), σ² - C(s0,S) C(S,S)^{-1} C(S,s0))
```

For NNGP, use only the m nearest neighbors of s0.

**C++ function needed**:
```cpp
// src/predict_spatial.cpp
Rcpp::NumericMatrix predict_gp(
  Rcpp::NumericMatrix coords_train,    // [n_train, 2]
  Rcpp::NumericMatrix coords_new,      // [n_new, 2]
  Rcpp::NumericMatrix w_samples,       // [n_samples, n_train]
  Rcpp::NumericVector sigma2_samples,  // [n_samples]
  Rcpp::NumericVector phi_samples,     // [n_samples]
  int nn,                              // neighbors for NNGP
  int cov_type                         // 0=exp, 1=matern, 2=gaussian
);
```

**Return**: `[n_samples, n_new]` matrix of spatial effect draws at new locations

### 1.2 ICAR/BYM2 Spatial Prediction

For areal data with `spatial_car()` or `spatial_bym2()`:

New locations must be assigned to existing areas, OR we interpolate.

**Option A**: Require `spatial_group` in newdata
```r
newdata$region <- ...  # must match training regions
predict(fit, newdata)
```
Simple: just look up the spatial effect for that region.

**Option B**: Interpolate for point predictions within areas
```r
predict(fit, newdata, coords.0 = coords_matrix, interpolate = TRUE)
```
Use IDW or similar to interpolate between areal effects.

**Recommendation**: Start with Option A (lookup), add Option B later.

### 1.3 HSGP Spatial Prediction

HSGP uses spectral approximation:
```
w(s) = Σ_j φ_j(s) * β_j
```

Prediction is trivial — just evaluate basis functions at new locations:
```r
# Compute basis at new coords
phi_new <- compute_hsgp_basis(coords_new, L, m)
w_new <- phi_new %*% t(beta_samples)  # [n_new, n_samples]
```

### 1.4 SVC Prediction

Spatially varying coefficients need coefficient surfaces predicted:
```r
# Return both intercept and slope surfaces
predict(fit, newdata, coords.0 = ..., include_svc = TRUE)
```

---

## Priority 2: Backend Support

### 2.1 Laplace Backend

Laplace stores posterior mode + Hessian. For prediction:
1. Use mode for point estimate
2. Sample from MVN(mode, H^{-1}) for uncertainty
3. Transform through prediction equation

```r
compute_predictions_laplace <- function(object, pred_data, type, n_samples = 1000) {
  mode <- object$.internal$mode
  H_inv <- object$.internal$hessian_inv

  # Sample from approximate posterior
  samples <- mvtnorm::rmvnorm(n_samples, mode, H_inv)

  # Then same logic as HMC
  ...
}
```

### 2.2 PG Backend

PG stores samples directly (like HMC). Should be straightforward:
```r
compute_predictions_pg <- function(object, pred_data, type) {
  # Extract beta, re from object$.internal
  # Same prediction logic as HMC
}
```

### 2.3 VI Backend

VI stores variational parameters. Sample from q(θ) then predict:
```r
compute_predictions_vi <- function(object, pred_data, type, n_samples = 1000) {
  # Sample from variational posterior
  samples <- sample_vi_posterior(object, n_samples)
  # Then same as HMC
}
```

---

## Priority 3: Temporal Prediction

### 3.1 GP Temporal

Same as spatial GP, but 1D:
```r
predict(fit, newdata, times.0 = new_times)
```

### 3.2 RW1/RW2 Temporal

For random walk models, extrapolation requires drawing from the RW prior:
- **Interpolation** (time within observed range): condition on neighbors
- **Extrapolation** (time beyond range): draw from RW increments

```r
# For time t+1 beyond max observed time T:
gamma[t+1] | gamma[T], sigma2 ~ N(gamma[T], sigma2)  # RW1
gamma[t+1] | gamma[T], gamma[T-1], sigma2 ~ N(2*gamma[T] - gamma[T-1], sigma2)  # RW2
```

### 3.3 AR1 Temporal

```r
gamma[t+k] = rho^k * gamma[T] + noise
```

---

## Priority 4: Other Features

### 4.1 Latent Factors

Latent factors are observation-level, so for new observations:
- Sample new factor values from N(0, 1) prior
- Or provide factor scores if available

### 4.2 Random Slopes

Need to handle:
```r
# Model: y ~ x + (x | group)
# Prediction for existing group: use fitted slope
# Prediction for new group: sample from slope distribution
```

---

## Implementation Order

```
Week 1: Spatial Prediction
├── 1.1 GP prediction (kriging)
├── 1.2 HSGP prediction (basis evaluation)
├── 1.3 NNGP prediction (neighbor-based)
└── 1.4 ICAR lookup

Week 2: Backend Support
├── 2.1 PG backend (easy, has samples)
├── 2.2 Laplace backend (sample from MVN)
└── 2.3 VI backend (sample from q)

Week 3: Temporal + Polish
├── 3.1 GP temporal
├── 3.2 RW extrapolation
├── 3.3 AR1 prediction
└── 3.4 Tests + documentation
```

---

## API Design

### Current
```r
predict(fit, newdata, type = "response", component = "ratio",
        summary = TRUE, probs = c(0.025, 0.5, 0.975),
        re_formula = NULL, allow_new_levels = FALSE)
```

### Extended
```r
predict(fit, newdata,
        type = c("response", "link"),
        component = c("ratio", "numerator", "denominator", "all"),
        summary = TRUE,
        probs = c(0.025, 0.5, 0.975),
        re_formula = NULL,
        allow_new_levels = FALSE,
        # NEW parameters:
        coords.0 = NULL,           # Spatial coords for new locations
        times.0 = NULL,            # Time indices for new times
        include_spatial = TRUE,    # Include spatial effects in prediction
        include_temporal = TRUE,   # Include temporal effects
        n_samples = NULL,          # For Laplace/VI: number of posterior samples
        return_spatial = FALSE,    # Return w.0.samples (like spOccupancy)
        return_temporal = FALSE)   # Return temporal effect samples
```

### Return Value (Extended)

When `return_spatial = TRUE` or `return_temporal = TRUE`:
```r
list(
  predictions = data.frame(...),  # Standard prediction summaries
  w.0.samples = matrix(...),      # [n_samples, n_new] spatial effects
  gamma.0.samples = matrix(...)   # [n_samples, n_new] temporal effects
)
```

---

## File Changes

```
R/ratio.R
├── predict.ratiod_fit()          # Extend with new parameters
├── build_prediction_data()       # Add coords.0, times.0 handling
├── compute_predictions_hmc()     # Add spatial/temporal
├── compute_predictions_pg()      # NEW
├── compute_predictions_laplace() # NEW
├── compute_predictions_vi()      # NEW
└── predict_spatial_gp()          # NEW helper

src/
├── predict_spatial.cpp           # NEW: GP/NNGP prediction
├── predict_spatial.h             # NEW
└── RcppExports.cpp               # Updated
```

---

## Test Cases

```r
# 1. GP spatial prediction
fit_gp <- ratiod(..., spatial = spatial_gp(...))
new_coords <- matrix(runif(20), ncol = 2)
pred <- predict(fit_gp, newdata = data.frame(x = 1:10),
                coords.0 = new_coords)

# 2. ICAR lookup
fit_icar <- ratiod(..., spatial = spatial_car(...))
pred <- predict(fit_icar, newdata = data.frame(region = c("A", "B"), x = 1:2))

# 3. Laplace backend
fit_lap <- ratiod(..., mode = "laplace")
pred <- predict(fit_lap, newdata = ..., n_samples = 2000)

# 4. PG backend
fit_pg <- ratiod(..., mode = "pg")
pred <- predict(fit_pg, newdata = ...)

# 5. Return spatial effects
pred <- predict(fit_gp, newdata = ..., coords.0 = ..., return_spatial = TRUE)
str(pred$w.0.samples)  # [n_samples, n_new]
```

---

## Validation

Compare with spOccupancy for binomial spatial models:
```r
# Fit equivalent models
fit_numdenom <- ratiod(y | n ~ x, spatial = spatial_gp(...), family = ratiod_binomial())
fit_spocc <- spPGOcc(...)

# Predict at same new locations
pred_nd <- predict(fit_numdenom, newdata, coords.0 = new_coords)
pred_sp <- predict(fit_spocc, X.0, coords.0 = new_coords)

# Compare: should be similar (different samplers, same model)
cor(pred_nd$mean, colMeans(pred_sp$psi.0.samples))  # Should be > 0.95
```
