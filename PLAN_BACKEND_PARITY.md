# Full Backend Parity Plan: GP, Multiscale, Temporal, RSR

## Overview

Extend Laplace and Pólya-Gamma (PG) backends to support all advanced spatial/temporal features currently only available in HMC.

## Current State

| Feature | HMC | Laplace | PG |
|---------|-----|---------|-----|
| ICAR spatial | ✅ | ✅ | ✅ |
| BYM2 spatial | ✅ | ✅ | ✅ |
| GP spatial (NNGP) | ✅ | ❌ | ❌ |
| Multiscale GP (local+regional) | ✅ | ❌ | ❌ |
| Multiscale temporal (trend+seasonal+short) | ✅ | ❌ | ❌ |
| RSR (spatial confounding) | ✅ | ❌ | ❌ |

## Implementation Plan

### Phase 1: Laplace Backend Extensions

#### 1.1 GP Spatial for Laplace (`laplace_core.cpp`)

**Approach**: GP effects add a dense covariance contribution to the Hessian. For NNGP, this becomes sparse.

**Key changes**:
- New function: `laplace_mode_gp()`
- Add GP contribution to Hessian:
  - For NNGP: Each observation's conditional variance depends on neighbors
  - Hessian diagonal: `1/cond_var_i` + likelihood contribution
  - Hessian off-diagonal: cross terms from conditional structure
- Add GP prior log-density to marginal likelihood

**Parameters**:
- `coords` (n_obs × 2): observation coordinates
- `nn_idx` (n_obs × nn): neighbor indices
- `nn_dist` (n_obs × nn): distances to neighbors
- `sigma2_gp`: GP variance
- `phi_gp`: GP range

**Complexity**: Moderate - need to handle sparse NNGP structure

#### 1.2 Multiscale GP Spatial for Laplace

**Approach**: Treat local + regional as two independent GP components

**Key changes**:
- New function: `laplace_mode_multiscale_gp()`
- Parameters: 2× n_obs spatial effects (local + regional)
- Hessian: block structure with local and regional components
- Range constraints enforced via bounded optimization

**Parameters**:
- Local: `w_local`, `sigma2_local`, `phi_local`
- Regional: `w_regional`, `sigma2_regional`, `phi_regional`
- Range bounds for identifiability

#### 1.3 Multiscale Temporal for Laplace

**Approach**: Temporal components have sparse precision matrices (tridiagonal for RW1, pentadiagonal for RW2)

**Key changes**:
- New function: `laplace_mode_multiscale_temporal()`
- Components:
  - Trend (RW1 or RW2): sparse precision matrix
  - Seasonal (cyclic RW1): sparse with wrap-around
  - Short-term (AR1 or IID): tridiagonal or diagonal
- Hessian: add temporal precision contributions to diagonal blocks

**Parameters**:
- `trend` (n_times): trend effects
- `seasonal` (period): seasonal effects
- `short_term` (n_times): short-term effects
- `sigma2_trend`, `sigma2_seasonal`, `sigma2_short`, `rho_short`

#### 1.4 RSR for Laplace

**Approach**: RSR projects spatial effects to be orthogonal to covariate space

**Key changes**:
- Pre-compute projection matrix P = I - X(X'X)^{-1}X'
- Replace spatial effects with P*phi
- Adjust Hessian accordingly

**Parameters**:
- `P` (n_obs × n_obs or sparse representation): projection matrix
- Works with any spatial type (ICAR, BYM2, GP)

---

### Phase 2: PG Backend Extensions

The PG backend uses Gibbs sampling, so we need full conditional distributions.

#### 2.1 GP Spatial for PG (`pg_spatial.cpp`)

**Approach**: For GP, the full conditional of w_i is:
```
w_i | w_{-i}, ... ~ N(cond_mean, cond_var)
```
where conditional moments come from NNGP structure.

**Key changes**:
- New function: `update_spatial_gp()`
- For each location i in NNGP order:
  - Compute conditional mean from neighbors
  - Combine with likelihood (PG pseudo-likelihood)
  - Sample from Gaussian full conditional
- New function: `update_gp_hyperparams()` for sigma2, phi

**Gibbs update**:
```
For i = 1:N (in NNGP order):
  cond_mean_gp = sum_j (alpha_j * w_neighbors[j])
  cond_var_gp = sigma2 - c' C^{-1} c

  # PG likelihood contribution
  sum_omega_i = sum of omega for obs at location i
  sum_kappa_i = sum of (y - n/2) for obs at location i

  # Combined precision and mean
  prec = 1/cond_var_gp + sum_omega_i
  mean = (cond_mean_gp/cond_var_gp + sum_kappa_i - offset_sum) / prec

  w[i] ~ N(mean, 1/prec)
```

#### 2.2 Multiscale GP for PG

**Approach**: Interleaved Gibbs updates for local and regional components

**Key changes**:
- Update local effects holding regional fixed
- Update regional effects holding local fixed
- Enforce range constraints during hyperparameter updates

#### 2.3 Multiscale Temporal for PG

**Approach**: Temporal effects have closed-form Gibbs updates due to sparse precision

**Key changes**:
- New function: `update_temporal_multiscale()`
- Trend: block update using sparse Cholesky
- Seasonal: cyclic RW1 update
- Short-term: AR1 or IID update

**For RW1 trend** (sequential update):
```
For t = 1:T:
  if t == 1:
    prec = tau + sum_omega_t
    mean = (tau * phi[2] + sum_kappa_t) / prec
  else if t == T:
    prec = tau + sum_omega_t
    mean = (tau * phi[T-1] + sum_kappa_t) / prec
  else:
    prec = 2*tau + sum_omega_t
    mean = (tau * (phi[t-1] + phi[t+1]) + sum_kappa_t) / prec
  phi[t] ~ N(mean, 1/prec)
```

#### 2.4 RSR for PG

**Approach**: Project spatial updates through RSR projection

**Key changes**:
- After updating raw spatial effects, project: `phi_rsr = P * phi`
- Use projected effects in linear predictor
- Projection is applied at each iteration

---

### Phase 3: R Interface Updates

#### 3.1 Backend Selection (`R/ratiod.R`)

Update `select_backend()` to handle new spatial types:
```r
select_backend <- function(formula, family, spatial, temporal, ...) {
  # Current logic handles ICAR/BYM2
  # Add handling for GP, multiscale_gp, multiscale_temporal, RSR
}
```

#### 3.2 Laplace Dispatcher (`R/backend_laplace.R`)

Add dispatch functions:
- `fit_laplace_gp()`
- `fit_laplace_multiscale_gp()`
- `fit_laplace_multiscale_temporal()`
- `fit_laplace_rsr()`

Each prepares data and calls appropriate C++ function.

#### 3.3 PG Dispatcher (`R/backend_pg.R`)

Add dispatch functions:
- `fit_pg_gp()`
- `fit_pg_multiscale_gp()`
- `fit_pg_multiscale_temporal()`
- `fit_pg_rsr()`

---

### Phase 4: Testing

#### Unit Tests

1. **GP Laplace**: Compare point estimates with HMC posterior means
2. **GP PG**: Compare posterior distributions with HMC
3. **Multiscale**: Verify components are identifiable with range constraints
4. **Temporal**: Test trend/seasonal/short-term separation
5. **RSR**: Verify orthogonality of spatial effects to covariates

#### Integration Tests

1. All three backends produce similar results on same data
2. Performance benchmarks (PG should be fastest for binomial)
3. Edge cases: few time points, small spatial field

---

## Implementation Order

1. **Laplace GP** - Foundation for understanding GP integration
2. **PG GP** - Build on Laplace experience
3. **Laplace Multiscale GP** - Extension of GP
4. **PG Multiscale GP** - Extension of PG GP
5. **Laplace Temporal** - Independent from spatial
6. **PG Temporal** - Independent from spatial
7. **Laplace RSR** - Modifier for any spatial type
8. **PG RSR** - Modifier for any spatial type

---

## File Changes Summary

### C++ Files

| File | Changes |
|------|---------|
| `src/laplace_core.cpp` | +800 lines: GP, multiscale GP, temporal, RSR functions |
| `src/laplace_core.h` | +50 lines: new function declarations |
| `src/pg_spatial.cpp` | +600 lines: GP update, multiscale GP update |
| `src/pg_spatial.h` | +40 lines: new function declarations |
| `src/pg_temporal.cpp` | NEW: +400 lines: temporal Gibbs updates |
| `src/pg_temporal.h` | NEW: +30 lines: temporal declarations |

### R Files

| File | Changes |
|------|---------|
| `R/backend_laplace.R` | +400 lines: GP/multiscale/temporal/RSR dispatch |
| `R/backend_pg.R` | +400 lines: GP/multiscale/temporal/RSR dispatch |
| `R/ratiod.R` | +50 lines: backend selection updates |

### Test Files

| File | Changes |
|------|---------|
| `tests/testthat/test-laplace-gp.R` | NEW: GP Laplace tests |
| `tests/testthat/test-pg-gp.R` | NEW: GP PG tests |
| `tests/testthat/test-backend-parity.R` | NEW: Cross-backend comparison |

---

## Estimated Complexity

| Component | Lines of Code | Complexity |
|-----------|---------------|------------|
| Laplace GP | ~250 C++ | Medium |
| Laplace Multiscale GP | ~150 C++ | Medium |
| Laplace Temporal | ~200 C++ | Medium |
| Laplace RSR | ~100 C++ | Low |
| PG GP | ~300 C++ | High (Gibbs derivation) |
| PG Multiscale GP | ~200 C++ | High |
| PG Temporal | ~250 C++ | Medium |
| PG RSR | ~100 C++ | Low |
| R interfaces | ~400 R | Low |
| Tests | ~400 R | Low |

**Total**: ~2,350 lines of new code

---

## Notes

1. **NNGP for Laplace**: The key insight is that NNGP makes the GP precision matrix sparse, which maps directly to sparse Hessian contributions.

2. **PG + GP**: The challenge is that GP effects are correlated across space. Sequential Gibbs updates work but may mix slowly. Consider block updates for efficiency.

3. **RSR projection**: For large n, storing P explicitly is expensive. Use the identity P*phi = phi - X*(X'X)^{-1}*X'*phi which only requires matrix-vector products.

4. **Hyperparameter updates in PG**: For GP range (phi), use Metropolis-Hastings within Gibbs since there's no closed-form full conditional.
