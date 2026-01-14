# GP Heisenbug Debug Plan

## Problem Summary

GP spatial models (`spatial_gp()`) crash intermittently with longer chains:
- **Working**: iter ≤ 10 for any n (10-100)
- **Crashing**: iter ≥ 50 crashes intermittently (random-state dependent)

Affected configurations (5 rows in gradient_methods.md):
| Row | Config |
|----:|--------|
| 7 | poisson_gamma + RE + GP |
| 8 | poisson_gamma + RE + MSGP |
| 18 | poisson_gamma + RE + GP + RW1 |
| 27 | negbin_negbin + RE + GP |
| 38 | binomial + RE + GP |

## Symptoms

1. Segmentation fault during HMC sampling (not initialization)
2. Crash is random-state dependent (heisenbug)
3. Works with short chains, fails with longer chains
4. Not reproducible with specific seed - varies by run

## Hypotheses

### H1: Memory corruption in NNGP computation
The `gp_nngp_log_lik()` function in `src/hmc_gp.cpp` has complex memory access patterns:
- Nested loops over neighbors
- Index arithmetic with nn_idx, nn_order
- Potential out-of-bounds access on specific parameter combinations

**Test**: Add bounds checking and assertions to all array accesses

### H2: Numerical instability in Cholesky
The small matrix Cholesky decomposition for conditional covariance could fail:
- Near-singular matrices at certain parameter values
- Accumulated numerical error over many iterations

**Test**: Add condition number checks and early return for ill-conditioned matrices

### H3: Stack overflow from deep recursion
HMC trajectory computation with many leapfrog steps:
- Each gradient call allocates temporary vectors
- Possible stack exhaustion with longer chains

**Test**: Profile stack usage, reduce temporary allocations

### H4: Race condition (if OpenMP enabled)
Parallel gradient computation could have race conditions:
- Shared state between threads
- Non-atomic updates

**Test**: Run with OMP_NUM_THREADS=1 to disable parallelism

## Debug Steps

### Phase 1: Reproduce reliably
```bash
# Run with AddressSanitizer (requires rebuild)
PKG_CXXFLAGS="-fsanitize=address -fno-omit-frame-pointer" R CMD INSTALL .

# Run test
Rscript test_gp_final.R  # with iter=50
```

### Phase 2: Narrow down location
1. Add debug output before/after each major function call
2. Identify which function is executing when crash occurs
3. Add assertions for array bounds, NaN/Inf values

### Phase 3: Fix
Based on findings from Phase 2:
- Fix bounds checking
- Add numerical safeguards
- Reduce memory allocation in hot paths

## Files to Modify

| File | Purpose |
|------|---------|
| `src/hmc_gp.cpp` | NNGP log-likelihood |
| `src/hmc_gp.h` | GP data structures |
| `src/hmc_sampler.cpp` | Gradient computation |
| `src/hmc_gp_autodiff.h` | Templated GP (for autodiff path) |

## Test Script

```r
# test_gp_final.R - working baseline
library(numdenom)
set.seed(123)
for (n in c(10, 20, 30, 50, 100)) {
  data <- data.frame(
    y_num = rpois(n, exp(2)),
    y_denom = rgamma(n, 5, 1),
    x = rnorm(n),
    lon = runif(n, 0, 10),
    lat = runif(n, 0, 10)
  )
  fit <- ratiod(y_num | y_denom ~ x, data = data,
                family = ratiod_poisson_gamma(),
                spatial = spatial_gp(~ lon + lat, nn = 5),
                iter = 10, warmup = 5, chains = 1, refresh = 0)
}
```

## Success Criteria

1. GP models run without crash for iter=400, warmup=200
2. Results match ICAR/BYM2 on equivalent test cases
3. Speedup vs Stan > 1x (currently 0.6x with numerical gradients)

## Priority

**Low** - GP is a specialized use case. ICAR/BYM2 cover most spatial modeling needs and are fully stable with 13-22x speedup vs Stan.

## Timeline

Deferred until core functionality is complete. Estimated effort: 2-4 hours with debugger access.
