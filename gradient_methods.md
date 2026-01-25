# Gradient Methods by Model Configuration

## Gradient Methods

| Method | Description | Complexity | Relative Speed |
|:------:|-------------|:----------:|:--------------:|
| **N** | Numerical finite differences | O(n×p) | 1.0x (baseline) |
| **A_t** | Autodiff tape-based (current) | O(n×ops) | ~0.6-1.5x (slower than N!) |
| **A** | Autodiff expression template (planned) | O(n) | ~8x (expected) |
| **H** | Hand-coded analytical | O(n) | ~9x faster |

### Current Implementation Status

```
N    - Numerical (reference)     IMPLEMENTED
A_t  - Tape-based autodiff       IMPLEMENTED (slow, to be superseded)
A    - Expression template       PLANNED (see autodiff_plan.md)
H    - Hand-coded                IMPLEMENTED (production default)
```

### Benchmark Results (n=500, 500 iter)

**Core families (no RE):**
```
                      N(s)    A_t(s)   A(s)    H(s)    H speedup
poisson_gamma         40.3    52.3    32.8     8.9     4.5x vs N
negbin_negbin         54.6    72.0    51.2    12.1     4.5x vs N
binomial              14.7    37.4    20.8     9.4     1.6x vs N
```

**With random effects (50 groups):**
```
                      N(s)    A_t(s)   A(s)    H(s)    H speedup
poisson_gamma+RE     425.1    53.6   321.7     8.7     49x vs N
negbin_negbin+RE     720.8    90.4   565.9    12.1     60x vs N
binomial+RE          155.1    34.7   251.9     9.4     16x vs N
```

**With ICAR spatial (50 units):**
```
                      N(s)    A_t(s)   A(s)     H(s)    H speedup
poisson_gamma+ICAR   531.4    55.9   628.1      8.9     60x vs N
negbin_negbin+ICAR  1376.3   108.1  1295.7     12.0    115x vs N
binomial+ICAR        272.4    40.3   421.0      9.8     28x vs N
```

**With RW1 temporal (20 time points):**
```
                      N(s)    A_t(s)   A(s)    H(s)    H speedup
poisson_gamma+RW1    368.3    41.5   324.3     8.8     42x vs N
```

**Key insight**:
- H (hand-coded) gives consistent ~9-12s regardless of model complexity
- A_t (tape) is fastest for complex models when H unavailable
- A (expression) has high overhead for many parameters
- H speedup: 4.5x for simple models, up to **115x** for complex spatial models

### Why Four Methods?

| Mode | Use Case |
|:----:|----------|
| N | Fallback for debugging, gradient verification |
| A_t | Legacy tape-based, reference for gradient verification |
| A | Production fallback (when H unavailable), rapid prototyping |
| H | Production default (fastest) |

All modes available via `gradient_mode` parameter: `"auto"` (default), `"N"`, `"A"`, `"H"`.
Currently `"A"` maps to A_t until expression templates are implemented.

---

## Benchmarking Requirements

### Publication-Ready Validation

**REQUIRED** for each model configuration before publication:

1. **Gradient verification**: `max(|grad_H - grad_A_t|) < 1e-5` and `max(|grad_A_t - grad_N|) < 1e-4`
2. **Stan comparison**: Posterior means within 2 SE of Stan reference (same data, same priors)
3. **Timing benchmark**: Record N/A_t/H times for standardized test (n=500, 500 iter, chains=1)

### Benchmark Protocol

```r
# Standard benchmark parameters
N_OBS <- 500          # Observations
N_ITER <- 500         # Iterations (incl. warmup)
N_WARMUP <- 250       # Warmup
N_CHAINS <- 1         # Single chain for timing
N_SITES <- 50         # For spatial models
N_TIMES <- 20         # For temporal models

# Run benchmark for each gradient mode
for (mode in c("N", "A", "H")) {
  time <- system.time({
    fit <- ratiod(..., gradient_mode = mode, iter = N_ITER, chains = N_CHAINS)
  })["elapsed"]
}

# Compare to Stan (brms or cmdstanr)
stan_fit <- brms::brm(...)  # Equivalent model
```

### Columns Legend

| Column | Meaning |
|--------|---------|
| **Grad** | Current production gradient mode: N, A_t, or H |
| **N(s)** | Timing with numerical gradients (seconds) |
| **A_t(s)** | Timing with tape-based autodiff gradients (seconds) |
| **H(s)** | Timing with hand-coded gradients (seconds) |
| **Stan(s)** | Timing with Stan/brms (seconds) |
| **H/Stan** | Speedup ratio: Stan time / H time |

---

## All Model Configurations

### Section 1: poisson_gamma Family (Rows 1-30)

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 1 | poisson_gamma | ✗ | ✗ | ✗ | ✗ | H | 9.1 | 12.8 | 12.5 | 12.9 | |
| 2 | poisson_gamma | ✓ | ✗ | ✗ | ✗ | H | 12.2 | 12.1 | 12.1 | 12.0 | |
| 3 | poisson_gamma | slopes | ✗ | ✗ | ✗ | H | 9.1 | | | | |
| 4 | poisson_gamma | crossed | ✗ | ✗ | ✗ | H | 11.9 | 12.0 | 11.9 | 12.1 | |
| 5 | poisson_gamma | ✓ | ICAR | ✗ | ✗ | H | 12.0 | 11.8 | 12.1 | 12.0 | |
| 6 | poisson_gamma | ✓ | BYM2 | ✗ | ✗ | H | 12.0 | 12.0 | 11.7 | 11.7 | |
| 7 | poisson_gamma | ✓ | GP | ✗ | ✗ | H | 380.1 | | | | O(N³) |
| 8 | poisson_gamma | ✓ | HSGP | ✗ | ✗ | H | 13.2 | | | | m=8 (29x vs GP) |
| 9 | poisson_gamma | ✓ | MSGP | ✗ | ✗ | H | 605.3 | | | | O(N³) MSGP |
| 10 | poisson_gamma | ✓ | pCAR | ✗ | ✗ | H | 8.7 | | | | |
| 11 | poisson_gamma | ✓ | ✗ | RW1 | ✗ | H | 8.6 | | | | |
| 12 | poisson_gamma | ✓ | ✗ | RW2 | ✗ | H | 8.5 | | | | |
| 13 | poisson_gamma | ✓ | ✗ | AR1 | ✗ | H | 8.4 | | | | |
| 14 | poisson_gamma | ✓ | ✗ | GP_t | ✗ | H | 9.9 | | | | |
| 15 | poisson_gamma | ✓ | ✗ | MS_t | ✗ | H | 9.8 | | | | |
| 16 | poisson_gamma | ✓ | ✗ | ✗ | ZI | H | 8.6 | | | | |
| 17 | poisson_gamma | ✓ | ✗ | ✗ | Hurdle | H | 11.9 | | | | |
| 18 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H | 8.8 | | | | |
| 19 | poisson_gamma | ✓ | BYM2 | RW1 | ✗ | H | 9.1 | | | | |
| 20 | poisson_gamma | ✓ | ICAR | AR1 | ✗ | H | 9.0 | | | | |
| 21 | poisson_gamma | ✓ | GP | RW1 | ✗ | H | 135.3 | | | | GP O(N³) |
| 22 | poisson_gamma | ✓ | HSGP | RW1 | ✗ | H | 4.6 | | | | |
| 23 | poisson_gamma | ✓ | MSGP | RW1 | ✗ | H | 571.7 | | | | MSGP O(N³) |
| 24 | poisson_gamma | ✓ | ICAR | ✗ | ZI | H | 9.3 | | | | |
| 25 | poisson_gamma | slopes | ICAR | ✗ | ✗ | H | 9.6 | | | | |
| 26 | poisson_gamma | ✓ | SVC | ✗ | ✗ | H | | | | | SVC with NNGP, needs benchmark |
| 27 | poisson_gamma | ✓ | ✗ | TVC | ✗ | H | | | | | TVC with RW1/RW2/AR1, needs benchmark |
| 28 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H | 172.8 | | | | ST-I (50×20) |
| 29 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H | 180.6 | | | | ST-IV (50×20) |
| 30 | poisson_gamma | ✓ | ✗ | ✗ | ✗ | H | 94.4 | | | | latent (N=50) |

### Section 2: negbin_negbin Family (Rows 31-60)

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 31 | negbin_negbin | ✗ | ✗ | ✗ | ✗ | H | 17.1 | 17.0 | 16.8 | 16.8 | |
| 32 | negbin_negbin | ✓ | ✗ | ✗ | ✗ | H | 16.9 | 17.2 | 17.1 | 17.0 | |
| 33 | negbin_negbin | slopes | ✗ | ✗ | ✗ | H | 13.1 | | | | |
| 34 | negbin_negbin | crossed | ✗ | ✗ | ✗ | H | 17.4 | 17.0 | 17.1 | 17.2 | |
| 35 | negbin_negbin | ✓ | ICAR | ✗ | ✗ | H | 16.9 | 17.1 | 5.5 | 17.2 | A_t anomaly |
| 36 | negbin_negbin | ✓ | BYM2 | ✗ | ✗ | H | 17.1 | 17.2 | 17.1 | 17.1 | |
| 37 | negbin_negbin | ✓ | GP | ✗ | ✗ | H | 92.2 | | | | O(N³) |
| 38 | negbin_negbin | ✓ | HSGP | ✗ | ✗ | H | 6.6 | 6.6 | 6.6 | | FIXED |
| 39 | negbin_negbin | ✓ | MSGP | ✗ | ✗ | H | 193.0 | | | | FIXED |
| 40 | negbin_negbin | ✓ | pCAR | ✗ | ✗ | H | 9.5 | | | | |
| 41 | negbin_negbin | ✓ | ✗ | RW1 | ✗ | H | 12.4 | | | | |
| 42 | negbin_negbin | ✓ | ✗ | RW2 | ✗ | H | 12.5 | | | | |
| 43 | negbin_negbin | ✓ | ✗ | AR1 | ✗ | H | 12.3 | | | | |
| 44 | negbin_negbin | ✓ | ✗ | GP_t | ✗ | H | 8.1 | | | | |
| 45 | negbin_negbin | ✓ | ✗ | MS_t | ✗ | H | 8.2 | | | | |
| 46 | negbin_negbin | ✓ | ✗ | ✗ | ZI | H | 12.1 | | | | |
| 47 | negbin_negbin | ✓ | ✗ | ✗ | Hurdle | H | 13.0 | | | | |
| 48 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H | 12.7 | | | | |
| 49 | negbin_negbin | ✓ | BYM2 | RW1 | ✗ | H | 10.6 | | | | |
| 50 | negbin_negbin | ✓ | ICAR | AR1 | ✗ | H | 12.6 | | | | |
| 51 | negbin_negbin | ✓ | GP | RW1 | ✗ | H | 101.2 | | | | O(N³) |
| 52 | negbin_negbin | ✓ | HSGP | RW1 | ✗ | H | 6.8 | | | | FIXED |
| 53 | negbin_negbin | ✓ | MSGP | RW1 | ✗ | H | 190.3 | | | | FIXED |
| 54 | negbin_negbin | ✓ | ICAR | ✗ | ZI | H | 12.4 | | | | |
| 55 | negbin_negbin | slopes | ICAR | ✗ | ✗ | H | 13.2 | | | | |
| 56 | negbin_negbin | ✓ | SVC | ✗ | ✗ | H | | | | | SVC with NNGP, needs benchmark |
| 57 | negbin_negbin | ✓ | ✗ | TVC | ✗ | H | | | | | TVC with RW1/RW2/AR1, needs benchmark |
| 58 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H | 392.6 | | | | ST-I (50×20) |
| 59 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H | 391.0 | | | | ST-IV (50×20) |
| 60 | negbin_negbin | ✓ | ✗ | ✗ | ✗ | H | 217.7 | | | | latent (N=50) |

### Section 3: binomial Family (Rows 61-100)

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 61 | binomial | ✗ | ✗ | ✗ | ✗ | H | 12.7 | 13.1 | 13.1 | 13.3 | |
| 62 | binomial | ✓ | ✗ | ✗ | ✗ | H | 13.3 | 13.3 | 13.2 | 13.3 | |
| 63 | binomial | slopes | ✗ | ✗ | ✗ | H | 9.9 | | | | |
| 64 | binomial | crossed | ✗ | ✗ | ✗ | H | 13.4 | 13.1 | 13.2 | 13.8 | |
| 65 | binomial | ✓ | ICAR | ✗ | ✗ | H | 13.3 | 13.0 | 13.2 | 12.8 | |
| 66 | binomial | ✓ | BYM2 | ✗ | ✗ | H | 13.0 | 13.1 | 13.2 | 13.1 | |
| 67 | binomial | ✓ | GP | ✗ | ✗ | H | 105.6 | | | | O(N³) |
| 68 | binomial | ✓ | HSGP | ✗ | ✗ | H | 3.1 | | | | |
| 69 | binomial | ✓ | MSGP | ✗ | ✗ | H | 645.2 | | | | O(N³) MSGP |
| 70 | binomial | ✓ | pCAR | ✗ | ✗ | H | 9.9 | | | | |
| 71 | binomial | ✓ | ✗ | RW1 | ✗ | H | 9.4 | | | | |
| 72 | binomial | ✓ | ✗ | RW2 | ✗ | H | 9.3 | | | | |
| 73 | binomial | ✓ | ✗ | AR1 | ✗ | H | 9.4 | | | | |
| 74 | binomial | ✓ | ✗ | GP_t | ✗ | H | 6.0 | | | | |
| 75 | binomial | ✓ | ✗ | MS_t | ✗ | H | 6.5 | | | | |
| 76 | binomial | ✓ | ✗ | ✗ | ZI | H | 14.7 | 14.8 | 14.7 | 14.4 | |
| 77 | binomial | ✓ | ✗ | ✗ | Hurdle | H | 13.1 | 13.1 | 13.5 | 13.3 | |
| 78 | binomial | ✓ | ✗ | ✗ | OI | H | 9.2 | | | | |
| 79 | binomial | ✓ | ✗ | ✗ | ZOIB | H | 9.3 | | | | |
| 80 | binomial | ✓ | ICAR | RW1 | ✗ | H | 9.7 | | | | |
| 81 | binomial | ✓ | BYM2 | RW1 | ✗ | H | 9.8 | | | | |
| 82 | binomial | ✓ | ICAR | AR1 | ✗ | H | 9.8 | | | | |
| 83 | binomial | ✓ | GP | RW1 | ✗ | H | 113.4 | | | | O(N³) |
| 84 | binomial | ✓ | HSGP | RW1 | ✗ | H | 3.6 | | | | |
| 85 | binomial | ✓ | MSGP | RW1 | ✗ | H | 656.2 | | | | O(N³) MSGP |
| 86 | binomial | ✓ | ICAR | ✗ | ZI | H | 14.2 | 14.4 | 14.4 | 14.5 | |
| 87 | binomial | slopes | ICAR | ✗ | ✗ | H | 10.4 | | | | |
| 88 | binomial | ✓ | SVC | ✗ | ✗ | H | | | | | SVC with NNGP, needs benchmark |
| 89 | binomial | ✓ | ✗ | TVC | ✗ | H | | | | | TVC with RW1/RW2/AR1, needs benchmark |
| 90 | binomial | ✓ | ICAR | RW1 | ✗ | H | 119.9 | | | | ST-I (50×20) |
| 91 | binomial | ✓ | ICAR | RW1 | ✗ | H | 121.4 | | | | ST-IV (50×20) |
| 92 | binomial | ✓ | ✗ | ✗ | ✗ | H | 57.2 | | | | latent (N=50) |

### Section 4: gamma_gamma Family (Rows 93-97)

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 93 | gamma_gamma | ✗ | ✗ | ✗ | ✗ | H | 31.0 | | | | |
| 94 | gamma_gamma | ✓ | ✗ | ✗ | ✗ | H | 298.6 | | | | SLOW |
| 95 | gamma_gamma | ✓ | ICAR | ✗ | ✗ | H | 570.1 | | | | SLOW |
| 96 | gamma_gamma | ✓ | ✗ | RW1 | ✗ | H | 373.8 | | | | SLOW |
| 97 | gamma_gamma | ✓ | ICAR | RW1 | ✗ | H | 409.0 | | | | SLOW |

### Section 5: lognormal Family (Rows 98-102)

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 98 | lognormal | ✗ | ✗ | ✗ | ✗ | H | 9.7 | | | | |
| 99 | lognormal | ✓ | ✗ | ✗ | ✗ | H | 81.1 | | | | |
| 100 | lognormal | ✓ | ICAR | ✗ | ✗ | H | 167.4 | | | | |
| 101 | lognormal | ✓ | ✗ | RW1 | ✗ | H | 122.8 | | | | |
| 102 | lognormal | ✓ | ICAR | RW1 | ✗ | H | 212.6 | | | | |

### Section 6: beta_binomial Family (Rows 103-107)

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 103 | beta_binomial | ✗ | ✗ | ✗ | ✗ | H | 48.0 | | | | |
| 104 | beta_binomial | ✓ | ✗ | ✗ | ✗ | H | 453.7 | | | | SLOW |
| 105 | beta_binomial | ✓ | ICAR | ✗ | ✗ | H | 698.9 | | | | SLOW |
| 106 | beta_binomial | ✓ | ✗ | RW1 | ✗ | H | 663.6 | | | | SLOW |
| 107 | beta_binomial | ✓ | ICAR | RW1 | ✗ | H | 1047.8 | | | | SLOW |

---

## Spatial Types Reference

| Code | Full Name | Implementation |
|:----:|-----------|----------------|
| ICAR | Intrinsic CAR (rho=1 fixed) | spatial_car(proper=FALSE) |
| pCAR | Proper CAR (rho estimated) | spatial_car(proper=TRUE) |
| BYM2 | Besag-York-Mollie 2 | spatial_bym2() |
| GP | Gaussian Process (NNGP) | spatial_gp() |
| HSGP | Hilbert Space GP | spatial_hsgp() |
| MSGP | Multi-scale GP | spatial_multiscale() |
| SVC | Spatially-varying coefficients | spatial_svc() |

## Temporal Types Reference

| Code | Full Name | Implementation |
|:----:|-----------|----------------|
| RW1 | Random Walk order 1 | temporal_rw1() |
| RW2 | Random Walk order 2 | temporal_rw2() |
| AR1 | Autoregressive order 1 | temporal_ar1() |
| GP_t | Temporal GP | temporal_gp() |
| MS_t | Multi-scale temporal | temporal_multiscale() |
| TVC | Time-varying coefficients | temporal_tvc() |

## ZI Types Reference

| Code | Full Name | Implementation |
|:----:|-----------|----------------|
| ZI | Zero-inflated | ratiod_zi*() or zi=zi_*() |
| Hurdle | Hurdle model | ratiod_hurdle_*() or zi=hurdle_*() |
| OI | One-inflated | ratiod_oibinomial() |
| ZOIB | Zero-and-one inflated | ratiod_zoibinomial() |

## Spatiotemporal Types Reference

| Code | Full Name | Description |
|:----:|-----------|-------------|
| ST-I | Knorr-Held Type I | Unstructured interaction |
| ST-II | Knorr-Held Type II | Temporal per location |
| ST-III | Knorr-Held Type III | Spatial per time |
| ST-IV | Knorr-Held Type IV | Full Kronecker product |

---

## Summary

### Benchmark Progress (2026-01-25)

| Status | Count | % |
|--------|------:|--:|
| Benchmarked (H timing) | 96 | 90% |
| Benchmarked (full 4-mode) | 18 | 17% |
| **SVC/TVC (H ready, needs bench)** | **6** | **6%** |
| **Total** | **107** | **100%** |

**All features now have H gradients:**
- SVC (rows 26, 56, 88) - NNGP-based, ready for benchmark
- TVC (rows 27, 57, 89) - RW1/RW2/AR1, ready for benchmark
- Latent factors (rows 30, 60, 92) - benchmarked with N=50 (see note below)

**Latent factor benchmark (N=50, K=2):**
| Row | Family | H(s) |
|-----|--------|-----:|
| 30 | poisson_gamma | 94.4 |
| 60 | negbin_negbin | 217.7 |
| 92 | binomial | 57.2 |

Standard N=500 exceeds timeout (N×K=1000 params). For larger N, use `mode = "vi"`.

**Newly benchmarked (38 models):**
- negbin GP spatial: rows 37-40 (92.2s, 0.3s, 14.0s, 9.5s)
- negbin temporal GP/MS: rows 44-45 (8.1s, 8.2s)
- negbin GP+temporal: rows 51-53 (101.2s, 0.4s, 14.2s)
- binomial GP spatial: rows 67-70 (105.6s, 3.1s, 645.2s, 9.9s)
- binomial temporal GP/MS: rows 74-75 (6.0s, 6.5s)
- binomial GP+temporal: rows 83-85 (113.4s, 3.6s, 656.2s)
- poisson_gamma MSGP: row 9 (605.3s)
- poisson_gamma GP+RW1: row 21 (135.3s)
- poisson_gamma MSGP+RW1: row 23 (571.7s)
- gamma_gamma family: rows 93-97 (31s-570s, SLOW)
- lognormal family: rows 98-102 (9.7s-212.6s)
- beta_binomial family: rows 103-107 (48s-1048s, SLOW)
- latent factor models: rows 30, 92 (2772s-4119s, VERY SLOW)

### Stan Validation Results (2026-01-24)

**Integrated benchmark + validation (10 models, all PASS):**

| Row | Model | numdenom | brms/Stan | Speedup | Diff | Status |
|-----|-------|----------|-----------|---------|------|--------|
| 1 | pg_base | 5.4s | 68.7s | **12.8x** | 0.0216 | **PASS** |
| 2 | pg_re | 4.5s | 69.4s | **15.4x** | 0.0132 | **PASS** |
| 5 | pg_icar | 5.5s | 79.3s | **14.5x** | 0.0141 | **PASS** |
| 11 | pg_rw1 | 5.0s | 8.0s | **1.6x** | 0.0133 | **PASS** |
| 31 | nb_base | 10.8s | 63.8s | **5.9x** | 0.0122 | **PASS** |
| 32 | nb_re | 11.3s | 75.1s | **6.6x** | 0.0132 | **PASS** |
| 61 | bin_base | 1.6s | 71.1s | **43.6x** | 0.0007 | **PASS** |
| 62 | bin_re | 1.9s | 85.9s | **44.7x** | 0.0009 | **PASS** |
| 65 | bin_icar | 2.1s | 76.1s | **35.9x** | 0.0011 | **PASS** |
| 71 | bin_rw1 | 2.1s | 13.1s | **6.3x** | 0.0011 | **PASS** |

**Notes:**
- All 10 models pass validation (posteriors within 2 SE of Stan)
- Binomial models fastest: **35-45x speedup** (virtually identical posteriors)
- Poisson-gamma models: **12-15x speedup**
- Negbin models: **6x speedup**
- Temporal models (RW1) have smaller speedups (brms efficient there too)
- Validation script: `benchmarks/bench_validated.R`
- Results saved: `benchmarks/results_validated.rds`

### Features by Implementation Status

| Feature | Status | H Gradient |
|---------|:------:|:----------:|
| Core families (pg, nb, bin) | ✓ | ✓ |
| Random intercepts | ✓ | ✓ |
| Random slopes (correlated) | ✓ | ✓ |
| Crossed RE | ✓ | ✓ |
| ICAR spatial | ✓ | ✓ |
| BYM2 spatial | ✓ | ✓ |
| GP spatial (NNGP) | ✓ | ✓ |
| HSGP spatial | ✓ | ✓ |
| MSGP (multi-scale) | ✓ | ✓ |
| Proper CAR | ✓ | ✓ |
| Temporal RW1/RW2/AR1 | ✓ | ✓ |
| Temporal GP | ✓ | ✓ |
| Multi-scale temporal | ✓ | ✓ |
| ZI/Hurdle (count) | ✓ | ✓ |
| ZI/Hurdle/OI/ZOIB (binomial) | ✓ | ✓ |
| SVC | ✓ | H |
| TVC | ✓ | H |
| Spatiotemporal (Knorr-Held) | ✓ | A |
| Latent factors | ✓ | H |
| gamma_gamma | ✓ | H |
| lognormal | ✓ | H |
| beta_binomial | ✓ | H |

### Key Findings

**Performance comparison (successful benchmarks):**
- **Core families (pg, nb, bin)**: 6-17s with H gradients
- **GP models**: 92-380s (O(N³) matrix operations)
- **HSGP**: 3-13s (20-100x faster than GP!)
- **MSGP**: 14-656s (varies by family, O(N³))
- **pCAR**: 9-10s (efficient sparse structure)
- **Temporal GP**: 6-10s (efficient temporal structure)
- **gamma_gamma family**: 31-570s (SLOW - needs optimization)
- **lognormal family**: 10-213s (moderate)
- **beta_binomial family**: 48-1048s (SLOW - needs optimization)
- **Latent factors (N=50)**: 57-218s (binomial fastest, negbin slowest)

**Performance tiers:**
| Tier | Time | Model types |
|:----:|:----:|-------------|
| Fast | <15s | Core families, HSGP, pCAR, temporal GP |
| Medium | 15-200s | lognormal, GP, GP+temporal |
| Slow | 200-700s | gamma_gamma, beta_binomial, MSGP |
| Very Slow | >1000s | Latent factors (high-dim, not gradient issue) |

**Latent factor note**: Times are slow due to high dimensionality (N×K params), not gradient efficiency. H gradients are O(N) which is optimal, but HMC struggles with 1000+ parameters. Consider `mode = "vi"` for latent factor models.

**All core features verified working:**
- Random slopes: ✓ (rows 3, 33, 63)
- Temporal (RW1/RW2/AR1): ✓ (rows 11-13, 41-43, 71-73)
- ZI/Hurdle: ✓ (rows 16-17, 46-47, 76-77)
- OI/ZOIB: ✓ (rows 78-79)
- Slopes + ICAR: ✓ (rows 25, 55, 87)
- GP: ✓ (rows 7, 37, 67)
- HSGP: ✓ (rows 8, 38, 68)
- MSGP: ✓ (rows 9, 39, 69)
- GP + temporal: ✓ (rows 21, 51, 83)
- HSGP + temporal: ✓ (rows 22, 52, 84)
- MSGP + temporal: ✓ (rows 23, 53, 85)
- gamma_gamma: ✓ (rows 93-97)
- lognormal: ✓ (rows 98-102)
- beta_binomial: ✓ (rows 103-107)
- Latent factors: ✓ (rows 30, 60, 92) - benchmarked at N=50

---

## Legend

- **H** = Hand-coded gradients (fastest, production default)
- **A** = Forward-mode autodiff (new implementation)
- **A_t** = Tape-based autodiff (legacy, heap allocation overhead)
- **N** = Numerical gradients (baseline)
- `-` = Bug prevents benchmarking
- Empty cells = benchmark not yet run
- A_t will remain available for gradient verification
