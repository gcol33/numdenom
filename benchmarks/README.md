# numdenom Benchmarks

Reproducible benchmarks validating speedups reported in `gradient_methods.md`.

## Benchmark Protocol

### Standard Parameters (for publication)

| Parameter | Value | Description |
|-----------|------:|-------------|
| `N_OBS` | 500 | Number of observations |
| `N_ITER` | 500 | Total iterations (warmup + sampling) |
| `N_WARMUP` | 250 | Warmup iterations |
| `N_CHAINS` | 1 | Single chain for timing consistency |
| `N_SITES` | 50 | Number of spatial sites |
| `N_TIMES` | 20 | Number of time points |

### Gradient Modes

| Code | Method | Description |
|:----:|--------|-------------|
| **N** | Numerical | Finite differences O(n×p), baseline/fallback |
| **A_t** | Tape autodiff | Tape-based autodiff, slow due to heap allocation |
| **A** | Forward autodiff | Expression template autodiff (fast) |
| **H** | Hand-coded | Analytical gradients (fastest, production default) |

### Data Generation

```r
set.seed(123)  # Reproducibility

# For count models (poisson_gamma, negbin_negbin)
y <- rpois(N, exp(2 + 0.5*x))       # or rnbinom()
effort <- rgamma(N, 10, 1)           # denominator

# For binomial models
y <- rbinom(N, trials, plogis(0.5 + 0.3*x))

# Spatial structure
adj_mat <- matrix(...)               # Adjacency based on distance ≤ 1.5
spatial_site <- factor(1:N_SITES)

# Temporal structure
time <- factor(rep(1:N_TIMES, length.out = N))
```

## Quick Start

```r
# Quick validation (100 iterations)
Rscript benchmarks/run_all.R --quick

# Full benchmark (500 iterations)
Rscript benchmarks/run_all.R

# Single row benchmark
Rscript benchmarks/bench_row3.R
```

## Files

| File | Description |
|------|-------------|
| `run_all.R` | Main benchmark runner |
| `helpers.R` | Data generation and timing functions |
| `bench_row*.R` | Individual row benchmarks |
| `bench_remaining_pg.R` | Remaining poisson_gamma benchmarks |
| `bench_remaining_nb.R` | Remaining negbin_negbin benchmarks |
| `results.rds` | Saved results (after running) |

## Coverage

**62 model configurations** across 3 families:

| Family | Configs | Description |
|--------|--------:|-------------|
| poisson_gamma | 20 | Rows 1-20 |
| negbin_negbin | 20 | Rows 21-40 |
| binomial | 22 | Rows 41-62 |

## Requirements

- `numdenom` package installed
- `brms` package for Stan comparison (optional)
