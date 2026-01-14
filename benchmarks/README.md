# numdenom Benchmarks

Reproducible benchmarks validating speedups reported in `gradient_methods.md`.

## Quick Start

```r
# Quick validation (100 iterations)
Rscript benchmarks/run_all.R --quick

# Full benchmark (400 iterations)
Rscript benchmarks/run_all.R
```

## Coverage

**35/40 model configurations validated** (5 GP configs have known issues)

| Family | Configs | Speedup Range |
|--------|---------|---------------|
| poisson_gamma | 16 | 9.6x - 25.2x |
| negbin_negbin | 11 | 3.8x - 8.9x |
| binomial | 8 | 13.5x - 27.6x |

## Files

| File | Description |
|------|-------------|
| `run_all.R` | Main benchmark runner |
| `helpers.R` | Data generation and timing functions |
| `results.rds` | Saved results (after running) |

## Gradient Methods

| Code | Method | Typical Speedup |
|:----:|--------|-----------------|
| H | Hand-coded analytical | ~9x |
| A | Autodiff | 3.8x - 27.6x |
| N | Numerical (GP only) | ~0.6x |

## Requirements

- `numdenom` package installed
- `brms` package for Stan comparison (optional)

Without brms, benchmarks will run numdenom only and report times without speedup ratios.

## Skipped Configs

Rows 7, 8, 18, 27, 38 (GP spatial) are skipped due to known heisenbug.
See `debug_gp_crash.md` and `feature/gp-heisenbug` branch for details.
