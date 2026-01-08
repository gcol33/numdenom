# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**quotr** is an opinionated R package for Bayesian hierarchical modelling of ratios, rates, and proportions in ecological and related data.

### Core Philosophy (Non-Negotiable)

**Ratios are not data. They are derived quantities.**

All inference is performed on the latent processes generating the numerator and denominator, never on their quotient. Ratios, rates, and proportions are computed post hoc with full uncertainty propagation.

- Shared latent structure is the **default**, not an option
- Independence between numerator and denominator must be explicitly requested (triggers warning)
- `offset()` is explicitly forbidden — enforces the philosophy
- Conceptually analogous to occupancy models (spOccupancy), but for ratio-based inference

## Development Commands

### Build and Install
```r
# Load package for development
devtools::load_all()

# Install package locally
devtools::install()

# Regenerate documentation (after roxygen2 changes)
devtools::document()

# Clean and reinstall (if C++ code changed)
devtools::clean_dll()
devtools::install()
```

### Testing
```r
# Run full test suite
devtools::test()

# Run complete package check
devtools::check()

# Run specific test file
testthat::test_file("tests/testthat/test-formula.R")
```

### Documentation
```r
# Build vignettes (REQUIRED: use centralized build script)
source("~/.R/build_pkgdown.R")
build_pkgdown_site()

# Or build vignettes only
devtools::build_vignettes()
```

## Architecture

### Inference Backends

quotr uses **native C++ backends** - no Stan dependency required.

| Backend | Description | Best For |
|---------|-------------|----------|
| `"auto"` | Auto-select optimal | Default, recommended |
| `"hmc"` | Native HMC/NUTS | Full MCMC, all models |
| `"laplace"` | Laplace approximation | Very large data (N > 50k) |
| `"pg"` | Pólya-Gamma Gibbs | Binomial (experimental) |

### Model Families

Three distribution families cover common ecological use cases:

| Family | Numerator | Denominator | Use Case |
|--------|-----------|-------------|----------|
| `quotr_negbin_negbin()` | NegBin | NegBin | Count/count ratios (relative abundance) |
| `quotr_binomial()` | Binomial | Fixed trials | Success/trial proportions |
| `quotr_poisson_gamma()` | Poisson | Gamma | CPUE (count/continuous effort) |

### Data Flow

```
quotr() ─┬─► quotr_formula()      # Parse numerator/denominator/shared
         │      ├─► validate_formula()    # Check syntax
         │      ├─► check_no_offset()     # Block offset() usage
         │      └─► parse_shared()        # Handle shared structure
         │
         ├─► select_backend()     # Auto-select optimal backend
         │
         ├─► fit_hmc() / fit_laplace() / fit_pg()   # Backend dispatch
         │      ├─► prepare_hmc_data()    # Data preparation
         │      └─► cpp_hmc_spatial()     # C++ HMC sampler
         │
         └─► quotr_fit object
                    │
                    ├─► ratio()           # Extract ratio posteriors
                    ├─► ratio_contrast()  # Compare conditions
                    ├─► pp_check()        # Posterior predictive
                    └─► loo()/waic()      # Model comparison
```

### Key Design Decisions

1. **Shared structure by default**: When random effects appear in both formulas, they are shared by default. This prevents spurious ratio effects from common unmeasured drivers.

2. **Non-centered parameterization**: All random effects use `z ~ std_normal()` with scaling, improving sampling efficiency for hierarchical models.

3. **PC priors**: Penalized complexity priors favor simpler models (smaller variance components). Default: P(sigma > 1) = 0.01.

4. **Ratios computed post hoc**: Never modeled directly. Computed as `exp(eta_num - eta_denom)` with full posterior uncertainty.

## File Organization

```
R/
├── quotr.R              # Main fitting function, print/summary methods
├── formula.R            # Formula parsing, offset blocking, shared inference
├── family.R             # quotr_negbin_negbin(), quotr_binomial(), quotr_poisson_gamma()
├── backend_hmc.R        # HMC/NUTS backend
├── backend_laplace.R    # Laplace approximation backend
├── backend_pg.R         # Pólya-Gamma Gibbs backend (binomial)
├── standata.R           # Data preparation for backends
├── ratio.R              # ratio(), ratio_contrast() extraction
├── spatial.R            # spatial_car(), spatial_bym2()
├── priors.R             # quotr_priors(), PC prior helpers
├── validate.R           # pp_check(), loo(), waic(), quotr_compare()
├── quotr-package.R      # Package documentation
└── zzz.R                # Package initialization

src/
├── hmc_spatial.cpp      # Main HMC sampler with spatial support
├── hmc_spatial.h        # HMC data structures
├── hmc_core.cpp         # Core HMC functions
├── hmc_simple.cpp       # Simple HMC implementation
├── laplace_core.cpp     # Laplace approximation
├── pg_binomial.cpp      # Pólya-Gamma Gibbs sampler
├── pg_rng.cpp           # Pólya-Gamma random number generation
└── autodiff.cpp         # Automatic differentiation helpers

tests/testthat/
├── test-formula.R       # Formula parsing, offset rejection
├── test-family.R        # Family creation and validation
├── test-backends.R      # Backend functionality tests
├── test-hmc.R           # HMC sampler tests
├── test-hmc-spatial.R   # Spatial HMC tests
└── test-spatial.R       # Spatial structure validation

vignettes/
├── philosophy.Rmd       # Why ratios are not data
└── getting-started.Rmd  # Basic usage examples
```

## Important Conventions

### Formula Specification

**Combined syntax (recommended):**
```r
quotr(
  count | effort ~ x + (1 | site),       # num | denom ~ predictors
  data = df,
  family = quotr_poisson_gamma()
)
```

**With process-specific terms:**
```r
quotr(
  count | effort ~ (1 | site),           # Shared structure
  formula_num = ~ depth + season,         # Numerator-only predictors
  formula_denom = ~ weather,              # Denominator-only predictors
  data = df,
  family = quotr_poisson_gamma()
)
```

- `shared = NULL` (default): Infer from matching RE in both formulas
- `shared = ~ (1 | group)`: Explicit shared random intercepts
- `shared = ~ 0`: Independence assumption (**triggers warning**)

### Random Effects Indexing

- R layer: 1-based indexing
- C++ layer: 0-based internally, 1-based at interface
- `re_idx[n, g]` gives the group index for observation n in grouping factor g

### Spatial Structure

- `level = "group"`: Spatial effect at site level (requires `group_var`)
- `level = "obs"`: Spatial effect at observation level
- Adjacency matrix must be symmetric
- ICAR prior with soft sum-to-zero constraint

## Testing Guidelines

- Use `set.seed()` for reproducibility
- Test formula parsing separately from model fitting
- Include edge cases: no RE, single RE, shared vs separate
- Test offset rejection explicitly
- Test independence warning with `shared = ~ 0`

## C++ Development

When modifying C++ code:

1. Edit source in `src/`
2. Run `devtools::load_all()` to recompile
3. Test with small data first (N=50, short chains)
4. Check for memory issues with valgrind (Linux)
5. Ensure OpenMP parallelization works correctly

## Dependencies

**Runtime:**
- posterior (draw manipulation)
- loo (model comparison)
- Rcpp (C++ interface)

**Suggested (optional):**
- bayesplot (visualization)
- Matrix (sparse adjacency)

**Development:**
- testthat (testing)
- devtools (package development)
- roxygen2 (documentation)
- pkgdown (website)

## Version Roadmap

### v1.0 (Current)
- Three families: negbin_negbin, binomial, poisson_gamma
- Shared random effects (default)
- Spatial CAR/BYM2
- Native HMC/NUTS backend (no Stan)
- LOO/WAIC model comparison

### v1.1 (Planned)
- Latent factors for unmeasured confounders
- GP/Matérn spatial (continuous)
- Temporal AR(1) / random walk

### v2.0 (Future)
- Spatiotemporal interaction
- Zero-inflation variants
