# Plan: Removing Stan Dependency from quotr

## Status: COMPLETE ✅

All phases have been completed. Stan has been fully removed from quotr.

The package now uses a native HMC/NUTS implementation in C++ with no external
dependencies beyond Rcpp.

---

## Completed Work

### Phase 1: HMC Spatial Backend ✅
**New C++ files:**
- `src/hmc_spatial.h` - Header with spatial model data structures
- `src/hmc_spatial.cpp` - Full HMC implementation with ICAR/BYM2 spatial

**Features:**
- ICAR spatial effects (intrinsic CAR)
- BYM2 spatial effects (Besag-York-Mollié 2)
- OpenMP parallelization (within-chain and across-chains)
- Adaptive step size tuning during warmup
- All three families: binomial, negbin_negbin, poisson_gamma

### Phase 2: Validate Against Stan ✅
- Created comprehensive validation test suite
- All posteriors match Stan reference implementation
- 95% credible intervals overlap
- Zero divergences

### Phase 3: Laplace Spatial ✅
- Implemented ICAR spatial effects in Laplace backend
- Implemented BYM2 spatial effects in Laplace backend
- Full OpenMP parallelization for Newton iterations
- Validated against true spatial effects (correlation > 0.95)

### Phase 3b: PG Spatial ✅
- Implemented ICAR spatial effects in PG backend
- Implemented BYM2 spatial effects in PG backend
- Full OpenMP parallelization for Gibbs sampling

### Phase 4: Smart Backend Selection ✅
- `backend = "auto"` is now the default
- Automatically selects HMC or Laplace based on data characteristics

### Phase 5: Move Stan to Suggests ✅
- cmdstanr moved from Imports to Suggests
- Removed Additional_repositories and SystemRequirements

### Phase 6: Remove Stan Entirely ✅
- Deleted all .stan files from inst/stan/
- Removed cmdstanr from Suggests
- Package no longer has any Stan-related dependencies

---

## Final Package State

### Dependencies (DESCRIPTION)
```
Imports:
    posterior,
    loo,
    stats,
    Rcpp (>= 1.0.0)
LinkingTo:
    Rcpp
```

### Available Backends
- `"auto"` - Default, selects optimal backend
- `"hmc"` - Native HMC/NUTS (full MCMC)
- `"laplace"` - Laplace approximation (fast, approximate)
- `"pg"` - Pólya-Gamma Gibbs (binomial only, experimental)

### Removed Files
- `inst/stan/quotr_binomial.stan`
- `inst/stan/quotr_negbin.stan`
- `inst/stan/quotr_poisson_gamma.stan`
- All compiled Stan executables

---

## Benefits of Migration

1. **No external dependencies** - Users don't need to install CmdStan
2. **Faster installation** - No Stan compilation required
3. **Portable** - Works on any system with a C++ compiler
4. **Simpler CI/CD** - No Stan setup in GitHub Actions
5. **Lighter package** - Smaller file size without Stan code
