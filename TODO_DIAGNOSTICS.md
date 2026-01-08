# quotr: Diagnostics & Plotting Features

Comparison with spOccupancy and other Bayesian R packages.

## Current State

### What quotr has:
- `print.quotr_fit()` - Basic model info
- `summary.quotr_fit()` - Parameter summaries with Rhat/ESS (all backends)
- `plot.quotr_fit()` - Traceplots and density plots (all backends)
- `mcmc_diagnostics()` - Rhat, ESS_bulk, ESS_tail computation
- `check_diagnostics()` - Diagnostic check with actionable warnings
- `n_divergent()` - Get divergence count
- `pp_check()` - Posterior predictive checks via bayesplot
- `loo()` / `waic()` - Model comparison
- Multi-chain HMC sampling

## Completed (Priority 1)

### 1. `plot.quotr_fit()` method
- [x] Traceplots with multi-chain overlay
- [x] Optional density plots alongside traces (`type = "dens"` or `type = "both"`)
- [x] Parameter selection (e.g., `plot(fit, pars = "beta")`)
- [x] Works with all backends (HMC, PG, Laplace)
- [x] Uses bayesplot if available, falls back to base R

### 2. Rhat / ESS for native backends
- [x] Compute split-Rhat for multi-chain HMC output
- [x] Compute bulk ESS and tail ESS
- [x] Display in `summary()` and `mcmc_diagnostics()`
- [x] Uses posterior package when available

### 3. Backend-agnostic `summary()`
- [x] Works with all backends (HMC, PG, Laplace)
- [x] Consistent output format
- [x] Shows Rhat/ESS for MCMC backends
- [x] Shows appropriate diagnostics per backend

### 4. Divergence reporting
- [x] Count divergent transitions (HMC backend)
- [x] Warning if divergences > 0 (in check_diagnostics)
- [x] Suggest remedies (reparameterization, etc.)
- [x] `n_divergent()` function for programmatic access

## Priority 2: Nice to Have

### 5. `as.mcmc()` / `as.mcmc.list()` conversion
- [ ] Convert draws to coda format for compatibility
- [ ] Enables use of coda::gelman.diag(), coda::effectiveSize()

### 6. `mcmc_*` bayesplot compatibility
- [x] Draws format works with bayesplot::mcmc_trace() (via plot.quotr_fit)
- [x] bayesplot::mcmc_dens_overlay() (via plot.quotr_fit)
- [ ] Direct bayesplot::mcmc_rhat(), mcmc_neff() integration

## Priority 3: Advanced

### 7. Geweke diagnostic
- [ ] For single-chain assessment
- [ ] Useful when multi-chain is impractical

### 8. `pairs()` method
- [ ] Bivariate posterior plots
- [ ] Highlight divergences

### 9. Energy diagnostics (HMC)
- [ ] E-BFMI computation
- [ ] Energy transition plots

## Usage Examples

```r
# Fit a model
fit <- quotr(count | effort ~ x + (1|site), data = df, family = quotr_poisson_gamma())

# Basic summary with Rhat/ESS
summary(fit)

# Plot traceplots
plot(fit)

# Plot densities with multi-chain overlay
plot(fit, type = "dens")

# Plot both trace and density
plot(fit, pars = "beta_num", type = "both")

# Get detailed diagnostics
mcmc_diagnostics(fit)

# Check diagnostics with actionable warnings
check_diagnostics(fit)

# Get divergence count
n_divergent(fit)
```

## Dependencies
- `posterior` package for Rhat/ESS (in Imports)
- `bayesplot` for visualization (in Suggests, optional)
- `coda` optional for compatibility (future)

## References

- [spOccupancy convergence guide](https://doserlab.com/files/spoccupancy-web/articles/modelconsiderations)
- [bayesplot MCMC diagnostics](https://mc-stan.org/bayesplot/articles/visual-mcmc-diagnostics.html)
- [Vehtari et al. Rhat paper](https://arxiv.org/abs/1903.08008)
