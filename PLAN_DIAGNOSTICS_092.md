# quotr v0.9.2: Enhanced Diagnostics Plan

## Current State

quotr has basic diagnostics:
- `mcmc_diagnostics()` - Returns Rhat, ESS_bulk, ESS_tail as data frame
- `check_diagnostics()` - Text-based check with warnings
- `plot.quotr_fit()` - Trace plots and density plots (uses bayesplot if available)
- `n_divergent()` - Get divergence count
- `pp_check()` - Posterior predictive checks via bayesplot

## Gap Analysis vs. bayesplot/spOccupancy

### Missing Visualizations

| Feature | bayesplot | spOccupancy | quotr |
|---------|-----------|-------------|-------|
| Trace plots | `mcmc_trace()` | `plot()` | `plot(fit)` |
| Density overlay | `mcmc_dens_overlay()` | - | `plot(fit, type="dens")` |
| Rhat plot | `mcmc_rhat()` | - | **MISSING** |
| ESS plot | `mcmc_neff()` | - | **MISSING** |
| Autocorrelation | `mcmc_acf()` | - | **MISSING** |
| Pairs plot | `mcmc_pairs()` | - | **MISSING** |
| Divergence scatter | `mcmc_scatter` + div | - | **MISSING** |
| Energy diagnostic | `mcmc_nuts_energy()` | - | **MISSING** |
| Parcoord | `mcmc_parcoord()` | - | **MISSING** |

### Missing Convenience Functions

| Feature | spOccupancy | quotr |
|---------|-------------|-------|
| Single summary plot | - | **MISSING** |
| Geweke diagnostic | `coda::geweke.diag` | **MISSING** |
| Combined diagnostic report | - | **MISSING** |

## v0.9.2 Implementation Plan

### Priority 1: Visual Diagnostics (High Impact)

#### 1.1 `plot_rhat()` - Rhat visualization
```r
plot_rhat(fit, threshold = 1.01)
```
- Points plot with color gradient (green < 1.01, yellow 1.01-1.05, red > 1.05)
- Horizontal reference line at threshold
- Works with/without bayesplot

#### 1.2 `plot_ess()` - Effective sample size
```r
plot_ess(fit, type = c("bulk", "tail"), threshold = 400)
```
- Ratio of ESS to total samples
- Color-coded by quality
- Reference line at threshold

#### 1.3 `plot_acf()` - Autocorrelation
```r
plot_acf(fit, pars = NULL, lags = 25)
```
- Grid of ACF plots by parameter
- Multi-chain overlay if available
- Helps diagnose slow mixing

#### 1.4 `plot_pairs()` - Bivariate posteriors
```r
plot_pairs(fit, pars = NULL, highlight_divergent = TRUE)
```
- Lower triangle: scatter plots
- Diagonal: histograms
- Upper triangle: correlation values
- Divergent transitions highlighted in red
- Critical for identifying non-identifiability

### Priority 2: HMC-Specific Diagnostics

#### 2.1 `plot_divergences()` - Divergence investigation
```r
plot_divergences(fit, pars = NULL)
```
- Parallel coordinates plot colored by divergent/non-divergent
- Scatter plot pairs with divergences highlighted
- Helps locate problematic posterior regions

#### 2.2 `plot_energy()` - Energy diagnostics (E-BFMI)
```r
plot_energy(fit)
```
- Overlaid histograms of E and E-transition
- Computes E-BFMI statistic
- Low E-BFMI indicates poor exploration

### Priority 3: Summary Functions

#### 3.1 `diagnostic_summary()` - Comprehensive report
```r
diagnostic_summary(fit)
```
Returns and optionally prints:
- Convergence status (PASS/WARN/FAIL)
- Divergence count with interpretation
- Rhat summary (worst parameters)
- ESS summary (worst parameters)
- E-BFMI (for HMC)
- Actionable recommendations

#### 3.2 `plot_diagnostics()` - Multi-panel diagnostic figure
```r
plot_diagnostics(fit)
```
Combined figure with:
- Top left: Rhat plot
- Top right: ESS plot
- Bottom left: Trace for worst Rhat parameter
- Bottom right: Energy (HMC) or ACF (other)

Requires `patchwork` for layout.

### Priority 4: Additional Helpers

#### 4.1 `geweke_test()` - Single-chain convergence
```r
geweke_test(fit, frac1 = 0.1, frac2 = 0.5)
```
- T-test between early and late chain portions
- Useful when running single chain
- Returns z-scores and p-values

#### 4.2 `as_draws()` - Export to posterior/bayesplot format
```r
as_draws(fit, format = c("array", "df", "list", "rvars"))
```
- Converts quotr draws to `posterior` package format
- Enables direct use of all bayesplot/posterior functions
- Bridge for power users

## Implementation Notes

### Dependencies
- All functions should work **without** bayesplot (base R fallback)
- When bayesplot is available, use it for nicer plots
- Suggest but don't require: `patchwork` for multi-panel

### Color Scheme (consistent across plots)
```r
# Diagnostic quality colors
quotr_colors <- list(
  good = "#2E7D32",     # Green - pass
  warn = "#F57F17",     # Yellow/orange - caution
  bad = "#C62828",      # Red - fail
  divergent = "#D32F2F" # Red for divergent transitions
)
```

### API Consistency
- All plot functions return ggplot2 objects (if ggplot2 available)
- All functions accept `pars` argument for parameter selection
- Use `print = TRUE/FALSE` for controlling output
- Match spOccupancy style where applicable

## File Organization

```
R/
├── diagnostics.R        # Main diagnostic functions (existing, rename from quotr.R section)
├── plot_diagnostics.R   # All diagnostic plotting functions (NEW)
```

## Testing Plan

1. Test all functions with/without bayesplot
2. Test with single chain and multi-chain fits
3. Test with HMC, PG, and Laplace backends
4. Test edge cases (no divergences, all good Rhat, etc.)

## Timeline Estimate

- Priority 1 (visual diagnostics): Core implementation
- Priority 2 (HMC-specific): Depends on HMC internals being exposed
- Priority 3 (summary functions): Wraps P1/P2
- Priority 4 (helpers): Nice-to-have

## References

- [bayesplot visual MCMC diagnostics](https://mc-stan.org/bayesplot/articles/visual-mcmc-diagnostics.html)
- [spOccupancy model considerations](https://doserlab.com/files/spoccupancy-web/articles/modelconsiderations)
- Vehtari et al. (2021) Rhat paper: https://arxiv.org/abs/1903.08008
- Gabry et al. (2019) Visualization in Bayesian workflow
