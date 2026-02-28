# Hand-Coded Gradient Implementation Plan

## Gradient Progression Rule

**Strict rule: N → A → H**

All gradient implementations must follow this progression:
1. **N (Numerical)**: Initial implementation using finite differences
2. **A (Autodiff)**: Add to templated `log_post_impl.h` for automatic differentiation
3. **H (Hand-coded)**: Optimize with analytical gradients when A is validated

Never skip from N directly to H. A serves as the validation reference.

---

## Current Status by Feature

| Feature | C++ Header | N | A_t | H | Notes |
|---------|-----------|:-:|:---:|:-:|-------|
| Basic models | `hmc_sampler.cpp` | ✓ | ✓ | ✓ | Complete |
| RE intercepts | `log_post_impl.h` | ✓ | ✓ | ✓ | Complete |
| RE slopes | `hmc_sampler.cpp` | ✓ | ✗ | ✓ | Skipped A_t (verify!) |
| Crossed RE | `hmc_sampler.cpp` | ✓ | ✗ | ✓ | Skipped A_t (verify!) |
| ICAR spatial | `log_post_impl.h` | ✓ | ✓ | ✓ | Complete |
| BYM2 spatial | `log_post_impl.h` | ✓ | ✓ | ✓ | Complete |
| GP spatial | `hmc_gp.h` | ✓ | ✗ | ✓ | H has NNGP gradients |
| MSGP spatial | `hmc_gp.h` | ✓ | ✗ | ✓ | Complete (multi-family) |
| HSGP spatial | `hmc_hsgp.h` | ✓ | ✗ | ✓ | Complete (multi-family) |
| SVC | `hmc_svc.h` + `hmc_svc_autodiff.h` | ✓ | ✓ | ✓ | **Complete** |
| Temporal RW/AR | `log_post_impl.h` | ✓ | ✓ | ✓ | Complete |
| Temporal GP | `hmc_temporal_gp.h` | ✓ | ✗ | ✓ | Complete |
| Multi-scale temporal | `hmc_multiscale_temporal_grad.h` + `hmc_temporal_multiscale_autodiff.h` | ✓ | ✓ | ✓ | **Complete** |
| TVC | `hmc_tvc.h` + `hmc_tvc_autodiff.h` + `hmc_tvc_grad.h` | ✓ | ✓ | ✓ | **Complete** |
| Spatiotemporal | `hmc_spatiotemporal.h` | ✓ | ✗ | ✓ | Complete (Types I-IV) |
| Latent factors | `hmc_latent.h` + `hmc_latent_autodiff.h` + `hmc_latent_grad.h` | ✓ | ✓ | ✓ | **Complete** |
| ZI (count) | `hmc_zi.h` | ✓ | ✗ | ✓ | H implemented |
| ZI binomial | `hmc_zi.h` | ✓ | ✗ | ✓ | H implemented |
| OI binomial | `hmc_zi.h` | ✓ | ✗ | ✓ | Complete |
| ZOIB | `hmc_zi.h` | ✓ | ✗ | ✓ | Complete |

**Legend**: ✓ = implemented, ✗ = missing

**SVC Implementation Details:**
- `hmc_svc.h`: Core NNGP likelihood, `svc_nngp_gradients()` hand-coded
- `hmc_svc_autodiff.h`: Templated `nngp_log_lik<T>()` for autodiff
- `hmc_sampler.cpp`: `compute_gradient_svc_handcoded()` at line 3454

**TVC Implementation Details:**
- `hmc_tvc.h`: Core RW1/RW2/AR1/IID priors
- `hmc_tvc_autodiff.h`: Templated priors for autodiff
- `hmc_tvc_grad.h`: Analytical gradients `rw1_grad_w`, `rw2_grad_w`, `ar1_grad_w`
- `hmc_sampler.cpp`: `compute_gradient_tvc_handcoded()` at line 3707

---

## Implementation Phases

### Completed Features

The following features are now fully implemented with N, A_t, and H:

#### ✅ SVC (Spatially-Varying Coefficients) - COMPLETE
- `hmc_svc.h`: Core NNGP likelihood and hand-coded `svc_nngp_gradients()`
- `hmc_svc_autodiff.h`: Templated `nngp_log_lik<T>()` with Cholesky decomposition
- `hmc_sampler.cpp`: `compute_gradient_svc_handcoded()` integrated at line 5180
- **Status**: Ready for benchmarking

#### ✅ TVC (Temporally-Varying Coefficients) - COMPLETE
- `hmc_tvc.h`: Core RW1/RW2/AR1/IID priors
- `hmc_tvc_autodiff.h`: Templated temporal priors for autodiff
- `hmc_tvc_grad.h`: Analytical gradients (`rw1_grad_w`, `rw2_grad_w`, `ar1_grad_w`)
- `hmc_sampler.cpp`: `compute_gradient_tvc_handcoded()` integrated at line 5183
- **Status**: Ready for benchmarking

#### ✅ OI/ZOIB Binomial - COMPLETE
- Already implemented in `hmc_zi.h` with H gradients
- Benchmarked in rows 78-79

#### ✅ Multi-scale Temporal (MS_t) - COMPLETE
- `hmc_multiscale_temporal_grad.h`: Hand-coded gradients for RW1, RW2, AR1, IID priors
- `hmc_temporal_multiscale_autodiff.h`: Templated functions for A_t mode
- `hmc_sampler.cpp`: `compute_gradient_multiscale_temporal_handcoded()` at line 4374
- **Benchmark**: H=73s, A=180s, A_t=241s, N=92s (2.5x speedup H/A)
- **Status**: Complete

#### ✅ Latent Factors - COMPLETE
- `hmc_latent.h`: Core functions (constraint application, priors)
- `hmc_latent_autodiff.h`: Templated functions for A_t mode
- `hmc_latent_grad.h`: Hand-coded gradients
- `hmc_sampler.cpp`: `compute_gradient_latent_handcoded()` integrated
- **Benchmark (N=50, K=2)**: poisson_gamma 0.4s, negbin_negbin 0.8s, binomial 0.1s, all 0 div
- **Status**: Benchmarked and validated

---

### All Core Features Complete

All model configurations now have H (hand-coded) gradients:
- Core families (poisson_gamma, negbin_negbin, binomial)
- Random intercepts and slopes
- Spatial (ICAR, BYM2, GP, HSGP, MSGP, SVC)
- Temporal (RW1, RW2, AR1, GP, TVC)
- Spatiotemporal (Types I-IV)
- ZI/Hurdle/OI/ZOIB
- **Latent factors**

---

## Validation Protocol

For each feature moving from A_t to H:

1. Generate test data with known parameters
2. Compute gradient using A_t (autodiff)
3. Compute gradient using H (hand-coded)
4. Check: `max(|grad_H - grad_A_t|) / max(|grad_A_t|) < 1e-5`
5. Benchmark: H should be at least 3x faster than A_t
6. Run full model fit and compare posteriors

---

## Next Benchmarks Required

All features now have H gradients. Priority benchmarks:

1. ~~**Benchmark SVC rows (26, 56, 88)**: Run with H gradient mode, validate against Stan~~
   - ✅ **Row 26 (poisson_gamma + SVC)**: 759.0s H; NNGP requires custom Stan (deferred)
   - ✅ **Row 56 (negbin_negbin + SVC)**: 764.6s H; NNGP requires custom Stan (deferred)
   - ✅ **Row 88 (binomial + SVC)**: 708.6s H; NNGP requires custom Stan (deferred)
   - **Note**: SVC uses NNGP approximation. Stan validation would require implementing NNGP from scratch in Stan, which is complex. H gradients are correct (validated against A_t).
2. ~~**Benchmark TVC rows (27, 57, 89)**: Run with H gradient mode, validate against Stan~~
   - ✅ **Row 27 (poisson_gamma + TVC)**: 6.6s H, 0 div; Stan 12.4s, 0 div; **VALIDATED**
   - ✅ **Row 57 (negbin_negbin + TVC)**: 15.0s H, 0 div; **VALIDATED**
   - ✅ **Row 89 (binomial + TVC)**: 13.0s H, 0 div; **VALIDATED**
3. ~~**Benchmark Latent rows (30, 60, 92)**: Run with H gradient mode (expected 10-50x speedup)~~
   - ✅ **Row 30 (poisson_gamma + latent)**: 0.4s H, 0 div; **VALIDATED**
   - ✅ **Row 60 (negbin_negbin + latent)**: 0.8s H, 0 div; **VALIDATED**
   - ✅ **Row 92 (binomial + latent)**: 0.1s H, 0 div; **VALIDATED**
