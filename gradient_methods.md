# Gradient Methods by Model Configuration

## Model Space

```
Families (3):   BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA
Spatial (5):    NONE, ICAR, BYM2, GP, MULTISCALE_GP
Temporal (4):   NONE, RW1, RW2, AR1
ZI (3):         NONE, ZI, HURDLE
RE (4):         NONE, intercept, slopes, crossed
```

## Gradient Methods

| Method | Complexity | Speedup |
|--------|-----------|---------|
| H | Hand-coded O(n) | ~9x |
| A | Autodiff O(n) | ~3-5x |
| N | Numerical O(n×p) | ~0.6x |

---

## All Model Configurations

| # | Family | RE | Spatial | Temporal | ZI | Grad | Checked vs Stan |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|:---------------:|
| 1 | poisson_gamma | ✗ | ✗ | ✗ | ✗ | H | ✅ |
| 2 | poisson_gamma | ✓ | ✗ | ✗ | ✗ | H | ✅ |
| 3 | poisson_gamma | slopes | ✗ | ✗ | ✗ | A | ✅ 9.6x |
| 4 | poisson_gamma | crossed | ✗ | ✗ | ✗ | A | ✅ 15.2x |
| 5 | poisson_gamma | ✓ | ICAR | ✗ | ✗ | A | ✅ 14.6x |
| 6 | poisson_gamma | ✓ | BYM2 | ✗ | ✗ | A | ✅ 21.1x |
| 7 | poisson_gamma | ✓ | GP | ✗ | ✗ | N | ✅ 0.6x |
| 8 | poisson_gamma | ✓ | MSGP | ✗ | ✗ | N | ✅ 0.6x |
| 9 | poisson_gamma | ✓ | ✗ | RW1 | ✗ | A | ✅ 11.0x |
| 10 | poisson_gamma | ✓ | ✗ | RW2 | ✗ | A | ✅ 10.2x |
| 11 | poisson_gamma | ✓ | ✗ | AR1 | ✗ | A | ✅ 9.9x |
| ~~12~~ | ~~poisson_gamma~~ | ~~✓~~ | ~~✗~~ | ~~IID~~ | ~~✗~~ | - | N/A (not impl.) |
| 13 | poisson_gamma | ✓ | ✗ | ✗ | ZI | A | ✅ 10.3x |
| 14 | poisson_gamma | ✓ | ✗ | ✗ | Hurdle | A | ✅ 10.3x |
| 15 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | A | ✅ 25.2x |
| 16 | poisson_gamma | ✓ | BYM2 | RW1 | ✗ | A | ✅ 11.4x |
| 17 | poisson_gamma | ✓ | ICAR | AR1 | ✗ | A | ✅ 11.1x |
| 18 | poisson_gamma | ✓ | GP | RW1 | ✗ | N | ✅ 0.6x |
| 19 | poisson_gamma | ✓ | ICAR | ✗ | ZI | A | ✅ 10.2x |
| 20 | poisson_gamma | slopes | ICAR | ✗ | ✗ | A | ✅ 18.9x |
| 21 | negbin_negbin | ✗ | ✗ | ✗ | ✗ | A | ✅ 6.4x |
| 22 | negbin_negbin | ✓ | ✗ | ✗ | ✗ | A | ✅ 3.8x |
| 23 | negbin_negbin | slopes | ✗ | ✗ | ✗ | A | ✅ 5.7x |
| 24 | negbin_negbin | crossed | ✗ | ✗ | ✗ | A | ✅ 8.9x |
| 25 | negbin_negbin | ✓ | ICAR | ✗ | ✗ | A | ✅ 4.9x |
| 26 | negbin_negbin | ✓ | BYM2 | ✗ | ✗ | A | ✅ 7.9x |
| 27 | negbin_negbin | ✓ | GP | ✗ | ✗ | N | ✅ 0.6x |
| 28 | negbin_negbin | ✓ | ✗ | RW1 | ✗ | A | ✅ 6.2x |
| 29 | negbin_negbin | ✓ | ✗ | AR1 | ✗ | A | ✅ 8.6x |
| 30 | negbin_negbin | ✓ | ✗ | ✗ | ZI | A | ✅ 5.8x |
| 31 | negbin_negbin | ✓ | ✗ | ✗ | Hurdle | A | ✅ 5.7x |
| 32 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | A | ✅ 7.9x |
| 33 | binomial | ✗ | ✗ | ✗ | ✗ | A | ✅ 16.5x |
| 34 | binomial | ✓ | ✗ | ✗ | ✗ | A | ✅ 15.3x |
| 35 | binomial | slopes | ✗ | ✗ | ✗ | A | ✅ 15.2x |
| 36 | binomial | ✓ | ICAR | ✗ | ✗ | A | ✅ 13.5x |
| 37 | binomial | ✓ | BYM2 | ✗ | ✗ | A | ✅ 17.9x |
| 38 | binomial | ✓ | GP | ✗ | ✗ | N | ✅ 0.6x |
| 39 | binomial | ✓ | ✗ | RW1 | ✗ | A | ✅ 14.2x |
| 40 | binomial | ✓ | ✗ | AR1 | ✗ | A | ✅ 27.6x |
| 41 | binomial | ✓ | ICAR | RW1 | ✗ | A | ✅ 22.6x |

---

## Summary

| Grad | Count | % | Checked |
|:----:|------:|--:|--------:|
| H | 2 | 5% | 2 |
| A | 33 | 82% | 33 |
| N | 5 | 13% | 5 |
| **Total** | 40 | | **40/40** |

### Known Issues
- GP uses numerical gradients (~0.6x vs Stan) - autodiff not yet implemented for GP
  - **Recommendation**: Use ICAR/BYM2 spatial effects instead (13-22x faster than Stan)
- ~~GP spatial heisenbug~~ - **FIXED** (added LLT error check + diagonal jitter in `hmc_gp.h`)

---

## PUBLICATION REQUIREMENT

**Every single row must be H or A before submission.**

N (numerical) = 0.6x Stan speed = **UNACCEPTABLE FOR PAPER**

### Strategy
1. Keep H for rows 1-2 (simple poisson_gamma) - 9x speedup
2. Implement autodiff version of `compute_log_post` using `ad::Var`
3. All other rows get A automatically - estimated 3-5x speedup

### Implementation
```cpp
// Template the log_post to work with double OR ad::Var
template<typename T>
T compute_log_post_impl(const std::vector<T>& params, const ModelData& data, ...);

// For gradient:
ad::init_tape();
auto params_ad = ad::make_vars(params);
auto log_post = compute_log_post_impl(params_ad, data, ...);
log_post.backward();
grad = ad::get_adjoints(params_ad);
```

This gives O(n) gradients for ALL models with ONE implementation.
