# Hand-Coded Gradient Implementation Plan

## Current Status
- **Completed**: 37/60 configurations (62%) have H or H* gradients
- **Remaining**: 23/60 configurations (38%) use autodiff (A)

## Completed (H)
| # | Family | RE | Spatial | Temporal | ZI |
|--:|--------|:--:|:-------:|:--------:|:--:|
| 1 | poisson_gamma | ✗ | ✗ | ✗ | ✗ |
| 2 | poisson_gamma | ✓ | ✗ | ✗ | ✗ |
| 9 | poisson_gamma | ✓ | ✗ | RW1 | ✗ |
| 10 | poisson_gamma | ✓ | ✗ | RW2 | ✗ |
| 11 | poisson_gamma | ✓ | ✗ | AR1 | ✗ |
| 12 | poisson_gamma | ✓ | ✗ | ✗ | ZI |
| 13 | poisson_gamma | ✓ | ✗ | ✗ | Hurdle |
| 21 | negbin_negbin | ✗ | ✗ | ✗ | ✗ |
| 22 | negbin_negbin | ✓ | ✗ | ✗ | ✗ |
| 29 | negbin_negbin | ✓ | ✗ | RW1 | ✗ |
| 30 | negbin_negbin | ✓ | ✗ | RW2 | ✗ |
| 31 | negbin_negbin | ✓ | ✗ | AR1 | ✗ |
| 32 | negbin_negbin | ✓ | ✗ | ✗ | ZI |
| 33 | negbin_negbin | ✓ | ✗ | ✗ | Hurdle |
| 41 | binomial | ✗ | ✗ | ✗ | ✗ |
| 42 | binomial | ✓ | ✗ | ✗ | ✗ |
| 49 | binomial | ✓ | ✗ | RW1 | ✗ |
| 50 | binomial | ✓ | ✗ | RW2 | ✗ |
| 51 | binomial | ✓ | ✗ | AR1 | ✗ |
| 5 | poisson_gamma | ✓ | ICAR | ✗ | ✗ |
| 25 | negbin_negbin | ✓ | ICAR | ✗ | ✗ |
| 45 | binomial | ✓ | ICAR | ✗ | ✗ |
| 6 | poisson_gamma | ✓ | BYM2 | ✗ | ✗ |
| 26 | negbin_negbin | ✓ | BYM2 | ✗ | ✗ |
| 46 | binomial | ✓ | BYM2 | ✗ | ✗ |
| 14 | poisson_gamma | ✓ | ICAR | RW1 | ✗ |
| 15 | poisson_gamma | ✓ | BYM2 | RW1 | ✗ |
| 16 | poisson_gamma | ✓ | ICAR | AR1 | ✗ |
| 34 | negbin_negbin | ✓ | ICAR | RW1 | ✗ |
| 35 | negbin_negbin | ✓ | BYM2 | RW1 | ✗ |
| 36 | negbin_negbin | ✓ | ICAR | AR1 | ✗ |
| 54 | binomial | ✓ | ICAR | RW1 | ✗ |
| 55 | binomial | ✓ | BYM2 | RW1 | ✗ |
| 56 | binomial | ✓ | ICAR | AR1 | ✗ |

## Phase 1: Temporal Models (Easy)
Temporal priors have simple O(T) gradients.

| # | Family | RE | Temporal | Priority |
|--:|--------|:--:|:--------:|:--------:|
| 9 | poisson_gamma | ✓ | RW1 | High |
| 10 | poisson_gamma | ✓ | RW2 | High |
| 11 | poisson_gamma | ✓ | AR1 | High |
| 29 | negbin_negbin | ✓ | RW1 | High |
| 30 | negbin_negbin | ✓ | RW2 | High |
| 31 | negbin_negbin | ✓ | AR1 | High |
| 49 | binomial | ✓ | RW1 | High |
| 50 | binomial | ✓ | RW2 | High |
| 51 | binomial | ✓ | AR1 | High |

**Gradient formulas:**
- RW1: `d/dφ[t] = τ * (φ[t-1] - 2*φ[t] + φ[t+1])` (interior)
- RW2: second difference stencil
- AR1: `d/dφ[t] = τ * (φ[t] - ρ*φ[t-1]) + ρ*τ*(φ[t+1] - ρ*φ[t])`

## Phase 2: ICAR Spatial (Medium)
ICAR has O(edges) gradient via adjacency list.

| # | Family | RE | Spatial |
|--:|--------|:--:|:-------:|
| 5 | poisson_gamma | ✓ | ICAR |
| 25 | negbin_negbin | ✓ | ICAR |
| 45 | binomial | ✓ | ICAR |

## Phase 3: BYM2 Spatial (Medium)
BYM2 = ICAR + IID with mixing parameter.

| # | Family | RE | Spatial |
|--:|--------|:--:|:-------:|
| 6 | poisson_gamma | ✓ | BYM2 |
| 26 | negbin_negbin | ✓ | BYM2 |
| 46 | binomial | ✓ | BYM2 |

## Phase 4: Zero-Inflation (Medium) - DONE

| # | Family | RE | ZI | Status |
|--:|--------|:--:|:--:|:------:|
| 12 | poisson_gamma | ✓ | ZI | ✅ H |
| 13 | poisson_gamma | ✓ | Hurdle | ✅ H |
| 32 | negbin_negbin | ✓ | ZI | ✅ H |
| 33 | negbin_negbin | ✓ | Hurdle | ✅ H |
| 52 | binomial | ✓ | ZI | A (uses ratiod_zibinomial) |
| 53 | binomial | ✓ | Hurdle | A (uses ratiod_hurdle_binomial) |

**Note**: Binomial ZI/Hurdle (rows 52, 53) use separate family classes (`ratiod_zibinomial()`, `ratiod_hurdle_binomial()`) rather than the `zi=` parameter, so they remain autodiff.

## Phase 5: Spatial + Temporal (Medium-Hard) - DONE

| # | Family | Spatial | Temporal | Status |
|--:|--------|:-------:|:--------:|:------:|
| 14 | poisson_gamma | ICAR | RW1 | ✅ H |
| 15 | poisson_gamma | BYM2 | RW1 | ✅ H |
| 16 | poisson_gamma | ICAR | AR1 | ✅ H |
| 34 | negbin_negbin | ICAR | RW1 | ✅ H |
| 35 | negbin_negbin | BYM2 | RW1 | ✅ H |
| 36 | negbin_negbin | ICAR | AR1 | ✅ H |
| 54 | binomial | ICAR | RW1 | ✅ H |
| 55 | binomial | BYM2 | RW1 | ✅ H |
| 56 | binomial | ICAR | AR1 | ✅ H |

## Phase 6: Random Slopes (Hard) - PARTIAL

| # | Family | RE | Status |
|--:|--------|:------:|:------:|
| 3 | poisson_gamma | slopes | ✅ H* (uncorrelated) |
| 23 | negbin_negbin | slopes | ✅ H* (uncorrelated) |
| 43 | binomial | slopes | ✅ H* (uncorrelated) |

**Note**: H* means hand-coded for uncorrelated slopes (`||` syntax). Correlated slopes (`|` syntax) require complex Cholesky gradient computation and remain autodiff.

## Phase 7: Crossed RE (Hard)
| # | Family | RE |
|--:|--------|:------:|
| 4 | poisson_gamma | crossed |
| 24 | negbin_negbin | crossed |
| 44 | binomial | crossed |

## Phase 8: GP Models (Keep Autodiff)
GP has O(n²/n³) - autodiff is appropriate.
- Rows 7, 8, 17, 20, 27, 28, 37, 40, 47, 48, 57, 60

## Implementation Order
1. ✅ Basic models (6 configs) - DONE
2. ✅ Phase 1: Temporal (9 configs) - DONE
3. ✅ Phase 2: ICAR (3 configs) - DONE
4. ✅ Phase 3: BYM2 (3 configs) - DONE
5. ✅ Phase 4: ZI/Hurdle (4/6 configs) - DONE (binomial ZI uses separate family classes, kept A)
6. ✅ Phase 5: Spatial+Temporal (9 configs) - DONE
7. ✅ Phase 6: Slopes (3 configs) - PARTIAL (uncorrelated only, H*)
8. ⬜ Phase 7: Crossed (3 configs)
9. ⬜ Phase 8: GP (keep A)
