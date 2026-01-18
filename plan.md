# Hand-Coded Gradient Implementation Plan

## Current Status
- **Completed**: 21/60 configurations (35%) have H gradients
- **Remaining**: 39/60 configurations (65%) use autodiff (A)

## Completed (H)
| # | Family | RE | Spatial | Temporal | ZI |
|--:|--------|:--:|:-------:|:--------:|:--:|
| 1 | poisson_gamma | ✗ | ✗ | ✗ | ✗ |
| 2 | poisson_gamma | ✓ | ✗ | ✗ | ✗ |
| 9 | poisson_gamma | ✓ | ✗ | RW1 | ✗ |
| 10 | poisson_gamma | ✓ | ✗ | RW2 | ✗ |
| 11 | poisson_gamma | ✓ | ✗ | AR1 | ✗ |
| 21 | negbin_negbin | ✗ | ✗ | ✗ | ✗ |
| 22 | negbin_negbin | ✓ | ✗ | ✗ | ✗ |
| 29 | negbin_negbin | ✓ | ✗ | RW1 | ✗ |
| 30 | negbin_negbin | ✓ | ✗ | RW2 | ✗ |
| 31 | negbin_negbin | ✓ | ✗ | AR1 | ✗ |
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

## Phase 4: Zero-Inflation (Medium)

| # | Family | RE | ZI |
|--:|--------|:--:|:--:|
| 12 | poisson_gamma | ✓ | ZI |
| 13 | poisson_gamma | ✓ | Hurdle |
| 32 | negbin_negbin | ✓ | ZI |
| 33 | negbin_negbin | ✓ | Hurdle |
| 52 | binomial | ✓ | ZI |
| 53 | binomial | ✓ | Hurdle |

## Phase 5: Spatial + Temporal (Medium-Hard)
| # | Family | Spatial | Temporal |
|--:|--------|:-------:|:--------:|
| 14-16 | poisson_gamma | ICAR/BYM2 | RW1/AR1 |
| 34-36 | negbin_negbin | ICAR/BYM2 | RW1/AR1 |
| 54-56 | binomial | ICAR/BYM2 | RW1/AR1 |

## Phase 6: Random Slopes (Hard)
| # | Family | RE |
|--:|--------|:------:|
| 3 | poisson_gamma | slopes |
| 23 | negbin_negbin | slopes |
| 43 | binomial | slopes |

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
5. ⬜ Phase 4: ZI/Hurdle (6 configs) - START HERE
6. ⬜ Phase 5: Spatial+Temporal (9 configs)
7. ⬜ Phase 6: Slopes (3 configs)
8. ⬜ Phase 7: Crossed (3 configs)
9. ⬜ Phase 8: GP (keep A)
