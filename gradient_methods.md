# Gradient Methods by Model Configuration

## Gradient Methods

| Method | Complexity | Speedup vs Stan |
|--------|-----------|-----------------|
| H | Hand-coded O(n) | ~9-25x |
| A | Autodiff O(n) | ~3-15x |

## All Model Configurations

| # | Family | RE | Spatial | Temporal | ZI | Grad |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|
| 1 | poisson_gamma | ✗ | ✗ | ✗ | ✗ | H |
| 2 | poisson_gamma | ✓ | ✗ | ✗ | ✗ | H |
| 3 | poisson_gamma | slopes | ✗ | ✗ | ✗ | H |
| 4 | poisson_gamma | crossed | ✗ | ✗ | ✗ | A |
| 5 | poisson_gamma | ✓ | ICAR | ✗ | ✗ | H |
| 6 | poisson_gamma | ✓ | BYM2 | ✗ | ✗ | H |
| 7 | poisson_gamma | ✓ | GP | ✗ | ✗ | A |
| 8 | poisson_gamma | ✓ | MSGP | ✗ | ✗ | A |
| 9 | poisson_gamma | ✓ | ✗ | RW1 | ✗ | H |
| 10 | poisson_gamma | ✓ | ✗ | RW2 | ✗ | H |
| 11 | poisson_gamma | ✓ | ✗ | AR1 | ✗ | H |
| 12 | poisson_gamma | ✓ | ✗ | ✗ | ZI | H |
| 13 | poisson_gamma | ✓ | ✗ | ✗ | Hurdle | H |
| 14 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H |
| 15 | poisson_gamma | ✓ | BYM2 | RW1 | ✗ | H |
| 16 | poisson_gamma | ✓ | ICAR | AR1 | ✗ | H |
| 17 | poisson_gamma | ✓ | GP | RW1 | ✗ | A |
| 18 | poisson_gamma | ✓ | ICAR | ✗ | ZI | A |
| 19 | poisson_gamma | slopes | ICAR | ✗ | ✗ | A |
| 20 | poisson_gamma | ✓ | MSGP | RW1 | ✗ | A |
| 21 | negbin_negbin | ✗ | ✗ | ✗ | ✗ | H |
| 22 | negbin_negbin | ✓ | ✗ | ✗ | ✗ | H |
| 23 | negbin_negbin | slopes | ✗ | ✗ | ✗ | H |
| 24 | negbin_negbin | crossed | ✗ | ✗ | ✗ | A |
| 25 | negbin_negbin | ✓ | ICAR | ✗ | ✗ | H |
| 26 | negbin_negbin | ✓ | BYM2 | ✗ | ✗ | H |
| 27 | negbin_negbin | ✓ | GP | ✗ | ✗ | A |
| 28 | negbin_negbin | ✓ | MSGP | ✗ | ✗ | A |
| 29 | negbin_negbin | ✓ | ✗ | RW1 | ✗ | H |
| 30 | negbin_negbin | ✓ | ✗ | RW2 | ✗ | H |
| 31 | negbin_negbin | ✓ | ✗ | AR1 | ✗ | H |
| 32 | negbin_negbin | ✓ | ✗ | ✗ | ZI | H |
| 33 | negbin_negbin | ✓ | ✗ | ✗ | Hurdle | H |
| 34 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H |
| 35 | negbin_negbin | ✓ | BYM2 | RW1 | ✗ | H |
| 36 | negbin_negbin | ✓ | ICAR | AR1 | ✗ | H |
| 37 | negbin_negbin | ✓ | GP | RW1 | ✗ | A |
| 38 | negbin_negbin | ✓ | ICAR | ✗ | ZI | A |
| 39 | negbin_negbin | slopes | ICAR | ✗ | ✗ | A |
| 40 | negbin_negbin | ✓ | MSGP | RW1 | ✗ | A |
| 41 | binomial | ✗ | ✗ | ✗ | ✗ | H |
| 42 | binomial | ✓ | ✗ | ✗ | ✗ | H |
| 43 | binomial | slopes | ✗ | ✗ | ✗ | H |
| 44 | binomial | crossed | ✗ | ✗ | ✗ | A |
| 45 | binomial | ✓ | ICAR | ✗ | ✗ | H |
| 46 | binomial | ✓ | BYM2 | ✗ | ✗ | H |
| 47 | binomial | ✓ | GP | ✗ | ✗ | A |
| 48 | binomial | ✓ | MSGP | ✗ | ✗ | A |
| 49 | binomial | ✓ | ✗ | RW1 | ✗ | H |
| 50 | binomial | ✓ | ✗ | RW2 | ✗ | H |
| 51 | binomial | ✓ | ✗ | AR1 | ✗ | H |
| 52 | binomial | ✓ | ✗ | ✗ | ZI | A |
| 53 | binomial | ✓ | ✗ | ✗ | Hurdle | A |
| 54 | binomial | ✓ | ICAR | RW1 | ✗ | H |
| 55 | binomial | ✓ | BYM2 | RW1 | ✗ | H |
| 56 | binomial | ✓ | ICAR | AR1 | ✗ | H |
| 57 | binomial | ✓ | GP | RW1 | ✗ | A |
| 58 | binomial | ✓ | ICAR | ✗ | ZI | A |
| 59 | binomial | slopes | ICAR | ✗ | ✗ | A |
| 60 | binomial | ✓ | MSGP | RW1 | ✗ | A |

## Summary

| Grad | Count | % |
|:----:|------:|--:|
| H | 37 | 62% |
| A | 23 | 38% |
| **Total** | **60** |

**Note**: Random slopes (both `|` correlated and `||` uncorrelated syntax) now use hand-coded gradients.
