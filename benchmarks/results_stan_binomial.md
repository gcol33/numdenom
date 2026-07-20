| Model | tulpaRatio (s) | Stan (s) | tulpaRatio ESS/s | Stan ESS/s | Efficiency |
|-------|---------------:|---------:|-----------------:|-----------:|-----------:|
| Bin_base | 0.25 | 5.64 | 3991.6 | 279.2 | 14.3x |
| Bin_RE | 2.04 | 29.70 | 796.6 | 115.6 | 6.9x |
| Bin_ZI | 6.16 | 28.78 | 220.7 | 98.6 | 2.2x |
| Bin_ICAR_RW1 | 59.25 | 191.58 | 1.7 | 14.4 | 0.1x |

Excluded:

- `Bin_ICAR` -- draws carry no parameter names
- `Bin_BYM2` -- draws carry no parameter names
- `Bin_RW1` -- tulpaRatio not converged (Rhat 1.27, ESS 12)
