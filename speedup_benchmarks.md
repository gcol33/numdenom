# GP Optimization Benchmarks

Tracking actual performance gains from speedup.md phases.

## Test Configuration
- **N:** 100 observations
- **nn:** 10 nearest neighbors
- **iter:** 300 (warmup: 100)
- **chains:** 1
- **Model:** binomial GP

## Results

| Phase | Description | Time (sec) | Samples/sec | Speedup |
|-------|-------------|-----------|-------------|---------|
| Baseline | Before Phase 1.1 | - | - | - |
| 1.1 | Preallocate vectors | - | - | +25% (reported) |
| 1.2 | Replace std::pow | - | - | +5% (reported) |
| Pre-1.3 | Before cache | 74.08 | 2.70 | baseline |
| **1.3** | Cache neighbor distances | 72.96 | 2.74 | **+1.5%** |

## Notes

- Phase 1.3 speedup was lower than expected (1.5% vs 10-15%)
- Distance computation not a bottleneck; Cholesky dominates
- Cumulative improvement from 1.1+1.2+1.3 needs fresh baseline measurement
