# GP Optimization Benchmarks

Tracking actual performance gains from speedup.md phases.

## Test Configuration
- **N:** 100 observations
- **nn:** 10 nearest neighbors (GP); 5 local + 10 regional (MSGP)
- **iter:** 300 (warmup: 100)
- **chains:** 1
- **Model:** binomial

## Current Results (2026-01-18)

| Model | Time (sec) | Samples/sec | Notes |
|-------|-----------|-------------|-------|
| Simple (no spatial) | 4.87 | 41.10 | Hand-coded gradients |
| GP | 18.33 | 10.91 | Autodiff with Phase 1 optimizations |
| MSGP | 22.32 | 8.96 | Hand-coded MSGP gradients |

## Historical Results (Phase 1 Development)

| Phase | Description | Time (sec) | Samples/sec | Speedup |
|-------|-------------|-----------|-------------|---------|
| Baseline | Before Phase 1.1 | - | - | - |
| 1.1 | Preallocate vectors | - | - | +25% (reported) |
| 1.2 | Replace std::pow | - | - | +5% (reported) |
| Pre-1.3 | Before cache | 74.08 | 2.70 | baseline |
| **1.3** | Cache neighbor distances | 72.96 | 2.74 | **+1.5%** |
| **Current** | All Phase 1 + hand-coded | 18.33 | 10.91 | **~4x vs Pre-1.3** |

## Notes

- Phase 1.3 speedup was lower than expected (1.5% vs 10-15%)
- Distance computation not a bottleneck; Cholesky dominates
- Current benchmarks show ~4x improvement vs Pre-1.3 baseline
- Hand-coded MSGP gradients provide ~25x speedup over autodiff for that model type
- Simple models with hand-coded gradients achieve ~41 samples/sec
