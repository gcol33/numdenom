# GP Optimization Benchmarks

Tracking actual performance gains from speedup.md phases.

## Test Configuration
- **N:** 100 observations
- **nn:** 10 nearest neighbors (GP); 5 local + 10 regional (MSGP)
- **iter:** 300 (warmup: 100)
- **chains:** 1
- **Model:** binomial

## Current Results (2026-01-19)

| Model | Time (sec) | Samples/sec | Gradient Mode | Notes |
|-------|-----------|-------------|---------------|-------|
| Simple (no spatial) | 4.87 | 41.10 | H (hand-coded) | Fast |
| GP | 75.28 | 2.66 | A (autodiff) | **Bottleneck: tape overhead** |
| MSGP | 22.32 | 8.96 | H (hand-coded) | 3.4x faster than GP autodiff |

## Investigation Results (2026-01-19)

### Phase 1: Vector/Memory Optimizations
| Optimization | Expected | Actual | Notes |
|-------------|----------|--------|-------|
| 1.1 Preallocate vectors | +25% | 0% | Var constructor creates tape nodes regardless |
| 1.2 Replace pow | +5% | N/A | Already done via cached distances |
| 1.3 Cache distances | +10% | +1.5% | Minimal impact |

### Phase 2: Autodiff Infrastructure
| Optimization | Expected | Actual | Notes |
|-------------|----------|--------|-------|
| 2.1 Arena allocator | +25% | 0% | Var allocation not bottleneck |
| 2.2 Static dispatch | +35% | -40% | Made things SLOWER |

## Root Cause Analysis

The GP autodiff is slow because:
1. Every `Var(0.0)` creates a tape node, even constants
2. NNGP makes N × (Cholesky + solves) = thousands of tape nodes per gradient
3. Each Cholesky creates ~n² nodes for zeros that get overwritten
4. Backward pass iterates through all nodes with std::function calls

## Conclusion

**Autodiff infrastructure optimizations don't help GP models.**

The only effective solution is **hand-coded GP gradients** (Phase 5.1):
- MSGP already has hand-coded gradients: 22 sec vs 75 sec for autodiff
- Target: GP hand-coded gradients should achieve similar ~3x improvement
- Expected result: GP from 75 sec to ~25 sec

## Next Steps

1. Implement hand-coded NNGP gradients (`compute_gradient_gp_handcoded`)
2. Keep autodiff as reference for validation
3. Skip Phases 2-4, go directly to Phase 5
