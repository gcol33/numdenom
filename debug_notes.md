# Debug Notes

## 2026-01-14: Spatial HMC Segfault

### Symptom
- Segmentation fault during HMC sampling when `spatial = spatial_car(...)` was used
- Crash occurred right after "Spatial: icar (9 units)" message
- Non-spatial models worked fine

### Root Cause
**R/C++ indexing mismatch in CSR adjacency structure**

In `R/backend_hmc.R` line 735:
```r
# BEFORE (buggy):
adj_col_idx <- c(adj_col_idx, neighbors)  # R 1-based indices
```

The `which()` function returns 1-based indices, but C++ expects 0-based:
```cpp
// In src/log_post_impl.h line 190:
int j = data.adj_col_idx[k];
phi_spatial[j]  // Out of bounds access when j=9 but phi_spatial.size()=9
```

### Fix
```r
# AFTER (fixed):
adj_col_idx <- c(adj_col_idx, neighbors - 1L)  # Convert to 0-based for C++
```

### Verification
All spatial models now work and are faster than Stan:
- poisson_gamma + ICAR: 14.6x
- negbin_negbin + ICAR: 4.9x
- binomial + ICAR: 13.5x

### Lessons Learned
1. Always verify R→C++ index conversions (1-based vs 0-based)
2. CSR format column indices must match array bounds in C++
3. Add bounds checking in debug builds for spatial indexing
