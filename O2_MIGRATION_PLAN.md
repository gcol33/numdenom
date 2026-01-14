# O2 Migration Plan

## Root Cause Analysis

The GP + RE segfault at -O2 stems from **three interrelated issues**:

### Issue 1: Rcpp Template Instantiation at Function Boundaries

The crash occurs in `RcppExports.cpp` during parameter marshalling, **before** the first line of `cpp_hmc_fit_gp` executes. The generated wrapper code:

```cpp
RcppExport SEXP _numdenom_cpp_hmc_fit_gp(...) {
BEGIN_RCPP
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type q_init(q_initSEXP);
    // ... 26 more parameter extractions
    rcpp_result_gen = Rcpp::wrap(cpp_hmc_fit_gp(...));
    return rcpp_result_gen;
END_RCPP
}
```

At -O2, GCC aggressively inlines `Rcpp::traits::input_parameter<T>::type` constructors. This creates **aliasing assumptions** where the compiler assumes the SEXP handles don't alias the R internals being constructed. When they do (via R's PROTECT mechanism), UB manifests.

### Issue 2: Missing Default Initialization

The `ModelData` struct has 70+ fields but no default constructor. Scalar fields like:
- `svc_sigma2_prior_scale`
- `svc_phi_prior_lower`
- `st_sigma2_prior_U`

...remain uninitialized (garbage) when not explicitly set. At -O1, the stack happens to contain zeros. At -O2, register allocation and dead store elimination expose the garbage values.

### Issue 3: Memory Barriers Don't Prevent Compile-Time Reordering

The code uses `std::atomic_thread_fence(std::memory_order_seq_cst)` which prevents **runtime** reordering but NOT **compile-time** reordering. The compiler can still:
- Move Rcpp extractions past the fence
- Eliminate "redundant" copies
- Assume values haven't changed between writes

---

## Migration Strategy

### Phase 1: Add Default Constructor to ModelData

Add a constructor that zero-initializes all scalar fields:

```cpp
struct ModelData {
  // ... existing fields ...

  // Default constructor: initialize ALL scalar fields
  ModelData() :
    p_num(0), p_denom(0),
    n_re_groups(0), n_re_terms(0), total_re_groups(0),
    has_re_slopes(false), has_re_correlated_slopes(false),
    total_re_params(0), total_sigma_params(0), total_chol_params(0),
    spatial_type(SpatialType::NONE), n_spatial_units(0), bym2_scale_factor(1.0),
    temporal_type(TemporalType::NONE), n_times(0), n_temporal_groups(0),
    n_temporal_params(0), temporal_cyclic(false), temporal_shared(false),
    tau_temporal_shape(1.0), tau_temporal_rate(0.01),
    zi_type(ZIType::NONE), p_zi(0), zi_prior_sd(2.5),
    has_svc(false), svc_sigma2_prior_scale(1.0),
    svc_phi_prior_lower(0.01), svc_phi_prior_upper(100.0),
    has_gp(false),
    gp_sigma2_prior_U(1.0), gp_sigma2_prior_alpha(0.01),
    gp_phi_prior_lower(0.01), gp_phi_prior_upper(100.0),
    has_multiscale_gp(false),
    ms_sigma2_local_prior_U(1.0), ms_sigma2_local_prior_alpha(0.01),
    ms_sigma2_regional_prior_U(1.0), ms_sigma2_regional_prior_alpha(0.01),
    has_multiscale_temporal(false),
    ms_sigma2_trend_prior_U(1.0), ms_sigma2_trend_prior_alpha(0.01),
    ms_sigma2_seasonal_prior_U(1.0), ms_sigma2_seasonal_prior_alpha(0.01),
    ms_sigma2_short_prior_U(1.0), ms_sigma2_short_prior_alpha(0.01),
    has_rsr(false), rsr_n(0),
    has_latent(false), latent_n_factors(0), latent_shared(false),
    latent_scale(false), latent_constraint(0), latent_sigma_prior_rate(1.0),
    has_spatiotemporal(false),
    st_sigma2_prior_U(1.0), st_sigma2_prior_alpha(0.01),
    st_phi_space_prior_lower(0.01), st_phi_space_prior_upper(100.0),
    st_phi_time_prior_lower(0.01), st_phi_time_prior_upper(100.0),
    N(0), sigma_beta(2.5), sigma_re_scale(2.5),
    phi_prior_shape(0.1), phi_prior_rate(0.1),
    tau_spatial_shape(1.0), tau_spatial_rate(0.01),
    model_type(ModelType::POISSON_GAMMA), n_threads(1)
  {}
};
```

### Phase 2: Consolidate Rcpp Interface to Single List Parameter

Instead of 27 individual typed parameters, pass a single `Rcpp::List`:

**Before (crash-prone at -O2):**
```cpp
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    // ... 24 more parameters
    bool verbose
);
```

**After (O2-safe):**
```cpp
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(Rcpp::List args);

// Internal implementation
Rcpp::List cpp_hmc_fit_gp(Rcpp::List args) {
    // Extract one at a time with explicit PROTECT
    Rcpp::NumericVector q_init = args["q_init"];
    Rcpp::IntegerVector y_num = args["y_num"];
    // ...
}
```

This reduces the number of template instantiations at the ABI boundary from 27 to 1.

### Phase 3: Use Compiler Attributes for Interface Functions

Mark the exported functions with optimization barriers:

```cpp
#ifdef __GNUC__
#define RCPP_INTERFACE __attribute__((optimize("O1")))
#else
#define RCPP_INTERFACE
#endif

// [[Rcpp::export]]
RCPP_INTERFACE
Rcpp::List cpp_hmc_fit_gp(Rcpp::List args);
```

Or use pragmas:

```cpp
#pragma GCC push_options
#pragma GCC optimize("O1")

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(...) { ... }

#pragma GCC pop_options
```

### Phase 4: Separate Interface Layer from Compute Layer

Create a thin interface layer that:
1. Extracts all Rcpp parameters
2. Copies to C++ types
3. Calls pure C++ compute functions

```
src/
├── interface/
│   ├── hmc_interface.cpp      # Rcpp exports, compiled with -O1
│   └── Makevars.interface     # Force -O1 for interface
├── compute/
│   ├── hmc_compute.cpp        # Pure C++ HMC, compiled with -O2
│   └── model_data.cpp         # ModelData setup
└── Makevars                   # Default -O2 with interface override
```

---

## Implementation Order

1. **Phase 1** - Low risk, immediate benefit
   - Add default constructor to ModelData
   - Add default constructors to nested structs (GPData, SVCData, etc.)
   - Test with -O2

2. **Phase 2** - Medium risk, significant refactor
   - Create `cpp_hmc_fit_gp_impl` with List parameter
   - Update R-side to construct single list
   - Deprecate old signature

3. **Phase 3** - Low risk, compiler-specific
   - Add attributes to interface functions
   - Test on GCC and Clang

4. **Phase 4** - High effort, long-term
   - Restructure source files
   - Separate Makevars rules
   - Full test suite validation

---

## Validation Checklist

Before merging O2 changes:

- [ ] GP only (no RE) - passes 100 iterations
- [ ] RE only (no GP) - passes 100 iterations
- [ ] GP + RE combined - passes 100 iterations
- [ ] Multi-chain (4 chains) - no crashes
- [ ] All 41 model configurations - gradient validation passes
- [ ] Valgrind/ASan clean (no memory errors)
- [ ] Benchmark shows expected speedup (~10-15%)

---

## Fallback

If O2 issues persist after all phases, the `-O1` global flag remains acceptable because:
1. Performance difference is 5-15% (not catastrophic)
2. HMC is bound by gradient computation, not Rcpp overhead
3. Correctness > speed for statistical software
