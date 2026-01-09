# Random Effects Implementation Plan

## Current State

| Feature | Parsing | HMC | Laplace | PG |
|---------|---------|-----|---------|-----|
| Random intercepts `(1\|group)` | ✅ | ✅ | ✅ | ✅ |
| Random slopes `(x\|group)` | ✅ | ❌ | ❌ | ❌ |
| Nested `(1\|site/plot)` | ⚠️ | ❌ | ❌ | ❌ |
| Crossed `(1\|A)+(1\|B)` | ✅ | ❌ | ❌ | ❌ |

**Problem**: Formulas are parsed correctly but slopes and multiple RE terms are silently ignored.

---

## Implementation Order

1. **Phase 1**: Multiple/Crossed RE `(1 | site) + (1 | year)` - foundational
2. **Phase 2**: Nested RE `(1 | site/plot)` - builds on Phase 1
3. **Phase 3**: Random slopes `(x | group)` - most complex

---

## Phase 1: Multiple/Crossed Random Effects

### 1.1 R Layer Changes

**formula.R:**
- `extract_random_effects()` already returns list - ensure all terms used
- Add validation for duplicate grouping variables

**standata.R:**
- Create `re_terms` list with per-term structure:
  ```r
  re_terms[[t]] <- list(
    group_var = "site",
    group_idx = integer vector,
    n_groups = integer
  )
  ```

**backend_hmc.R / backend_laplace.R / backend_pg.R:**
- Loop over all RE terms, not just `re_terms[[1]]`
- Pass `n_re_terms` and per-term structures to C++

**priors.R:**
- Support named vector: `sigma_re = c(site = 0.5, year = 0.3)`
- Or single prior applied to all terms

### 1.2 C++ Layer Changes

**hmc_sampler.h:**
```cpp
struct ModelData {
  // Multiple RE terms
  int n_re_terms;
  std::vector<int> re_n_groups;           // Groups per term
  std::vector<std::vector<int>> re_group; // [term][obs] -> group index
};

struct ParamLayout {
  std::vector<int> log_sigma_re_idx;  // One per RE term
  std::vector<int> re_start;          // Start index per term
  std::vector<int> re_end;            // End index per term
};
```

**hmc_sampler.cpp:**
- `compute_param_layout()`: Allocate space per term
- `compute_log_post()`: Sum RE contributions across terms
  ```cpp
  for (int t = 0; t < n_re_terms; t++) {
    int g = re_group[t][i] - 1;
    if (g >= 0) eta += re[re_start[t] + g];
  }
  ```

### 1.3 Tests

```r
test_that("crossed random effects work", {
  df$site <- factor(rep(1:10, each = 10))
  df$year <- factor(rep(1:5, 20))
  fit <- ratiod(y | n ~ x + (1 | site) + (1 | year), data = df)
  expect_true("sigma_site" %in% names(fit$summary))
  expect_true("sigma_year" %in% names(fit$summary))
})
```

---

## Phase 2: Nested Random Effects

### 2.1 R Layer Changes

**formula.R:**
- Parse `(1 | site/plot)` → expand to `(1 | site) + (1 | site:plot)`
- Create interaction variable `site:plot`

```r
expand_nested_re <- function(re_term, data) {
  if (grepl("/", re_term$group_var)) {
    groups <- strsplit(re_term$group_var, "/")[[1]]
    expanded <- list()
    for (i in seq_along(groups)) {
      var_name <- if (i == 1) groups[1] else paste(groups[1:i], collapse = ":")
      expanded[[i]] <- list(
        group_var = var_name,
        has_intercept = re_term$has_intercept,
        slope_vars = re_term$slope_vars
      )
    }
    return(expanded)
  }
  return(list(re_term))
}
```

**standata.R:**
- Create interaction columns: `data[[paste(g1, g2, sep = ":")]] <- interaction(data[[g1]], data[[g2]])`

### 2.2 C++ Layer

No changes needed - nested RE uses same C++ infrastructure as crossed RE.

### 2.3 Tests

```r
test_that("nested syntax expands correctly", {
  f <- ratiod_formula(y | n ~ (1 | site/plot), data = df)
  expect_equal(length(f$numerator$random_effects), 2)
  expect_equal(f$numerator$random_effects[[2]]$group_var, "site:plot")
})
```

---

## Phase 3: Random Slopes

### 3.1 R Layer Changes

**formula.R:**
- Already parses `slope_vars` - ensure passed through

**standata.R:**
- Build design matrix `Z_re[N, n_coefs * n_groups]`
- For `(1 + x | group)`: intercept column + x column per group

**priors.R:**
- Add `sigma_slopes` prior (vector)
- Add `cor_re` prior for correlation (LKJ default)
- Support uncorrelated syntax `(x || group)`

### 3.2 C++ Layer Changes

**hmc_sampler.h:**
```cpp
struct RETermData {
  int n_groups;
  int n_coefs;                    // 1 for intercept-only, 2+ with slopes
  std::vector<int> group_idx;     // obs -> group
  std::vector<double> Z_slopes;   // Slope design matrix (flattened)
  bool correlated;
};

struct ParamLayout {
  // Per-term: log_sigma[n_coefs], L_omega[(n_coefs*(n_coefs-1))/2], z[n_groups * n_coefs]
};
```

**hmc_sampler.cpp:**
- Non-centered parameterization: `u[g,j] = sigma[j] * sum_k(L[j,k] * z[g,k])`
- Linear predictor: `eta[i] += sum_j(Z[i,j] * u[group[i],j])`
- LKJ prior on correlation matrix

### 3.3 Tests

```r
test_that("random slopes recover true values", {
  df <- sim_random_slopes(sigma_int = 0.5, sigma_slope = 0.3, cor = 0.4)
  fit <- ratiod(y | n ~ x + (1 + x | group), data = df)
  expect_equal(fit$sigma_int, 0.5, tolerance = 0.2)
  expect_equal(fit$sigma_slope, 0.3, tolerance = 0.2)
})
```

---

## Warning System (Interim)

Until full implementation, add warnings for unsupported features:

```r
warn_unsupported_re <- function(re_terms) {
  for (term in re_terms) {
    if (length(term$slope_vars) > 0) {
      warning("Random slopes not yet supported - slopes will be ignored: ",
              paste(term$slope_vars, collapse = ", "))
    }
  }
  if (length(re_terms) > 1) {
    warning("Multiple random effects not yet supported - only first term will be used")
  }
}
```

---

## File Checklist

### Phase 1 (Crossed RE)
- [ ] R/formula.R - validate no duplicate group vars
- [ ] R/standata.R - build multi-term RE structure
- [ ] R/backend_hmc.R - pass all RE terms to C++
- [ ] R/backend_laplace.R - pass all RE terms to C++
- [ ] R/backend_pg.R - pass all RE terms to C++
- [ ] R/priors.R - named sigma_re support
- [ ] src/hmc_sampler.h - multi-term data structures
- [ ] src/hmc_sampler.cpp - multi-term log_post
- [ ] src/laplace_core.cpp - multi-term support
- [ ] src/pg_binomial.cpp - multi-term support
- [ ] tests/testthat/test-re-crossed.R

### Phase 2 (Nested RE)
- [ ] R/formula.R - expand nested syntax
- [ ] R/standata.R - create interaction columns
- [ ] tests/testthat/test-re-nested.R

### Phase 3 (Random Slopes)
- [ ] R/formula.R - pass slope_vars through
- [ ] R/standata.R - build Z_slopes matrix
- [ ] R/priors.R - sigma_slopes, cor_re priors
- [ ] src/hmc_sampler.h - slope data structures
- [ ] src/hmc_sampler.cpp - slope contribution to eta
- [ ] src/hmc_re.h - new header for RE utilities
- [ ] tests/testthat/test-re-slopes.R
