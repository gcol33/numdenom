# B1d-1 design constraints (derived from tulpaGlmm audit)

Authored 2026-05-04. The previous B1c agent leaked a half-finished B1d-1
attempt (now stashed, ignored). This note captures the design spec the
next agent must follow so tulpaRatio's RE surface is consistent with
tulpaGlmm's already-shipped implementation. **Both packages plug into
the same tulpa engine; they must use the same RE layout.**

## Source of truth: tulpa exposes RE infrastructure publicly

`tulpa/inst/include/tulpa/model_data.h` already carries the full multi-term
RE surface. Used today by tulpaGlmm. Fields:

- `n_re_terms` (int)
- `re_parameterization` (1 = non-centered)
- `re_group_multi_flat` (vector<int>, size `N * n_terms`, group index per obs per term, 1-based)
- `re_n_groups_multi[t]` (int per term)
- `re_offsets[t]` (cumulative group offset for indexing into the flat u vector)
- `re_n_coefs[t]` (int per term: 1 = intercept-only; 1 + n_slopes if slopes present)
- `re_n_slopes[t]` (int per term)
- `re_correlated[t]` (bool per term: LKJ-Cholesky path)
- `re_n_chol[t]` (int per term: `k(k-1)/2` if correlated && k > 1, else 0)
- `re_slope_matrices[t]` (row-major flatten `[i * n_slopes + s]`)
- `total_re_groups`, `total_re_params`, `total_sigma_params`, `total_chol_params`
- `has_re_slopes`, `has_re_correlated_slopes` (bool)
- `sharing.re[k]` (which process k receives the RE contribution)
- `sigma_re_scale` (half-Normal scale hyperprior on σ_re; default 1.0)

Plus a legacy single-term mirror (`re_group`, `n_re_groups`) populated
when `n_terms == 1 && n_coefs[0] == 1`. **Do not write to those legacy
fields directly** — populate the multi-term arrays and let
`compute_param_layout` produce the parameter layout.

## Param layout (already produced by tulpa, no tulpaRatio override)

`tulpa::ParamLayout` (from `inst/include/tulpa/param_layout.h`) already
emits the canonical RE layout when given a `ModelData` with the multi-term
RE arrays set. Slot names from `glmm_fit.cpp:300-333`:

- `log_sigma_re_multi[t]` (or `log_sigma_re_slopes[t][c]` per coefficient if slopes)
- `chol_re_start_multi[t]..chol_re_end_multi[t]` (correlated slopes only)
- `re_start_multi[t]..re_end_multi[t]` for `n_groups[t] * n_coefs[t]` deviations,
  group-major then coef

**tulpaRatio MUST use this same layout — do not invent a new one.**
The previous B1c agent's stashed work added q_init reorder logic; that
is the wrong move. The legacy tulpaRatio backend already produces this
exact layout (it's the same underlying engine). Verify by reading
`tulpaRatio/src/log_post_impl.h` and the `compute_param_layout` call —
the layout slots match `ParamLayout` 1:1.

## R-side spec: copy tulpaGlmm's `re_spec` shape

tulpaGlmm threads RE from R as a list with these keys (from
`populate_helpers.h:14-25`):

```r
re_spec <- list(
  n_terms        = <int>,
  groups         = list(<integer N>, ...),  # 1-based group index per obs per term
  n_groups       = <integer[n_terms]>,
  n_coefs        = <integer[n_terms]>,      # 1 = intercept-only; 1 + n_slopes
  correlated     = <logical[n_terms]>,      # TRUE => LKJ-Cholesky on joint cov
  has_intercept  = <logical[n_terms]>,
  slope_matrices = list(<NumericMatrix N x n_slopes>, ...),
  shared         = <logical[n_processes]>,  # which η_k gets RE
  sigma_re_scale = <double, default 1.0>
)
```

**Adopt this exact contract for tulpaRatio's bridge.** It will be
identical work in tulpaGlmm and tulpaRatio.

## Lift the C++ helper into tulpa, don't copy it

`tulpaGlmm::populate_re()` (in `tulpaGlmm/src/populate_helpers.h`, ~120 LOC)
is the canonical implementation. Two options:

1. **Move it into tulpa as a public header** (`tulpa/inst/include/tulpa/populate_re.h`)
   so tulpaRatio and any future model package use it directly.
   This is the cleanest, "always clean, always scaling" move per CLAUDE.md
   ("if shared logic exists across packages, extract it"). Requires a
   tulpa-side commit.

2. **Vendor a copy into tulpaRatio** — quicker for B1d-1, but creates
   exactly the duplication CLAUDE.md forbids ("never copy-paste across
   specialized variants"). NOT recommended.

**Recommend option 1.** Tiny tulpa commit moves `populate_re()` from
tulpaGlmm into a public header; tulpaGlmm switches to including it
from tulpa; tulpaRatio includes it for free. ABI doesn't change
(it's an inline helper, not a struct layout change).

## Scope decision: B1d-1 doesn't need to be conservative

The earlier scope ("single grouping factor only") was set before
auditing tulpaGlmm. Since tulpa's engine already supports
multi-term + correlated slopes, and tulpaGlmm exercises that path,
**tulpaRatio's spec path can support the full surface in one shot.**

Recommended B1d-1 allowlist:

- single-term intercept-only — yes
- single-term intercept + slopes (uncorrelated) — yes
- single-term intercept + slopes (LKJ-correlated) — yes
- multi-term — yes (tulpaGlmm proves the engine handles it)
- nested / crossed grouping factors — NO (separate phase; their R-side
  representation is more involved and orthogonal to the layout)

Net effect: B1d-1 covers all the legacy RE configurations except
nested/crossed, which is a parsing concern not an engine concern.

## H-kernel RE prior gradient

The B2 H-kernel in tulpaRatio (`src/lik_grad_h_kernel.cpp`) currently
does not include RE prior gradients (B2 was non-RE only). The engine's
generic gradient path handles RE prior gradients natively when the
spec uses autodiff or composes via `extra_prior_arena`/`gradient_fn`.

For B1d-1: ship autodiff-only RE in the spec path (`gradient_mode = "A_r"`),
gate H-mode away when RE is present. Document this in the report.
H-kernel RE comes in a follow-up that mirrors tulpaGlmm's
`glmm_laplace.cpp` derivative work (`R/laplace_derivs.R` in tulpaGlmm
has the closed-form derivatives — read that before porting).

## Hard rules for the next agent

1. **Do NOT invent new RE fields in `RatioConfig` or `ModelData`.** Use
   tulpa's existing fields. If something is missing, that's a tulpa
   change — surface it, propose a tulpa-side commit, do not patch it
   into tulpaRatio.

2. **Do NOT translate or reorder `q_init`.** The legacy tulpaRatio
   backend produces the canonical layout; the spec path consumes the
   same layout. Any reorder logic is wrong.

3. **Read tulpaGlmm in full first**: `populate_helpers.h`, `glmm_fit.cpp`
   (the `cpp_glmm_fit` function), one `family_*.h` file (e.g.
   `family_poisson.h`) to see how the family LikelihoodFn integrates
   with RE deviations.

4. **The stashed B1c-agent work (`stash@{0}`) is wrong-headed by design.**
   Do not pop it.
