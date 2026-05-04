# B1b — Response Payload Design

## Pattern

**Per-family POD struct, threaded through `ModelData::model_response_data` as
`void*`.**  Pattern 2 from the plan, matching `tulpaOcc/src/occu_fit.cpp`
(`OccResponseData`, `DynOccResponseData`, ...).

The bridge (`tulpa_bridge.cpp`) constructs the right struct based on
`cfg.family`, owns it for the duration of the NUTS call via a
`ResponsePayload` RAII wrapper, and reinterprets it inside each templated
likelihood callback.

## Why Pattern 2 over a tagged union

- **No junk fields.** `BinomialResponseData` carries integer counts only;
  `LognormalResponseData` carries doubles. A union would force every family
  to allocate fields it doesn't use.
- **Reads cleanly inside the templated likelihood.** Each `lik_<family>.cpp`
  does one `static_cast`, no enum-tag switch.
- **Append-only.** Adding a new family is one struct + one `if (family == "X")`
  branch in `build_response_payload`; existing structs are unchanged.

## Payload list

| Family            | Struct                       | Fields                                    |
|-------------------|------------------------------|-------------------------------------------|
| binomial          | `BinomialResponseData`       | `vector<int> y, n`                        |
| beta_binomial     | `BetaBinomialResponseData`   | `vector<int> y, n`                        |
| poisson_gamma     | `PoissonGammaResponseData`   | `vector<int> y_num`, `vector<double> y_denom_cont` |
| negbin_gamma      | `NegbinGammaResponseData`    | `vector<int> y_num`, `vector<double> y_denom_cont` |
| negbin_negbin     | `NegbinNegbinResponseData`   | `vector<int> y_num, y_denom`              |
| gamma_gamma       | `GammaGammaResponseData`     | `vector<double> y_num_cont, y_denom_cont` |
| lognormal         | `LognormalResponseData`      | `vector<double> y_num_cont, y_denom_cont` |

`binomial` and `beta_binomial` share the same shape but stay as separate
types — the dispatcher reads them in different families and separating types
prevents accidental cross-cast.

## Extra parameters

Indexed via `layout.extra_offset + k`, populated by `tulpa::compute_layout`
from `LikelihoodSpec::n_extra_params`. Per-family ordering matches the legacy
`q_init` block:

| Family          | Extras (order)                                  | n_extra_params |
|-----------------|--------------------------------------------------|----------------|
| binomial        | —                                                | 0              |
| beta_binomial   | `log_phi`                                        | 1              |
| poisson_gamma   | `log_shape` (Gamma denom)                        | 1              |
| negbin_gamma    | `log_phi_num` (NegBin), `log_phi_denom` (Gamma)  | 2              |
| negbin_negbin   | `log_phi_num`, `log_phi_denom`                   | 2              |
| gamma_gamma     | `log_shape_num`, `log_shape_denom`               | 2              |
| lognormal       | `log_sigma_num`, `log_sigma_denom`               | 2              |

## Priors on extra parameters

`extra_prior` is **not** registered.  Setting it would force tulpa's gradient
dispatcher to numerical gradients (`hmc_gradient_dispatch.h:36`).  Instead the
per-observation templated likelihood adds the prior contribution **once at
i==0**, so autodiff differentiates through it cleanly:

```cpp
if (i == 0) {
    ll = ll + lik::log_prior_gamma_log(log_phi_num, shape, rate);
    ll = ll + lik::log_prior_gamma_log(log_phi_denom, shape, rate);
}
```

This matches the legacy `log_prior_gamma` and `log_prior_half_cauchy` in
`autodiff_utils.h`.  Hyperpriors (`phi_prior_shape`, `phi_prior_rate`,
`sigma_prior_scale`) come in via `RatioConfig` and are stashed in a static
struct per builder call (single-chain B1b — a future spec-context pointer
generalises to multi-chain).

## Process count per family

Re-checked against the legacy `log_post_impl.h`:

- Single-process: `binomial`, `beta_binomial` (denominator is fixed trial count).
- **Two-process**: every count/continuous family uses both `eta_num` and
  `eta_denom`. Initial draft incorrectly modelled `poisson_gamma` and
  `negbin_gamma` as 1-process; corrected after reading the legacy branches in
  full.

## Bridge change

`tulpa_bridge.cpp::cpp_tulpaRatio_run_nuts_specs` now accepts both
integer and continuous response vectors plus `X_denom`, hyperpriors, and a
single `build_response_payload` switch.  ~150 lines net for the entire bridge
(was ~140 in B1a; +10 lines for the family switch and X_denom handling).
