# Binomial-ratio likelihood: math reference for B1a

## Observation model

For each observation `i = 1, ..., N`:

- `n_i` = trial count (denominator, fixed integer; not modelled, no design matrix)
- `y_i` = success count (numerator, integer in `[0, n_i]`)
- `x_i` = covariate row, length `p_num` (intercept + predictors)
- `beta` = numerator regression coefficients, length `p_num`

Linear predictor (single process):

```
eta_i = x_i^T beta + offset_i        (offset is empty/0 in B1a)
```

Probability via inverse-logit (logit link):

```
p_i = sigmoid(eta_i) = 1 / (1 + exp(-eta_i))
```

Likelihood:

```
y_i | n_i ~ Binomial(n_i, p_i)
```

## Log-likelihood per observation

Constant terms (`lchoose(n, y)`) are kept for parity with the legacy implementation
but do not contribute gradient. Implemented in `tulpaRatio/src/autodiff_utils.h`:

```
log_lik_binomial(y, n, p) = y * log(p) + (n - y) * log(1 - p) + lchoose(n, y)
```

## Total log-posterior

Sum of per-obs log-likelihood plus all priors handled by the engine
(N(0, sigma_beta^2) on betas, plus any latent structure priors when present).
For B1a the only priors are the fixed-effect betas; no spatial/temporal/RE.

## Gradient

B1a uses reverse-mode AD (`A_r`, arena), so the gradient is provided by the
templated `LikelihoodFn<arena::Var>`. No `FullGradFn` override; no `ResidualFn`.
The closed-form gradient (for B2 H-kernel reference, not implemented here):

```
d log_lik_i / d eta_i = y_i - n_i * p_i
```

## n_processes

`n_processes = 1`. The numerator is the only modelled process; the denominator
is the trial count, *not* a separate regression target. (Confirmed by reading
`R/backend_hmc.R` lines 818-823: `X_denom` is forced to a 0-column matrix when
`model_type %in% c("binomial", "beta_binomial")`, and `log_post_impl.h` line
1041-1057 references only `eta_num` for the binomial branch.)

## Links

Only `logit` is used by `ratiod_binomial()`. The family struct allows
`link = "logit"`; other links are not exposed (`R/family.R` line 119). B1a
therefore registers only the logit branch in `lik_helpers.h`. Adding probit /
cloglog later is a constant-time addition to the helper file plus a new
`p = link_inv(eta)` call selected by `cfg.num_link_str`.

## Zero-inflation

Out of scope for B1a. The legacy code wraps the binomial log-lik in
`log_lik_zi_binomial`, `log_lik_oi_binomial`, `log_lik_zoib`,
`log_lik_hurdle_binomial` (autodiff_utils.h:417-496). B1a only registers the
plain `log_lik_binomial` path; the dispatcher errors out for ZI variants so
they fall through to the legacy backend via the feature-flag guard.

## Extra parameters

None for plain binomial. `n_extra_params = 0`. (Beta-binomial would add
`phi_num`, but that is out of scope.)
