# tulpaRatio 1.4.3

* **`temporal_tvc(structure = ...)`: `"iid"` unblocked, `"gp"` now errors
  instead of silently downgrading to RW1, and the Gibbs backend now rejects
  anything but `"rw1"` instead of silently ignoring the argument (#32).**
  `TemporalType::IID` had a complete log-prior and gradient in both the H and
  autodiff paths (`tvc_term_log_prior`, `tvc_prior_gradients_ws`, and the
  templated density via `hmc_tvc_autodiff.h`'s `using` declarations), and the
  C++ string parser already recognized `"iid"` -- but `temporal_tvc()`'s
  `match.arg` only offered `"rw1"/"rw2"/"ar1"/"gp"`, so no R call could ever
  reach it. `structure = "iid"` is now accepted, verified against
  `test_gradient_check.cpp`'s finite-difference harness in both H and A_r
  modes (0 deviation), and recovers a simulated intercept and slope in
  `test-recovery-constrained-fields.R`. Separately, `"gp"` was accepted and
  documented as a first-class option, but `TVCData` has no GP fields at all
  (no lengthscale, no covariance branch) -- the C++ parser silently mapped
  any unrecognized string, `"gp"` included, to RW1, so
  `temporal_tvc(..., structure = "gp")` ran to completion and silently fit an
  RW1 varying coefficient. `temporal_tvc()` now errors immediately if
  `structure = "gp"` is requested, rather than reaching the sampler at all.
  Independently, the Gibbs backend's TVC update
  (`# ---- 5. Update TVC coefficients via univariate MH with RW1 conditional
  proposal ----`) is hardcoded to RW1 and never read the `tvc_structure` field
  R was already sending it, so `mode = "gibbs"` silently ran RW1 regardless of
  what `structure` asked for; the Gibbs backend now rejects any TVC structure
  but `"rw1"` with a clear error pointing at `mode = "hmc"`. The C++ side's
  own duplicated string-to-enum parser in `hmc_sampler.cpp` was replaced with
  a call to the existing `ratiod_tvc::parse_tvc_structure()`, which had no
  callers before this.

* **`spatial_car(..., proper = TRUE)` now fits a proper CAR instead of
  silently downgrading to ICAR (#31).** The R object already computed rho
  bounds and printed "Proper CAR"; a complete C++ implementation
  (`hmc_car_proper.h`) already existed too. Neither was wired to the other --
  every backend collapsed `type = "car_proper"` to `"icar"` before it reached
  C++ (`# Handle all ICAR-type models (car, car_proper both use ICAR
  parameterization)`), so `rho` stayed fixed at 1 and the roxygen example's
  claim that `summary(fit_car)` shows a `rho_spatial` parameter was false. The
  HMC backend now threads `rho` through as its own parameter (`Q(rho) = D -
  rho*W`, `Beta(1,1)` prior by default, logit-transformed), added to
  `SpatialType`, `ParamLayout`, and the shared `spatial_gmrf_prior_grad`/
  `compute_log_post` machinery that `compute_gradient_composite` and every H
  gradient mode already read from. `phi`'s gradient is exact; `rho`'s
  log-determinant term has no cheap closed form (it needs `trace(Q^-1 W)`), so
  it is central-differenced on that one scalar and chain-ruled to logit scale,
  verified against `test_gradient_check.cpp`'s finite-difference harness (0
  deviation across all 29 parameters on the `car_proper` field). Proper CAR is
  full rank for `rho < 1`, so unlike ICAR/BYM2 it is not hard-centered.
  Autodiff gradient modes (`A`, `A_r`, `A_t`) refuse at the front door with a
  clear error rather than differentiate a density missing the field's prior
  entirely, since the log-determinant is not yet expressed in the templated
  density; `mode = "hmc"` with the default `gradient_mode = "auto"` (or `"H"`)
  is unaffected. Gibbs already rejected `car_proper` (its type whitelist never
  included it); Laplace, PG, VI, SGHMC, and ESS now reject it explicitly too,
  instead of silently fitting an ICAR model or -- for the three backends that
  call `compute_log_post_impl<double>` directly -- silently dropping the
  spatial prior's contribution to the posterior altogether. Fixing this also
  surfaced a second, independent bug in `predict()`'s areal-prediction path,
  which read `phi_spatial` starting one column early for any type it treated
  as "ICAR-shaped" once `car_proper` genuinely had an extra `rho` parameter
  in front of it.

* **A spatiotemporal Type IV interaction has a non-centered parameterization,
  which fixes chains freezing on data with little or no true interaction
  (#24).** `spatiotemporal(..., type = "IV")` reached `rhat` 1.14 at
  iter=1000/warmup=500 on data simulated with no space-time signal, and a 4x
  longer run moved that to 2.05 while every chain's own posterior sd shrank.
  The interaction's precision `tau` has no likelihood signal pulling it away
  from the boundary in that case, so its posterior drifts to very large
  values (tau ~ 4.3e8 was observed at warmup's end); under the centered
  parameterization the field's conditional scale shrinks as `1/sqrt(tau)`,
  so exploring both regions needs step sizes that differ by orders of
  magnitude, and NUTS answered by pinning the trajectory length at
  `max_treedepth` on essentially every iteration and collapsing the step size
  toward its floor -- the signature of a funnel between a variance component
  and the field it scales, not of a chain that merely needs more draws. A
  finite-difference check of `compute_gradient_spatiotemporal_handcoded`'s
  Type IV (Kronecker) block, absent until now, confirmed the analytic
  gradient was not the cause. `spatiotemporal(..., parameterization =
  "noncentered")` samples a tau-free `z` and reconstructs the interaction as
  `z / sqrt(tau)`, decoupling the two: `rhat` on the same repro drops to 1.00
  at iter=1000 and stays at 1.00 at iter=4000, with stable per-chain sd and
  a step size about 10x larger. The non-centered path already existed in the
  gradient code but was never reachable from R and was itself unchecked
  against finite differences; the warmup-end precision-informed mass-matrix
  warm-start specific to Type IV (which reads the raw parameter as the
  centered field directly) is now skipped under the non-centered
  parameterization, which it does not describe. `parameterization` still
  defaults to `"centered"`, since data that do carry real space-time
  structure move `tau` away from the boundary and may not need it.

* **The Gibbs BYM2 sampler no longer leaks the spatial field's level into the
  intercept unchecked (#15).** The per-site update held the structured field
  (`phi`) to sum zero by moving its mean into the intercept every sweep,
  unconditionally: no acceptance test weighed that move against the
  intercept's own prior. `dev_notes/gibbs_prior_recovery.R` (now
  `tests/testthat/test-gibbs-prior-recovery.R`) fits with zero binomial trials,
  where the likelihood is flat and the posterior must equal the prior exactly;
  under a `N(0, 0.5^2)` intercept prior the sampled sd came out at 1049 instead
  of 0.5, a ~2000x inflation. The move now competes for acceptance like every
  other update that touches the intercept, and gained a dedicated block-level
  move (mirroring the one the unstructured component already had) so the
  now-resisted level direction still mixes: BYM2 intercept ESS is 749 at
  iter=2000/warmup=1000 where it used to need the same bar ICAR clears at
  1000/500. The Gibbs priors already matched `compute_log_post`'s (Half-Cauchy
  `sigma_total`, Uniform(0,1) `rho`, Gamma `tau`, threaded through `fit_gibbs()`
  via `priors`) from prior work on this issue; this closes out the remaining
  checklist items, including removing the unused `pc_lambda` left over from an
  earlier tau update.

* **A per-fit `cores` budget can now reach within-chain work (#17).** For a
  single-chain HMC fit, `cores` used to be capped at `chains` before the user
  ever saw it (`min(chains, cpp_get_max_threads())`), so a one-chain fit always
  got exactly one thread for its gradient and likelihood loops regardless of
  machine size, and the regions that do carry a `num_threads()` clause
  (`compute_gradient_analytical`'s fallback path, the `linalg_fast.h` kernels)
  had no way to reach more than one. `cores` is now a genuine budget split two
  ways: `concurrent_chains = min(cores, chains)` chains run at once, each with
  `cores %/% concurrent_chains` threads of its own. Reaching that width when
  chains also run concurrently needed OpenMP nesting enabled
  (`omp_set_max_active_levels(2)`): a region inside a chain body is a nested
  one, and OpenMP pins nested teams to one thread by default. Most common model
  configurations (plain Poisson-gamma/binomial/negbin with simple random
  effects) take a vectorized Eigen gradient path that never reaches these OMP
  regions at all; the scalar fallback that does carry them serves zero-
  inflation, correlated random slopes, and GP/multiscale spatial models.

* **The two log posteriors are now the same function (#18).** `compute_log_post`
  and the templated `compute_log_post_impl<double>` that the `A_r` / `A` / `A_t`
  gradient modes differentiate returned different numbers on every model and
  omitted different structures. For those modes the runtime gradient check
  differences that same templated density, so a structure missing from it was
  invisible to both the gradient and its own numerical reference: they agreed on
  the same wrong value. Differencing the two densities against each other
  instead, over every structure the parameter layout allocates for, is what
  found the following.

  | structure | before | after |
  | --- | --- | --- |
  | HSGP spatial field | absent: all 11 parameters gradient exactly 0, worst relative deviation 1.00 | 1.3e-7 |
  | random slopes, uncorrelated | slope variance absent, 1.00 | 1.2e-9 |
  | random slopes, correlated | Cholesky factor and variance absent, 1.08 | 1.0e-8 |
  | crossed random intercepts | second term absent: 5 effects and its variance 0, 1.33 | 1.2e-9 |
  | TVC precision | a different prior on each side, 0.376 | 4.2e-10 |
  | collapsed ICAR / BYM2 | read `params[-1]`, access violation | refused |

  The residual constant that remained on every model was the binomial
  coefficient: the eta-form likelihood drops it as constant in eta, which is
  right for a Laplace inner solve and wrong for a log posterior that is reported
  and compared. `lp - lp_impl` was a flat -7156.27 across all fields and is now
  0 to floating-point noise.

* **A collapsed spatial field refuses an autodiff gradient mode instead of
  crashing.** The collapsed parameterizations marginalize the field out by
  locating its mode with Newton and adding a Laplace correction, which is not a
  closed-form function of the parameters, so the templated density cannot
  express them and the layout allocates no slot for the field. Reading that
  absent slot was an access violation. `gradient_mode` in `"A_r"`, `"A"`, `"A_t"`
  now errors at the front door naming the parameterization; `"H"` and `"N"` carry
  the analytic density and are unaffected. The same declaration covers the HSGP
  multi-scale GP and the non-centred GP, which are expressible but not written
  yet (#26).

* **The TVC precision takes the prior it is configured with.** `tvc_tau_shape`
  and `tvc_tau_rate` reach the sampler from `priors` and defaulted to
  `Gamma(2, 0.5)`, but two of the three implementations ignored them for a
  hard-coded PC prior with `P(sigma > 1) = 0.01`. The density, the specialized
  gradient and the composite gradient now all read the configured
  hyperparameters, through one derivative helper.

* **The random-effect prior is one implementation.** `hmc_re.h` computes the
  prior and the effective effects for every layout the sampler allocates -- one
  or more crossed terms, intercept-only or with slopes, correlated or not,
  centred or non-centred -- and both densities call it, so a term cannot be added
  to one and not the other. The TVC quadratic forms, per-term prior and eta
  assembly move the same way: templated in `hmc_tvc.h` and re-exported to
  `ratiod_tvc_ad`, replacing copies that had already drifted on the cyclic
  wrap-around edge. The RW1/RW2/AR1 templates take a pointer, so a caller holding
  a strided slice passes an offset rather than copying into a vector.

* `cpp_gradient_check()` reaches the structures it could not build before
  (`hsgp`, `tvc`, `re`, `re_crossed`, `re_slopes`, `re_slopes_corr`,
  `icar_collapsed`, `bym2_collapsed`), differences either density via
  `reference =`, and reports `log_post_impl` and `impl_gap` alongside the
  gradient. `test-gradient-correctness.R` asserts across every field that the two
  densities return the same number, that no block has an identically-zero
  gradient under any autodiff mode, and that a collapsed model refuses one.
  Its tolerance now allows an absolute `1e-6` before the relative comparison: a
  central difference of a log posterior of magnitude 1e3-1e4 resolves a gradient
  entry to about 1e-7, and the HSGP basis coefficients run to 6e-4, where 1e-4
  relative asked for more than the difference can deliver.

# tulpaRatio 1.4.2

* **Every intrinsic field is now identified at the engine's constant (#12).**
  `s2z_precision(n)` returns `1 / (kappa * n)^2` from `sd(sum phi) = kappa * n`,
  so callers pass a PRECISION. The spatiotemporal and TVC penalties took that
  constant as an argument and their call sites filled it with a bare `0.001` --
  which is `kappa` itself, weaker by ~2.5e6 at `T = 20`, pinning the field's sum
  at sd 31.6 instead of 0.02. Each penalty now derives the constant from the
  length of the sum it pins, so no call site can supply one. The interaction
  margins take their own constants rather than sharing one: a space margin sums
  `S` terms and a time margin `T`.

* **SVC keeps its own constant, deliberately.** Its penalty is rewritten on the
  sum rather than the mean -- the two forms are the same penalty, related by a
  factor of `n_obs` -- and both it and its gradient now read one
  `svc_sum_to_zero_ridge()`. That constant is NOT `s2z_precision(n_obs)`: an
  NNGP field's prior is proper, so its mean is identified and what is wanted is
  a ridge, not the pin a rank-deficient field needs. Substituting the intrinsic
  constant was tried and reverted -- at `N = 80` it is 156.25 against the
  ridge's 0.0125, and at that stiffness 4 chains x 1000 iterations return in 9
  seconds with per-chain posterior SD 0, Rhat 15.1 and the slope at 0.095
  against a truth of 0.3, where the ridge samples for minutes. Deriving the
  right constant for a proper field on its own terms is #25.

* **The multiscale trend and seasonal arms are pinned (#12).** Neither carried a
  sum-to-zero term on any path. Both are intrinsic and both enter the same
  linear predictor, so each had a constant null direction aliased with the
  intercept and with the other -- a two-dimensional ridge the fitted series
  hides, because only the level moves. Both are now pinned in the log posterior
  and in the analytic gradient; the proper short-term arm (AR1/IID) identifies
  its own level and is left alone. On a 500-observation binomial fit against a
  known intercept of 0.5, over 4 chains:

  | trend + seasonal | intercept | posterior SD | Rhat | slope (truth 0.3) |
  |---|---|---|---|---|
  | RW1 + cyclic seasonal, before | 2.6101 | 7.1840 | 1.589 | 0.2886 |
  | RW1 + cyclic seasonal, after | 0.5054 | 0.0170 | 1.012 | 0.2878 |
  | RW2, before | 3.7200 | 1.1098 | 2.163 | 0.2926 |
  | RW2, after | 0.4868 | 0.0167 | 1.000 | 0.2907 |

  The slope is right either way, which is why nothing downstream flagged it.

* **The spatiotemporal interaction's margins are identified (#12).** Same fit
  with a Knorr-Held Type IV interaction over 9 units x 5 times: the intercept
  posterior SD moves from 0.1106 to 0.0109 and Rhat from 3.43 to 1.14. It does
  not reach 1.05, and a 4x longer run makes it worse rather than better; that
  residual is tracked separately as #24.

* **The cyclic RW rank is no longer overcounted (#12).** RW1 and RW2 ranks were
  computed inline as `cyclic ? T : T-1` / `T-2` at 16 sites. A cycle-graph
  Laplacian still annihilates the constant, so a cyclic RW1 has rank `T-1`, and
  a cyclic RW2 `T-1` as well (a linear ramp is not periodic). Every site now
  calls `tulpa::rw1_rank()` / `rw2_rank()` from the engine, which cannot drift
  from the engine's own normalizers.

* **The spatiotemporal analytic gradient matches the penalty it differentiates.**
  Three further `0.001` literals sat in the ST gradient paths, including the one
  feeding the fused log-posterior reconstruction. With the old constant the
  resulting mismatch was too small to see; at the correct precision
  `test-gradient-correctness.R` fails on `icar_st` and `bym2_st` until the
  gradient carries the same per-margin constants. `hmc_sampler.h`'s
  `build_and_factorize()` keeps its `0.001`, renamed `diag_ridge`: it is a
  Tikhonov ridge `lambda*I` that makes a Kronecker of two intrinsic precisions
  factorize, not the rank-1-per-margin sum-to-zero penalty.

* **The multiscale, TVC and SVC sum-to-zero terms are written once.** The
  double and autodiff twins of the multiscale kernels, `tvc_sum_to_zero_penalty`
  and `svc_sum_to_zero_penalty` are collapsed into one body templated over the
  scalar type. The missing multiscale pin survived a previous migration pass
  precisely because it had to be found in six places; there is now one.
  `src/soft_sum_to_zero.h` is gone in favour of the engine's
  `tulpa/soft_sum_to_zero.h`.

* **Recovery tests cover the fields that were unconstrained.**
  `test-recovery-constrained-fields.R` gains multiscale (RW1 trend + cyclic
  seasonal, and RW2 trend), TVC and spatiotemporal cases. Shape and class
  assertions pass just as happily when a field's level is free, which is how
  all three items reached a release. The TVC and spatiotemporal cases assert
  recovery but not `Rhat`: neither model mixes to 1.05, which reproduces on the
  released build for TVC (#23) and is a separate residual for the interaction
  (#24). Both comments say so and name the issue to restore them under.

# tulpaRatio 1.4.1

* **The spec path's ICAR / BYM2 field is identified (#19).** The bridge that
  builds the engine's `ModelData` assigned the CSR adjacency without the
  component partition the sum-to-zero augmentation iterates over, so the field's
  constant direction carried no precision and its level random-walked:
  `mean(phi_spatial)` reached +200.9, -342.8 and +423.1 at seeds 42, 43 and 44
  where the legacy backend holds it within 0.02 of zero, and `tau_spatial` came
  out at 16.89 against the legacy 11.90. The field's shape and the fixed effects
  were unaffected, which is why it read as a healthy fit. The bridge now calls
  `ModelData::set_spatial_adjacency()`, which derives the partition and the
  component count from the adjacency (tulpa 0.0.108). The two B1d spec-vs-legacy
  parity tests run again, having been skipped since the defect was filed.

# tulpaRatio 1.4.0

* `tratio()` replaces `ratiod()` as the package's front door, joining the
  `tulpa*` family of fitting verbs alongside `tulpa::tulpa()` and
  `tulpaObs::tobs()`. The `ratiod_*` family constructors, the `ratiod_fit`
  class, and every other `ratiod_`-prefixed name are unchanged.

* `tratio()` carries statistical arguments in its signature and every perf,
  numerical, and tuning knob in a single `control = list()`, matching the
  convention the engine and tulpaObs already follow. Moved into `control`:
  `chains`, `iter`, `warmup`, `thin`, `cores`, `seed`, `verbose`,
  `adapt_delta`, `max_treedepth`, `metric`, `riemannian`, `gradient_mode`,
  `re_param`, `vi_variant`, and the stochastic-gradient knobs (`batch_size`,
  `epsilon`, `alpha`, `schedule_*`, `use_schedule`) that previously arrived
  through `...`. An unrecognised knob is now an error rather than a silent
  no-op, and a `warmup` that would leave no post-warmup draws is rejected.

* `refresh` is removed. It was a documented argument that no backend ever
  read, so passing it changed nothing.

## Bug fixes

* Spatiotemporal models fitted under an autodiff gradient mode now sample the
  density they are meant to (#14). The log posterior is written twice, once as
  `compute_log_post` and once as the templated `compute_log_post_impl` that the
  `arena`, `forward`, and `tape` modes differentiate, and the spatiotemporal
  interaction was present only in the first. The interaction parameters
  therefore carried no prior and contributed nothing to either linear
  predictor, so their gradient was exactly zero and every other parameter was
  differentiated against a linear predictor short one effect. The
  interaction now enters the templated log posterior as well, covering the
  Knorr-Held types I to IV, the separable and non-separable GP forms, the HSGP
  basis form, and the non-centred Type IV parameterization. The interaction
  math in `src/hmc_spatiotemporal.h` is templated in place rather than copied
  into a second autodiff header, so both paths call one implementation.
  `gradient_mode = "handcoded"` was correct throughout and is unchanged, to the
  last bit.

* The samplers no longer hold their per-thread scratch in the shape that
  corrupts the heap on Windows (#16). Every workspace in `hmc_sampler.cpp` was
  a `static thread_local` object owning heap buffers. Under the mingw toolchain
  such an object gets emutls-backed storage and a destructor registered through
  `__cxa_thread_atexit`, and the two are released by thread-exit hooks whose
  relative order is unspecified, so the destructor can free through storage that
  is already gone and the process dies at some later unrelated free. A worker
  thread exits only when libgomp narrows a team, which is why the fault appeared
  when a fit asked for fewer chains than the one before it. The 42 declarations
  now go through `RATIOD_TLS_WORKSPACE` (`src/tls_workspace.h`), which keeps the
  workspace behind a constant-initialized pointer: no initialization guard, no
  destructor registration, nothing for a dying worker to free. No object in the
  package registers a thread-exit destructor any more.

* `tratio()` no longer fails against a tulpa version the requirement allows. It
  calls `tulpa::tulpa_check_control()`, exported from tulpa 0.0.99, while
  DESCRIPTION asked only for `tulpa (>= 0.0.95)`.

# tulpaRatio 1.3.1

* `diagnostics()` replaces `mcmc_diagnostics()`, following the engine rename in
  tulpa 0.0.95. It delegates to `tulpa::diagnostics()`, which selects chain or
  approximation diagnostics from the fit's draws provenance rather than from the
  function's name. `mcmc_diagnostics()` is deprecated and still returns the same
  value.

## Bug fixes

* A fit no longer decides how many threads later fits get (#16). The Laplace
  and Polya-Gamma backends take a per-fit `cores`, which they applied by moving
  the process-wide OpenMP thread count and leaving it moved, and it defaults to
  one thread. Every region in the session that read that value afterwards was
  therefore sized by whichever of those fits ran last, in fits it knew nothing
  about. Each backend now restores the previous value when it returns, on the
  interrupt path as well as the normal one.

* Successive fits in one session no longer corrupt the heap on Windows (#8).
  The OpenMP team for chain-parallel work was sized per fit from `cores`, so a
  fit asking for fewer cores than the one before it shrank the team; libgomp
  then destroyed the surplus workers and ran their `thread_local` destructors
  while later work was still in flight, and the damage surfaced as
  `STATUS_HEAP_CORRUPTION` at an unrelated `free()`. Four chains followed by
  two was enough to trigger it. The team now only ever grows, sized by the
  widest core budget any fit has asked for, and `cores` bounds how many chains
  are in flight instead, so it keeps its meaning. Growing a team was always
  safe; only shrinking faulted. Its size never comes from
  `omp_get_max_threads()`: the Laplace and Polya-Gamma backends move that value
  through `omp_set_num_threads()` and leave it moved, so reading it let a
  Laplace fit ahead of the first chain fit pin the team at one thread and
  serialize the chains of every fit after it.

* The Gibbs spatial backend runs its chains in parallel under OpenMP (#7). It
  looped over chains in R, so a 4-chain fit cost close to four times a 1-chain
  fit; on a 40-site binomial ICAR model that ratio drops from 3.7x to 0.94x.
  `cores` now reaches this backend, bounded by the chain count and the machine.
  Chain seeds are still derived in R, so a fit with a given seed returns the
  same draws as before, and the parallel and serial paths agree bitwise.

* The handcoded gradient no longer disagrees with its own log posterior when a
  temporal effect is not shared between numerator and denominator. The
  vectorized and fused kernels added the temporal effect to the denominator's
  linear predictor and fed the denominator residual back into the temporal
  gradient, both unconditionally, while `compute_log_post()` applied the effect
  to the numerator alone. NUTS therefore sampled a density whose gradient
  pointed elsewhere, silently, for every family with a real denominator
  (`poisson_gamma`, `negbin_negbin`, `negbin_gamma`, `gamma_gamma`,
  `lognormal`). Binomial models were unaffected. The gradient checks now cover
  a continuous-denominator family with `shared = FALSE`, which is the only
  configuration that can see this.

* `ModelData`'s scalar members are initialised at declaration. The constructor
  set only `unique_id`, so roughly thirty members held indeterminate values
  until assigned; the GP entry point never assigned `re_parameterization` and
  read stack residue to choose between the centred and non-centred
  parameterisation.

* The scalar gradient fallback reduces its per-thread partial sums in
  thread-index order rather than in a critical section, whose summation order
  varies from run to run.

* `mcmc_diagnostics()` now reports per-parameter Rhat and ESS (#4). It handed a
  multi-parameter draws array to the single-variable `posterior::rhat()` /
  `ess_bulk()` / `ess_tail()`, collapsing every parameter to one scalar that was
  then recycled across the returned rows: a well-mixed four-chain fit reported
  `rhat = 2.124`, `ess_bulk = 1`. Diagnostics now delegate to
  `tulpa::mcmc_diagnostics()`, the engine's per-parameter implementation, which
  removes the local re-derivation (`compute_diagnostics_basic()`,
  `compute_split_rhat()`, `compute_ess_basic()`). `summary()`, `plot_rhat()`,
  `plot_ess()`, `diagnostic_summary()` and `check_diagnostics()` all inherit the
  correction, and a converged fit no longer prints a false non-convergence
  warning.

* `print()` on a `ratiod_diagnostic_summary` no longer errors when a fit has a
  parameter with Rhat > 1.01 or ESS < 400. The worst-Rhat and worst-ESS tables
  are column extracts of the diagnostics table and dispatched on
  `print.ratiod_diagnostics()`, which rounded columns the extract does not
  carry.

## Internal

* B1a PoC: plain binomial fits can route through tulpa's `LikelihoodSpec`
  path (`tulpa::get_nuts_fn()`) behind a feature flag
  (`options(tulpaRatio.use_specs = TRUE)` or `TULPARATIO_USE_SPECS=1`).
  Off by default; legacy backend remains the production path until full
  parity ships in B5. Posterior means agree with the legacy backend within
  Monte Carlo noise.

* B1b: the same feature flag now covers all 7 ratio families (binomial,
  poisson_gamma, negbin_gamma, negbin_negbin, gamma_gamma, lognormal,
  beta_binomial) for the simplest non-ZI / no-spatial / no-RE / single-chain
  configuration. Autodiff only — H-kernel port is deferred to B2.

* B1c: zero-inflation, hurdle, and one-inflation variants now route through
  the `LikelihoodSpec` path. Each family's templated likelihood dispatches on
  `data.zi_type` and calls a shared mixture helper in
  `src/lik_specs/lik_helpers.h`; per-family allowlist lives in
  `src/lik_dispatch.cpp` and `R/backend_hmc.R::SPEC_ZI_COMPAT`. Posterior
  parity with the legacy backend matches within Monte Carlo noise for every
  binomial × {ZI, hurdle, OI, ZOIB} combination; for count families
  (poisson_gamma, negbin_*) the legacy A_r path was already silently
  ignoring ZI, so the spec path here gives the mathematically correct
  posterior — recorded as a behaviour change, not a parity mismatch (see
  `dev_notes/B1c_zi_surface.md`). Autodiff only; ZI variants for the
  hand-coded H-kernel remain deferred.

* The tulpa ABI test no longer pins a hardcoded version number. It compared
  `cpp_tulpa_abi_version()` against a literal `32L`, a second copy of a
  constant tulpa already owns, so every tulpa ABI bump broke the test with a
  stale-literal failure that said nothing about compatibility. Two unguarded
  accessors (`cpp_tulpa_compiled_abi_version()`, the value baked in from the
  `LinkingTo` headers, and `cpp_tulpa_runtime_abi_version()`, the value in the
  loaded tulpa DLL) let the test assert the invariant that matters — that the
  two agree — and report both numbers when they do not. tulpa remains the
  single source of truth for the version.

* Documentation is regenerated under roxygen2 8.0.0, matching the rest of the
  packages that link against tulpa. `RoxygenNote` migrates to
  `Config/roxygen2/version`. Two external links now resolve to a topic alias
  rather than an Rd filename (`posterior::as_draws_df()`,
  `loo::stacking_weights()`), which clears the "Non-topic package-anchored
  link(s)" note from `R CMD check`; `INFERENCE_TIERS` drops the `\docType{data}`
  and `\format{}` entries roxygen2 8.0.0 no longer emits for documented values;
  and the package page lists authors alongside the maintainer.

# numdenom 1.3.0

## New Inference Backends

* **Tiered inference mode system**: Inference backends are now organized into
  three tiers based on epistemic guarantees:
  - Tier 1 (Exact): HMC, ESS, Polya-Gamma - diagnosable convergence
  - Tier 2 (Structured): Laplace - controlled approximation
  - Tier 3 (Optimized): VI, SGHMC, SGLD - no convergence guarantee (explicit opt-in)

* **Elliptical Slice Sampling (ESS)**: New Tier 1 backend for models with
  Gaussian priors. Use `mode = "ess"`.

* **Variational Inference (VI)**: New Tier 3 backend with mean-field, full-rank,
  and low-rank variants. Use `mode = "vi"`.

* **Stochastic Gradient MCMC**: New Tier 3 backends for very large data:
  - SGHMC: `mode = "sghmc"`
  - SGLD: `mode = "sgld"`

## Performance Improvements

* **Thread-safe autodiff**: A_t (tape autodiff) mode now uses RAII-based
  TapeScope for thread isolation. Parallel chains work correctly with all
  gradient modes.

* **Hand-coded gradients**: Added O(N) analytical gradients for:
  - GP spatial models (5.4x speedup)
  - Spatially varying coefficients (SVC)
  - Time-varying coefficients (TVC)
  - Correlated/uncorrelated random slopes
  - Crossed random effects
  - ZI/Hurdle binomial families
  - Latent factor models

* **Spatial prediction**: New `predict()` method for `ratiod_fit` objects
  supports prediction at new spatial locations.

## Internal Changes

* Forward-mode autodiff (`fwd::Dual`) for thread-safe gradient computation
* GPU backend stubs (preparation for future CUDA support)
* CRT-based negative binomial Polya-Gamma sampler

---

# numdenom 1.2.0

## New Features

### Spatiotemporal Interaction

* **Spatiotemporal interaction effects**: New `spatiotemporal()` function specifies
  space-time interactions using Knorr-Held (2000) Type I-IV models:
  - Type I: Unstructured (IID) interaction
  - Type II: Structured time at each location
  - Type III: Structured space at each time point
  - Type IV: Fully structured (Kronecker product)
  - Separable: For GP-based spatial/temporal effects

* **Non-separable spatiotemporal GP**: New `spatiotemporal_gp()` function for continuous
  space-time GP models with Gneiting or Cressie-Huang non-separable covariance.

* **Spatiotemporal effect extraction**: New `spatiotemporal_effects()` extracts posterior
  distributions in array, long, or summary format with visualization support.

### Zero-Inflation Variants

* **Zero-inflated binomial**: New `ratiod_zibinomial()` for proportions with excess zeros.

* **One-inflated binomial**: New `ratiod_oibinomial()` for proportions with excess ones
  (100% success/detection).

* **Zero-and-one inflated binomial (ZOIB)**: New `ratiod_zoibinomial()` for proportions
  with excess at both boundaries.

* **Hurdle binomial**: New `ratiod_hurdle_binomial()` for binomial data with hurdle
  at zero.

## Documentation

* New `vignette("spatial-temporal")` with examples of space-time modeling.
* New `vignette("zero-inflation")` covering ZI, OI, ZOIB, and hurdle binomial models.

---

# numdenom 1.1.0

## New Features

* **Latent factors for unmeasured confounders**: New `latent_factor()` function allows
  specifying latent factors that capture shared unmeasured variation between numerator
  and denominator processes. Latent factors are observation-level random effects with
  sum-to-zero constraints for identifiability. Use `latent = latent_factor(n_factors = 1)`
  in the `ratiod()` call.

* **Latent factor extraction**: New `latent_factors()` function extracts posterior
  summaries or full draws for latent factor scores from fitted models.

## Documentation

* Updated `vignette("random-effects")` with section on latent factors.

---

# numdenom 1.0.0

Initial release with:

* Three model families: `ratiod_negbin_negbin()`, `ratiod_binomial()`, `ratiod_poisson_gamma()`
* Native HMC/NUTS backend (no Stan dependency)
* Shared random effects (default) between numerator and denominator
* Spatial structure: CAR, BYM2, GP/NNGP, RSR
* Temporal structure: AR(1), RW1, cyclic
* Random slopes (correlated and uncorrelated)
* Nested and crossed random effects
* LOO/WAIC model comparison
* Zero-inflation support
