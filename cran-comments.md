## R CMD check results

0 errors | 0 warnings | 2 notes

* This is a new submission.

* NOTE: "New submission" - expected for first CRAN submission.

* NOTE: "unable to verify current time" - network/timestamp issue during check, not a package problem.

## Test environments

* local Windows 11, R 4.5.2
* win-builder (devel and release) - pending
* R-hub (ubuntu-gcc, fedora-clang, macos-arm64) - pending

## Package Description

ratiod is an opinionated framework for Bayesian hierarchical modelling of ratios, rates, and proportions. The core premise is that ratios are derived quantities computed from latent processes, not direct data. The package provides:

- Three inference backends: HMC/NUTS (default), Laplace approximation, Polya-Gamma Gibbs
- Spatial models: CAR, BYM2, GP (NNGP), RSR
- Temporal models: AR1, RW1, RW2, multiscale decomposition
- Full uncertainty propagation to ratio posteriors

## Downstream dependencies

There are currently no downstream dependencies for this package.
