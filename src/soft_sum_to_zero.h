#ifndef ratiod_SOFT_SUM_TO_ZERO_H
#define ratiod_SOFT_SUM_TO_ZERO_H

namespace ratiod_constraints {

// Soft sum-to-zero identification prior for an intrinsic (rank-deficient)
// field of length n -- ICAR, BYM2, RW1/RW2, and the spatiotemporal and TVC
// interaction fields. Such a field has a free mean that is not separately
// identified from the intercept, so the two trade off along a ridge unless the
// mean is pinned.
//
// The prior is sum(phi) ~ Normal(0, (kappa * n)^2), following Morris et al.
// (2019), "Bayesian hierarchical spatial models: implementing the Besag York
// Mollie model in Stan", Spatial and Spatio-temporal Epidemiology 31:100301,
// which is also what brms emits for car(type = "icar").
//
// Returns the precision for the penalty  -0.5 * precision * (sum phi)^2.
// Note that kappa scales a STANDARD DEVIATION: the precision it implies,
// 1 / (kappa * n)^2, is many orders of magnitude larger than kappa itself.
inline double s2z_precision(int n, double kappa = 0.001) {
  const double sd = kappa * static_cast<double>(n);
  return 1.0 / (sd * sd);
}

} // namespace ratiod_constraints

#endif // ratiod_SOFT_SUM_TO_ZERO_H
