// ar1_shared.h
// The two quantities every AR1 correlation in the package reads: the floored
// 1 - rho^2, and the prior its sampled logit coordinate carries.
//
// Both exist once because both are read from more than one place that must
// agree. A density and its analytic gradient that floor 1 - rho^2 at different
// points, or with an added epsilon on one side and a clamp on the other, are
// orders of magnitude apart wherever the floor binds; a prior written out at
// the density and again at the gradient is a prior that can drift.

#ifndef RATIOD_AR1_SHARED_H
#define RATIOD_AR1_SHARED_H

#include "autodiff_utils.h"

namespace ratiod_ar1 {

// The one floor on 1 - rho^2, applied as a clamp.
constexpr double ONE_MINUS_RHO2_FLOOR = 1e-10;

template <typename T>
inline T one_minus_rho2(const T& rho) {
  return ratiod::math::safe_max(T(1.0) - rho * rho,
                                T(ONE_MINUS_RHO2_FLOOR));
}

// Beta(a, b) prior on u, expressed in the sampled coordinate logit(u) with the
// transform's log-Jacobian log(u) + log(1 - u) folded in:
//
//   log p(u) + log|du / dlogit(u)| = a log u + b log(1 - u) + const
//
// The package maps a correlation onto u two ways -- u = (rho + 1) / 2 for a rho
// on (-1, 1), u = rho for a rho on (0, 1) -- and each site passes its own u.
template <typename T>
inline T log_prior_logit_rho(const T& u, double a, double b) {
  return T(a) * ratiod::math::safe_log(u) +
         T(b) * ratiod::math::safe_log(T(1.0) - u);
}

// d/dlogit(u) of the above.
inline double log_prior_logit_rho_grad(double u, double a, double b) {
  return a * (1.0 - u) - b * u;
}

// AR1 correlation on (-1, 1): Beta(2, 2) on u = (rho + 1) / 2, the prior
// priors_default() documents for a temporal correlation.
constexpr double RHO_PRIOR_A = 2.0;
constexpr double RHO_PRIOR_B = 2.0;

// AR1 correlation on (0, 1), where u is rho itself: Uniform(0, 1).
constexpr double RHO_UNIT_PRIOR_A = 1.0;
constexpr double RHO_UNIT_PRIOR_B = 1.0;

}  // namespace ratiod_ar1

#endif  // RATIOD_AR1_SHARED_H
