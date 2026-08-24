// hmc_tvc_autodiff.h
// Templated TVC (Temporally-Varying Coefficients) functions
// Works with both double and ad::Var for automatic differentiation

#ifndef RATIOD_HMC_TVC_AUTODIFF_H
#define RATIOD_HMC_TVC_AUTODIFF_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_tvc.h"  // For TVCData and TemporalType

namespace ratiod_tvc_ad {

using ratiod_tvc::TVCData;
using ratiod_temporal::TemporalType;
using namespace ratiod::math;

// The TVC quadratic forms, per-term prior, full prior and eta assembly are
// templated in hmc_tvc.h, next to the TVCData they read, and named here so the
// autodiff density keeps reading them under this namespace. One implementation
// serves both densities: the copies that used to live here had drifted from
// hmc_tvc.h on the cyclic wrap-around edge.
using ratiod_temporal::rw1_quadratic_form_t;
using ratiod_temporal::rw2_quadratic_form_t;
using ratiod_temporal::ar1_log_density_t;
using ratiod_tvc::tvc_term_log_prior;
using ratiod_tvc::tvc_log_prior;
using ratiod_tvc::compute_tvc_eta;


// The sum-to-zero constraint is ratiod_tvc::tvc_sum_to_zero_penalty in
// hmc_tvc.h, templated over the scalar type and shared with the plain sampler.

// =============================================================================
// TVC hyperparameter priors
// =============================================================================

// Half-Cauchy prior on tau (precision)
// Actually this is on sigma = 1/sqrt(tau), then transformed
template<typename T>
T log_prior_tau_half_cauchy(const T& log_tau, double scale) {
    T tau = safe_exp(log_tau);
    T sigma = T(1.0) / safe_sqrt(tau);
    // Half-Cauchy: 2 / (pi * scale * (1 + (sigma/scale)^2))
    // log form + Jacobian for log_tau -> tau -> sigma
    return T(std::log(2.0 / M_PI / scale))
           - safe_log(T(1.0) + sigma * sigma / T(scale * scale))
           - T(0.5) * log_tau;  // Jacobian: d(sigma)/d(tau) * d(tau)/d(log_tau)
}

// Gamma prior on tau
template<typename T>
T log_prior_tau_gamma(const T& log_tau, double shape, double rate) {
    T tau = safe_exp(log_tau);
    // Gamma(shape, rate): tau^{shape-1} * exp(-rate*tau) * rate^shape / Gamma(shape)
    // + Jacobian for log transform
    return T(shape - 1.0) * log_tau - rate * tau + log_tau;
}

} // namespace ratiod_tvc_ad

#endif // RATIOD_HMC_TVC_AUTODIFF_H
