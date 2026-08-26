// hmc_temporal_multiscale.h
// Multi-scale temporal decomposition: trend + seasonal + short-term
// Builds on existing RW1/RW2/AR1 infrastructure
//
// Templated over the scalar type: double for the plain sampler, the autodiff
// types for gradient modes. One body per kernel, so the log-posterior the
// sampler evaluates and the one the autodiff paths differentiate cannot carry
// different terms.

#ifndef RATIOD_HMC_TEMPORAL_MULTISCALE_H
#define RATIOD_HMC_TEMPORAL_MULTISCALE_H

#include <vector>
#include <cmath>
#include <string>
#include "hmc_temporal.h"  // For TemporalType enum
#include "autodiff_utils.h"

#include <tulpa/soft_sum_to_zero.h>  // s2z_precision
#include <tulpa/sum_to_zero.h>       // rw1_rank / rw2_rank

namespace ratiod_temporal {

using ratiod::math::safe_log;
using ratiod::math::safe_sqrt;

// TemporalType enum is defined in hmc_temporal.h

// Multi-scale temporal data structure
struct MultiscaleTemporalData {
  int n_times = 0;                   // Number of unique time points
  int n_groups = 0;                  // Number of groups (for panel data)
  int n_obs = 0;                     // Total observations

  std::vector<int> time_index;       // Time index for each observation (1-based)
  std::vector<int> group_index;      // Group index for each observation (1-based)

  // Component specifications
  TemporalType trend_type = TemporalType::NONE;    // rw1, rw2, or none
  int seasonal_period = 0;                          // 0 if no seasonal, else period (e.g., 12)
  TemporalType short_term_type = TemporalType::NONE; // ar1, iid, or none

  bool shared = true;                // Shared between num/denom
};

// -----------------------------------------------------------------------------
// RW1 log-likelihood (intrinsic first-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW1: sum of (phi[t] - phi[t-1])^2 / (2*sigma2).
// `pin` adds the soft sum-to-zero term that identifies the constant null
// direction against the intercept; see multiscale_temporal_log_lik.
template <typename T>
inline T rw1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,
    bool cyclic = false,
    bool pin = false
) {
  int n = static_cast<int>(phi.size());
  if (n < 2) return T(0.0);

  T log_lik = T(0.0);

  // First differences
  for (int t = 1; t < n; t++) {
    T diff = phi[t] - phi[t - 1];
    log_lik = log_lik - T(0.5) * diff * diff / sigma2;
  }

  // Cyclic: add connection from last to first
  if (cyclic) {
    T diff = phi[0] - phi[n - 1];
    log_lik = log_lik - T(0.5) * diff * diff / sigma2;
  }

  // Normalizing constant (improper prior, omit for sampling)
  log_lik = log_lik
          - T(0.5 * tulpa::rw1_rank(n, cyclic)) * safe_log(T(2.0 * M_PI) * sigma2);

  if (pin) log_lik = log_lik + sum_to_zero_penalty(phi.data(), n);

  return log_lik;
}

// -----------------------------------------------------------------------------
// RW2 log-likelihood (intrinsic second-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2 / (2*sigma2)
template <typename T>
inline T rw2_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,
    bool cyclic = false,
    bool pin = false
) {
  int n = static_cast<int>(phi.size());
  if (n < 3) return T(0.0);

  T log_lik = T(0.0);

  // Second differences
  for (int t = 2; t < n; t++) {
    T diff2 = phi[t] - T(2.0) * phi[t - 1] + phi[t - 2];
    log_lik = log_lik - T(0.5) * diff2 * diff2 / sigma2;
  }

  // Cyclic: add wrap-around connections
  if (cyclic) {
    T diff2_1 = phi[0] - T(2.0) * phi[n - 1] + phi[n - 2];
    T diff2_2 = phi[1] - T(2.0) * phi[0] + phi[n - 1];
    log_lik = log_lik - T(0.5) * diff2_1 * diff2_1 / sigma2;
    log_lik = log_lik - T(0.5) * diff2_2 * diff2_2 / sigma2;
  }

  // Normalizing constant
  log_lik = log_lik
          - T(0.5 * tulpa::rw2_rank(n, cyclic)) * safe_log(T(2.0 * M_PI) * sigma2);

  // Only the constant direction is pinned. A non-cyclic RW2 also annihilates a
  // linear ramp, which the intercept does not absorb and a slope term does, so
  // that direction is left to the temporal covariate rather than pinned here.
  if (pin) log_lik = log_lik + sum_to_zero_penalty(phi.data(), n);

  return log_lik;
}

// -----------------------------------------------------------------------------
// AR1 log-likelihood (stationary first-order autoregressive)
// -----------------------------------------------------------------------------

// Log-likelihood for AR1: phi[t] = rho * phi[t-1] + epsilon[t]
template <typename T>
inline T ar1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,            // Innovation variance
    const T& rho                // Autocorrelation (-1 < rho < 1)
) {
  int n = static_cast<int>(phi.size());
  if (n < 2) return T(0.0);

  T log_lik = T(0.0);

  // Marginal distribution of first observation
  T marginal_var = sigma2 / ratiod_ar1::one_minus_rho2(rho);
  log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * marginal_var);
  log_lik = log_lik - T(0.5) * phi[0] * phi[0] / marginal_var;

  // Conditional distributions
  for (int t = 1; t < n; t++) {
    T resid = phi[t] - rho * phi[t - 1];
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * resid * resid / sigma2;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// IID log-likelihood (independent identically distributed)
// -----------------------------------------------------------------------------

template <typename T>
inline T iid_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2
) {
  int n = static_cast<int>(phi.size());
  T log_lik = T(0.0);

  for (int t = 0; t < n; t++) {
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * phi[t] * phi[t] / sigma2;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Multi-scale temporal log-likelihood
// -----------------------------------------------------------------------------

// Combined log-likelihood for trend + seasonal + short-term
template <typename T>
inline T multiscale_temporal_log_lik(
    const std::vector<T>& trend,       // Length n_times (or empty)
    const std::vector<T>& seasonal,    // Length seasonal_period (or empty)
    const std::vector<T>& short_term,  // Length n_times (or empty)
    const T& sigma2_trend,
    const T& sigma2_seasonal,
    const T& sigma2_short,
    const T& rho_short,                // Only used if short_term is AR1
    const MultiscaleTemporalData& temp_data
) {
  T log_lik = T(0.0);

  // Trend and seasonal are both intrinsic and both enter the SAME linear
  // predictor, so each carries a constant null direction that is unidentified
  // against the intercept and against the other component. Both are pinned.
  // The short-term arm is proper (AR1/IID), identifies its own level, and is
  // left alone.

  // Trend component
  if (temp_data.trend_type == TemporalType::RW1 && !trend.empty()) {
    log_lik = log_lik + rw1_log_lik(trend, sigma2_trend, false, true);
  } else if (temp_data.trend_type == TemporalType::RW2 && !trend.empty()) {
    log_lik = log_lik + rw2_log_lik(trend, sigma2_trend, false, true);
  }

  // Seasonal component (always cyclic RW1)
  if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
    log_lik = log_lik + rw1_log_lik(seasonal, sigma2_seasonal, true, true);
  }

  // Short-term component
  if (temp_data.short_term_type == TemporalType::AR1 && !short_term.empty()) {
    log_lik = log_lik + ar1_log_lik(short_term, sigma2_short, rho_short);
  } else if (temp_data.short_term_type == TemporalType::IID && !short_term.empty()) {
    log_lik = log_lik + iid_log_lik(short_term, sigma2_short);
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Compute total temporal effect at each observation
// -----------------------------------------------------------------------------

// eta_temporal[i] = trend[time_idx[i]] + seasonal[time_idx[i] % period] + short[time_idx[i]]
template <typename T>
inline void compute_temporal_eta(
    const std::vector<T>& trend,
    const std::vector<T>& seasonal,
    const std::vector<T>& short_term,
    const MultiscaleTemporalData& temp_data,
    std::vector<T>& eta_temporal  // Output: length n_obs
) {
  int N = temp_data.n_obs;
  eta_temporal.resize(N);

  for (int i = 0; i < N; i++) {
    T effect = T(0.0);
    int t_idx = temp_data.time_index[i] - 1;  // Convert to 0-based

    // Trend contribution
    if (!trend.empty() && t_idx >= 0 &&
        t_idx < static_cast<int>(trend.size())) {
      effect = effect + trend[t_idx];
    }

    // Seasonal contribution (wrap around using modulo)
    if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
      int s_idx = t_idx % temp_data.seasonal_period;
      if (s_idx >= 0 && s_idx < static_cast<int>(seasonal.size())) {
        effect = effect + seasonal[s_idx];
      }
    }

    // Short-term contribution
    if (!short_term.empty() && t_idx >= 0 &&
        t_idx < static_cast<int>(short_term.size())) {
      effect = effect + short_term[t_idx];
    }

    eta_temporal[i] = effect;
  }
}

// -----------------------------------------------------------------------------
// Priors for temporal hyperparameters
// -----------------------------------------------------------------------------

// PC prior for temporal variance (favor simpler models with smaller variance)
template <typename T>
inline T log_prior_sigma2_temporal_pc(const T& sigma2, double U, double alpha) {
  T rate = T(-std::log(alpha) / U);
  T sigma = safe_sqrt(sigma2);
  return safe_log(rate) - rate * sigma - safe_log(T(2.0) * sigma);
}

// The R-side name of a temporal structure, read into the enum every branch
// dispatches on. A name with no case here reads NONE, which leaves the field
// out of the density entirely, so every constructor's `type` string has to
// appear below.
inline TemporalType parse_temporal_type(const std::string& type_str) {
  if (type_str == "rw1") return TemporalType::RW1;
  if (type_str == "rw2") return TemporalType::RW2;
  if (type_str == "ar1") return TemporalType::AR1;
  if (type_str == "iid") return TemporalType::IID;
  if (type_str == "gp") return TemporalType::GP;
  if (type_str == "none") return TemporalType::NONE;
  return TemporalType::NONE;
}

} // namespace ratiod_temporal

#endif // RATIOD_HMC_TEMPORAL_MULTISCALE_H
