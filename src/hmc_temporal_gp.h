// hmc_temporal_gp.h
// Gaussian Process temporal effects for irregularly-spaced time series
// Supports exponential, Matern, Gaussian, and periodic covariance functions

#ifndef RATIOD_HMC_TEMPORAL_GP_H
#define RATIOD_HMC_TEMPORAL_GP_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace ratiod_temporal_gp {

// Covariance function types for temporal GP
enum class TemporalCovType { EXPONENTIAL, MATERN, GAUSSIAN, PERIODIC };

// Temporal GP data structure
struct TemporalGPData {
  int n_obs = 0;                      // Number of unique time points (NOT total obs)
  int n_groups = 1;                   // Number of groups (for panel data)

  std::vector<double> time_values;    // Numeric time values (length n_obs)
  std::vector<int> group_index;       // Group index for each obs (1-based)

  TemporalCovType cov_type = TemporalCovType::EXPONENTIAL;
  double nu = 1.5;                    // Matern smoothness (if applicable)
  double period = 1.0;                // Period for periodic covariance
  bool shared = true;                 // Whether GP is shared between num/denom
};

// -----------------------------------------------------------------------------
// Temporal covariance functions
// -----------------------------------------------------------------------------

// Exponential covariance: sigma^2 * exp(-d / phi)
inline double temporal_cov_exponential(double d, double sigma2, double phi) {
  return sigma2 * std::exp(-d / phi);
}

// Matern 3/2 covariance: sigma^2 * (1 + sqrt(3)*d/phi) * exp(-sqrt(3)*d/phi)
inline double temporal_cov_matern32(double d, double sigma2, double phi) {
  double r = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + r) * std::exp(-r);
}

// Matern 5/2 covariance
inline double temporal_cov_matern52(double d, double sigma2, double phi) {
  double r = std::sqrt(5.0) * d / phi;
  return sigma2 * (1.0 + r + r * r / 3.0) * std::exp(-r);
}

// Gaussian (squared exponential) covariance: sigma^2 * exp(-(d/phi)^2)
inline double temporal_cov_gaussian(double d, double sigma2, double phi) {
  double r = d / phi;
  return sigma2 * std::exp(-r * r);
}

// Periodic covariance: sigma^2 * exp(-2 * sin^2(pi * d / period) / phi^2)
inline double temporal_cov_periodic(double d, double sigma2, double phi, double period) {
  double sin_term = std::sin(M_PI * d / period);
  return sigma2 * std::exp(-2.0 * sin_term * sin_term / (phi * phi));
}

// Generic covariance function dispatcher
inline double compute_temporal_cov(double d, double sigma2, double phi,
                                   TemporalCovType cov_type, double nu = 1.5,
                                   double period = 1.0) {
  switch (cov_type) {
    case TemporalCovType::EXPONENTIAL:
      return temporal_cov_exponential(d, sigma2, phi);
    case TemporalCovType::MATERN:
      if (nu <= 1.0) {
        return temporal_cov_exponential(d, sigma2, phi);  // nu=0.5 equivalent
      } else if (nu <= 2.0) {
        return temporal_cov_matern32(d, sigma2, phi);  // nu=1.5
      } else {
        return temporal_cov_matern52(d, sigma2, phi);  // nu=2.5
      }
    case TemporalCovType::GAUSSIAN:
      return temporal_cov_gaussian(d, sigma2, phi);
    case TemporalCovType::PERIODIC:
      return temporal_cov_periodic(d, sigma2, phi, period);
    default:
      return temporal_cov_exponential(d, sigma2, phi);
  }
}

// -----------------------------------------------------------------------------
// State-space representation for efficient O(n) inference
// (Only for exponential and Matern with half-integer nu)
// -----------------------------------------------------------------------------

// AR(1) representation for exponential covariance
// phi[t] = rho * phi[t-1] + epsilon, where rho = exp(-dt / range)
struct StateSpaceAR1 {
  double marginal_var;    // sigma^2
  double range;           // phi (range parameter)
};

// Compute AR(1) transition parameters given time gap dt
inline void ar1_transition(double dt, const StateSpaceAR1& ss,
                           double& rho, double& cond_var) {
  rho = std::exp(-dt / ss.range);
  cond_var = ss.marginal_var * (1.0 - rho * rho);
}

// Compute log-likelihood using state-space (O(n) algorithm)
// Assumes observations are sorted by time!
inline double temporal_gp_log_lik_statespace(
    const std::vector<double>& f,        // Temporal effect values
    const std::vector<double>& times,    // Time values (sorted)
    double sigma2,                        // Marginal variance
    double phi,                           // Range parameter
    TemporalCovType cov_type
) {
  int N = f.size();
  if (N == 0) return 0.0;

  // Only use state-space for exponential covariance
  if (cov_type != TemporalCovType::EXPONENTIAL) {
    // Fall back to direct computation for other covariance types
    return -1.0;  // Signal to use direct method
  }

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * f[0] * f[0] / sigma2;

  // Remaining observations: conditional on previous
  for (int i = 1; i < N; i++) {
    double dt = times[i] - times[i-1];

    // AR(1) transition
    double rho = std::exp(-dt / phi);
    double cond_var = sigma2 * (1.0 - rho * rho);

    // Ensure positive variance
    if (cond_var < 1e-10) cond_var = 1e-10;

    double cond_mean = rho * f[i-1];
    double resid = f[i] - cond_mean;

    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Direct GP log-likelihood (for general covariance)
// O(n^3) but works for any covariance function
// Use only for small n or when state-space not applicable
// -----------------------------------------------------------------------------

inline double temporal_gp_log_lik_direct(
    const std::vector<double>& f,        // Temporal effect values
    const std::vector<double>& times,    // Time values
    double sigma2,                        // Marginal variance
    double phi,                           // Range parameter
    TemporalCovType cov_type,
    double nu = 1.5,
    double period = 1.0
) {
  int N = f.size();
  if (N == 0) return 0.0;

  // Build covariance matrix
  std::vector<double> K(N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double d = std::abs(times[i] - times[j]);
      double cov = compute_temporal_cov(d, sigma2, phi, cov_type, nu, period);
      K[i * N + j] = cov;
      K[j * N + i] = cov;  // Symmetric
    }
    // Add small nugget for numerical stability
    K[i * N + i] += 1e-8;
  }

  // Cholesky decomposition: K = L * L^T
  std::vector<double> L(N * N, 0.0);
  for (int j = 0; j < N; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = K[j * N + k];
      for (int m = 0; m < k; m++) {
        sum -= L[j * N + m] * L[k * N + m];
      }
      if (j == k) {
        L[j * N + j] = std::sqrt(std::max(1e-10, sum));
      } else {
        L[j * N + k] = sum / L[k * N + k];
      }
    }
  }

  // Solve L * y = f (forward substitution)
  std::vector<double> y(N);
  for (int j = 0; j < N; j++) {
    double sum = f[j];
    for (int k = 0; k < j; k++) {
      sum -= L[j * N + k] * y[k];
    }
    y[j] = sum / L[j * N + j];
  }

  // Log-likelihood: -0.5 * (N * log(2*pi) + log|K| + f' K^{-1} f)
  // log|K| = 2 * sum(log(L_ii))
  // f' K^{-1} f = y' y
  double log_det = 0.0;
  for (int j = 0; j < N; j++) {
    log_det += std::log(L[j * N + j]);
  }
  log_det *= 2.0;

  double quad = 0.0;
  for (int j = 0; j < N; j++) {
    quad += y[j] * y[j];
  }

  return -0.5 * (N * std::log(2.0 * M_PI) + log_det + quad);
}

// -----------------------------------------------------------------------------
// Main GP log-likelihood dispatcher
// -----------------------------------------------------------------------------

inline double temporal_gp_log_lik(
    const std::vector<double>& f,
    const TemporalGPData& gp_data,
    double sigma2,
    double phi
) {
  int N = gp_data.n_obs;

  // Try state-space first (exponential covariance)
  if (gp_data.cov_type == TemporalCovType::EXPONENTIAL) {
    return temporal_gp_log_lik_statespace(f, gp_data.time_values,
                                           sigma2, phi, gp_data.cov_type);
  }

  // Fall back to direct computation
  return temporal_gp_log_lik_direct(f, gp_data.time_values, sigma2, phi,
                                     gp_data.cov_type, gp_data.nu, gp_data.period);
}

// -----------------------------------------------------------------------------
// Priors for GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for temporal variance (PC prior style)
inline double log_prior_temporal_sigma2_pc(double sigma2, double U, double alpha) {
  double rate = -std::log(alpha) / U;
  double sigma = std::sqrt(sigma2);
  return std::log(rate) - rate * sigma - std::log(2.0 * sigma);
}

// Log prior for temporal range (uniform or PC)
inline double log_prior_temporal_phi_uniform(double phi, double lower, double upper) {
  if (phi < lower || phi > upper) return -INFINITY;
  return -std::log(upper - lower);
}

// -----------------------------------------------------------------------------
// Gradient computation (for HMC)
// -----------------------------------------------------------------------------

// Numerical gradient of GP log-likelihood w.r.t. f (temporal effects)
inline void temporal_gp_gradient_f(
    const std::vector<double>& f,
    const TemporalGPData& gp_data,
    double sigma2,
    double phi,
    std::vector<double>& grad_f,
    double epsilon = 1e-6
) {
  int N = gp_data.n_obs;
  grad_f.resize(N);

  double base_ll = temporal_gp_log_lik(f, gp_data, sigma2, phi);

  std::vector<double> f_plus = f;
  for (int i = 0; i < N; i++) {
    f_plus[i] = f[i] + epsilon;
    double ll_plus = temporal_gp_log_lik(f_plus, gp_data, sigma2, phi);
    grad_f[i] = (ll_plus - base_ll) / epsilon;
    f_plus[i] = f[i];  // Reset
  }
}

// Parse covariance type from string
inline TemporalCovType parse_temporal_cov_type(const std::string& cov_str) {
  if (cov_str == "exponential") return TemporalCovType::EXPONENTIAL;
  if (cov_str == "matern") return TemporalCovType::MATERN;
  if (cov_str == "gaussian") return TemporalCovType::GAUSSIAN;
  if (cov_str == "periodic") return TemporalCovType::PERIODIC;
  return TemporalCovType::EXPONENTIAL;  // Default
}

} // namespace ratiod_temporal_gp

#endif // RATIOD_HMC_TEMPORAL_GP_H
