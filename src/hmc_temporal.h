// hmc_temporal.h
// Temporal random effects support for HMC backend
// Supports RW1, RW2, and AR1 temporal structures

#ifndef QUOTR_HMC_TEMPORAL_H
#define QUOTR_HMC_TEMPORAL_H

#include <vector>
#include <cmath>

namespace ratiod_temporal {

// =====================================================================
// Temporal structure types
// =====================================================================

enum class TemporalType { NONE, RW1, RW2, AR1, IID };

// Temporal data container
struct TemporalData {
  TemporalType type;
  std::vector<int> time_index;   // Maps obs to time point (1-based, 0 = no temporal)
  std::vector<int> group_index;  // Maps obs to temporal group (1-based, for panel data)
  int n_times;                   // Number of time points
  int n_groups;                  // Number of groups (1 if no grouping)
  int n_temporal_params;         // Total temporal parameters (n_times * n_groups)
  bool cyclic;                   // Whether RW is cyclic
  bool shared;                   // Whether effect is shared between num/denom

  // Prior parameters
  double tau_temporal_shape;     // Gamma shape for precision
  double tau_temporal_rate;      // Gamma rate for precision
};

// =====================================================================
// RW1 precision matrix quadratic form
// =====================================================================

// Compute phi' Q_RW1 phi for RW1 prior
// Q_RW1 is the first-order random walk precision matrix
inline double rw1_quadratic_form(
    const double* phi,
    int T,
    bool cyclic
) {
  double quad = 0.0;

  // Sum of squared differences
  for (int t = 0; t < T - 1; t++) {
    double diff = phi[t + 1] - phi[t];
    quad += diff * diff;
  }

  // Cyclic: add edge from T to 1
  if (cyclic) {
    double diff = phi[0] - phi[T - 1];
    quad += diff * diff;
  }

  return quad;
}

// =====================================================================
// RW2 precision matrix quadratic form
// =====================================================================

// Compute phi' Q_RW2 phi for RW2 prior
// Q_RW2 penalizes second differences (curvature)
inline double rw2_quadratic_form(
    const double* phi,
    int T,
    bool cyclic
) {
  if (T < 3) return 0.0;

  double quad = 0.0;

  // Sum of squared second differences
  for (int t = 0; t < T - 2; t++) {
    double diff2 = phi[t] - 2.0 * phi[t + 1] + phi[t + 2];
    quad += diff2 * diff2;
  }

  // Cyclic: wrap around
  if (cyclic && T >= 3) {
    // t = T-2
    double diff2_1 = phi[T - 2] - 2.0 * phi[T - 1] + phi[0];
    quad += diff2_1 * diff2_1;
    // t = T-1
    double diff2_2 = phi[T - 1] - 2.0 * phi[0] + phi[1];
    quad += diff2_2 * diff2_2;
  }

  return quad;
}

// =====================================================================
// AR1 log-density
// =====================================================================

// Compute log-density for AR1 process
// phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
// Marginal variance: sigma^2 / (1 - rho^2)
inline double ar1_log_density(
    const double* phi,
    int T,
    double rho,
    double tau  // precision = 1/sigma^2
) {
  if (T < 2) return 0.0;

  double log_dens = 0.0;

  // First observation: phi[0] ~ N(0, sigma^2 / (1 - rho^2))
  double marginal_var = 1.0 / (tau * (1.0 - rho * rho));
  log_dens -= 0.5 * phi[0] * phi[0] / marginal_var;
  log_dens -= 0.5 * std::log(2.0 * M_PI * marginal_var);

  // Conditional: phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
  double sigma2 = 1.0 / tau;
  for (int t = 1; t < T; t++) {
    double resid = phi[t] - rho * phi[t - 1];
    log_dens -= 0.5 * resid * resid / sigma2;
    log_dens -= 0.5 * std::log(2.0 * M_PI * sigma2);
  }

  return log_dens;
}

// =====================================================================
// Temporal log-prior contribution
// =====================================================================

// Compute log-prior for temporal effects (single group)
inline double temporal_log_prior(
    const double* phi,
    int T,
    TemporalType type,
    double tau,           // precision for RW1/RW2, or conditional precision for AR1
    double rho,           // AR1 autocorrelation (ignored for RW)
    bool cyclic
) {
  double log_prior = 0.0;

  if (type == TemporalType::RW1) {
    // RW1: p(phi|tau) propto tau^{(T-1)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw1_quadratic_form(phi, T, cyclic);
    int rank = cyclic ? T : T - 1;  // Rank of precision matrix
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::RW2) {
    // RW2: p(phi|tau) propto tau^{(T-2)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw2_quadratic_form(phi, T, cyclic);
    int rank = cyclic ? T : T - 2;  // Rank of precision matrix
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::AR1) {
    // AR1: proper prior
    log_prior += ar1_log_density(phi, T, rho, tau);
  }

  return log_prior;
}

// =====================================================================
// Sum-to-zero constraint (soft)
// =====================================================================

// Apply soft sum-to-zero constraint penalty for RW models
inline double sum_to_zero_penalty(const double* phi, int T, double lambda = 0.001) {
  double sum = 0.0;
  for (int t = 0; t < T; t++) {
    sum += phi[t];
  }
  return -0.5 * lambda * sum * sum;
}

} // namespace ratiod_temporal

#endif // QUOTR_HMC_TEMPORAL_H
