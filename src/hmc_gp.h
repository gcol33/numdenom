// hmc_gp.h
// Gaussian Process spatial effects with NNGP approximation
// Supports single-scale GP and multi-scale (local + regional) GP

#ifndef RATIOD_HMC_GP_H
#define RATIOD_HMC_GP_H

#include <vector>
#include <cmath>
#include "hmc_svc.h"  // Reuse covariance functions and NNGP infrastructure

namespace ratiod_gp {

using ratiod_svc::CovType;
using ratiod_svc::compute_cov;

// Single-scale GP data structure
struct GPData {
  int n_obs;                          // Number of observations
  int nn;                             // Number of nearest neighbors

  std::vector<double> coords;         // Coordinates (n_obs x 2, flattened)

  // NNGP neighbor structure
  std::vector<int> nn_idx;            // Neighbor indices (n_obs x nn, flattened)
  std::vector<double> nn_dist;        // Distances to neighbors (n_obs x nn, flattened)
  std::vector<int> nn_order;          // Observation ordering for NNGP
  std::vector<int> nn_order_inv;      // Inverse ordering

  CovType cov_type;
  double nu;                          // Matern smoothness (if applicable)
  bool shared;                        // Whether GP is shared between num/denom
};

// Multi-scale GP data structure
struct MultiscaleGPData {
  int n_obs;

  std::vector<double> coords;         // Coordinates (n_obs x 2, flattened)

  // Local scale neighbor structure
  int nn_local;
  std::vector<int> nn_idx_local;
  std::vector<double> nn_dist_local;
  std::vector<int> nn_order_local;
  std::vector<int> nn_order_inv_local;

  // Regional scale neighbor structure
  int nn_regional;
  std::vector<int> nn_idx_regional;
  std::vector<double> nn_dist_regional;
  std::vector<int> nn_order_regional;
  std::vector<int> nn_order_inv_regional;

  // Range prior bounds (for identifiability)
  double range_local_lower, range_local_upper;
  double range_regional_lower, range_regional_upper;

  CovType cov_type;
  double nu;
  bool shared;
};

// -----------------------------------------------------------------------------
// Single-scale GP NNGP likelihood
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for single spatial field
// w: spatial effect values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
inline double gp_nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
    int obs_idx = gp_data.nn_order[i];

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      if (gp_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build covariance vector and matrix
    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = gp_data.nn_dist[nn_flat_idx];
      c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int nn_idx1 = gp_data.nn_order[gp_data.nn_idx[i * nn + j1] - 1];
      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int nn_idx2 = gp_data.nn_order[gp_data.nn_idx[i * nn + j2] - 1];

        if (j1 == j2) {
          C_mat[j1 * n_neighbors + j2] = sigma2;
        } else {
          double d12 = std::sqrt(
            std::pow(gp_data.coords[nn_idx1 * 2] - gp_data.coords[nn_idx2 * 2], 2) +
            std::pow(gp_data.coords[nn_idx1 * 2 + 1] - gp_data.coords[nn_idx2 * 2 + 1], 2)
          );
          C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    // Cholesky decomposition and solve
    std::vector<double> L(n_neighbors * n_neighbors, 0.0);
    for (int j = 0; j < n_neighbors; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = C_mat[j * n_neighbors + k];
        for (int m = 0; m < k; m++) {
          sum -= L[j * n_neighbors + m] * L[k * n_neighbors + m];
        }
        if (j == k) {
          L[j * n_neighbors + j] = std::sqrt(std::max(1e-10, sum));
        } else {
          L[j * n_neighbors + k] = sum / L[k * n_neighbors + k];
        }
      }
    }

    // Solve L * y = c_vec
    std::vector<double> y(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
      double sum = c_vec[j];
      for (int k = 0; k < j; k++) {
        sum -= L[j * n_neighbors + k] * y[k];
      }
      y[j] = sum / L[j * n_neighbors + j];
    }

    // Solve L^T * alpha = y
    std::vector<double> alpha(n_neighbors);
    for (int j = n_neighbors - 1; j >= 0; j--) {
      double sum = y[j];
      for (int k = j + 1; k < n_neighbors; k++) {
        sum -= L[k * n_neighbors + j] * alpha[k];
      }
      alpha[j] = sum / L[j * n_neighbors + j];
    }

    // Conditional mean and variance
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int nn_orig_idx = gp_data.nn_order[gp_data.nn_idx[i * nn + j] - 1];
      cond_mean += alpha[j] * w[nn_orig_idx];
    }

    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec[j] * alpha[j];
    }
    double cond_var = std::max(1e-10, sigma2 - c_Cinv_c);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Multi-scale GP likelihood
// -----------------------------------------------------------------------------

// Compute log-likelihood for multi-scale GP (local + regional)
// w_local: local-scale spatial effect (length n_obs)
// w_regional: regional-scale spatial effect (length n_obs)
// Each component evaluated independently with its own range constraint
inline double multiscale_gp_log_lik(
    const std::vector<double>& w_local,
    const std::vector<double>& w_regional,
    double sigma2_local,
    double phi_local,
    double sigma2_regional,
    double phi_regional,
    const MultiscaleGPData& ms_data
) {
  // Create temporary GPData structures for each scale
  GPData gp_local;
  gp_local.n_obs = ms_data.n_obs;
  gp_local.nn = ms_data.nn_local;
  gp_local.coords = ms_data.coords;
  gp_local.nn_idx = ms_data.nn_idx_local;
  gp_local.nn_dist = ms_data.nn_dist_local;
  gp_local.nn_order = ms_data.nn_order_local;
  gp_local.nn_order_inv = ms_data.nn_order_inv_local;
  gp_local.cov_type = ms_data.cov_type;

  GPData gp_regional;
  gp_regional.n_obs = ms_data.n_obs;
  gp_regional.nn = ms_data.nn_regional;
  gp_regional.coords = ms_data.coords;
  gp_regional.nn_idx = ms_data.nn_idx_regional;
  gp_regional.nn_dist = ms_data.nn_dist_regional;
  gp_regional.nn_order = ms_data.nn_order_regional;
  gp_regional.nn_order_inv = ms_data.nn_order_inv_regional;
  gp_regional.cov_type = ms_data.cov_type;

  // Compute log-likelihood for each scale
  double ll_local = gp_nngp_log_lik(w_local, sigma2_local, phi_local, gp_local);
  double ll_regional = gp_nngp_log_lik(w_regional, sigma2_regional, phi_regional, gp_regional);

  return ll_local + ll_regional;
}

// -----------------------------------------------------------------------------
// Priors for GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for spatial variance (PC prior style)
// P(sigma > U) = alpha => sigma ~ Exponential(rate = -log(alpha)/U)
inline double log_prior_sigma2_pc(double sigma2, double U, double alpha) {
  double rate = -std::log(alpha) / U;
  double sigma = std::sqrt(sigma2);
  // Exponential prior on sigma, transform to sigma2
  // p(sigma) = rate * exp(-rate * sigma)
  // Jacobian: d(sigma)/d(sigma2) = 1/(2*sigma)
  return std::log(rate) - rate * sigma - std::log(2.0 * sigma);
}

// Log prior for range parameter (uniform on log scale within bounds)
inline double log_prior_phi_uniform(double phi, double lower, double upper) {
  if (phi < lower || phi > upper) return -INFINITY;
  // Uniform on [lower, upper]
  return -std::log(upper - lower);
}

// Log prior for range with PC-style (favor larger ranges = simpler models)
inline double log_prior_phi_pc(double phi, double U, double alpha) {
  // P(phi < U) = alpha => 1/phi follows Exponential
  if (phi <= 0) return -INFINITY;
  double rate = -std::log(1.0 - alpha) / U;
  // Prior favors larger phi (simpler, smoother spatial structure)
  return std::log(rate) - rate / phi - 2.0 * std::log(phi);
}

// -----------------------------------------------------------------------------
// Gradient computation for GP parameters (for HMC)
// -----------------------------------------------------------------------------

// Numerical gradient of NNGP log-likelihood w.r.t. w (spatial effects)
// For use in HMC updates
inline void gp_gradient_w(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w,  // Output: gradient (length n_obs)
    double epsilon = 1e-6
) {
  int N = gp_data.n_obs;
  grad_w.resize(N);

  double base_ll = gp_nngp_log_lik(w, sigma2, phi, gp_data);

  // Finite difference for each w[i]
  std::vector<double> w_plus = w;
  for (int i = 0; i < N; i++) {
    w_plus[i] = w[i] + epsilon;
    double ll_plus = gp_nngp_log_lik(w_plus, sigma2, phi, gp_data);
    grad_w[i] = (ll_plus - base_ll) / epsilon;
    w_plus[i] = w[i];  // Reset
  }
}

} // namespace ratiod_gp

#endif // RATIOD_HMC_GP_H
