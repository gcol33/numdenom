// hmc_svc.h
// Spatially-Varying Coefficients (SVC) with NNGP approximation
// Implements Nearest Neighbor Gaussian Process for scalable GP inference

#ifndef RATIOD_HMC_SVC_H
#define RATIOD_HMC_SVC_H

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <vector>
#include <cmath>
#include <algorithm>

// Fallback definition of M_PI if not provided by <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ratiod_svc {

// Covariance function types
enum class CovType { EXPONENTIAL, MATERN, GAUSSIAN, SPHERICAL };

// SVC data structure
struct SVCData {
  int n_obs;                          // Number of observations
  int n_svc;                          // Number of spatially-varying coefficients
  int nn;                             // Number of nearest neighbors

  std::vector<double> coords;         // Coordinates (n_obs x 2, flattened)
  std::vector<int> svc_indices;       // Which design matrix columns have SVCs
  std::vector<double> X_svc;          // Design matrix subset for SVC terms (n_obs x n_svc)

  // NNGP neighbor structure
  std::vector<int> nn_idx;            // Neighbor indices (n_obs x nn, flattened)
  std::vector<double> nn_dist;        // Distances to neighbors (n_obs x nn, flattened)
  std::vector<int> nn_order;          // Observation ordering for NNGP
  std::vector<int> nn_order_inv;      // Inverse ordering

  CovType cov_type;
  bool shared;                        // Whether SVC is shared between num/denom
};

// -----------------------------------------------------------------------------
// Covariance functions
// -----------------------------------------------------------------------------

// Exponential covariance: sigma^2 * exp(-d / phi)
inline double cov_exponential(double d, double sigma2, double phi) {
  return sigma2 * std::exp(-d / phi);
}

// Matern 3/2 covariance: sigma^2 * (1 + sqrt(3)*d/phi) * exp(-sqrt(3)*d/phi)
inline double cov_matern32(double d, double sigma2, double phi) {
  double r = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + r) * std::exp(-r);
}

// Gaussian (squared exponential) covariance: sigma^2 * exp(-(d/phi)^2)
inline double cov_gaussian(double d, double sigma2, double phi) {
  double r = d / phi;
  return sigma2 * std::exp(-r * r);
}

// Spherical covariance
inline double cov_spherical(double d, double sigma2, double phi) {
  if (d >= phi) return 0.0;
  double r = d / phi;
  return sigma2 * (1.0 - 1.5 * r + 0.5 * r * r * r);
}

// Generic covariance function dispatcher
inline double compute_cov(double d, double sigma2, double phi, CovType cov_type) {
  switch (cov_type) {
    case CovType::EXPONENTIAL:
      return cov_exponential(d, sigma2, phi);
    case CovType::MATERN:
      return cov_matern32(d, sigma2, phi);
    case CovType::GAUSSIAN:
      return cov_gaussian(d, sigma2, phi);
    case CovType::SPHERICAL:
      return cov_spherical(d, sigma2, phi);
    default:
      return cov_exponential(d, sigma2, phi);
  }
}

// -----------------------------------------------------------------------------
// NNGP likelihood computation
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for a single SVC term
// w: vector of SVC values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
// Returns log p(w | sigma2, phi) under NNGP approximation
inline double nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const SVCData& svc_data
) {
  int N = svc_data.n_obs;
  int nn = svc_data.nn;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = svc_data.nn_order[0];
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
    int obs_idx = svc_data.nn_order[i];

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      if (svc_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build covariance vector c(s_i, s_{N(i)}) and matrix C(s_{N(i)}, s_{N(i)})
    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = svc_data.nn_dist[nn_flat_idx];
      c_vec[j] = compute_cov(d, sigma2, phi, svc_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int nn_idx1 = svc_data.nn_order[svc_data.nn_idx[i * nn + j1] - 1];
      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int nn_idx2 = svc_data.nn_order[svc_data.nn_idx[i * nn + j2] - 1];

        if (j1 == j2) {
          C_mat[j1 * n_neighbors + j2] = sigma2;
        } else {
          // Compute distance between neighbors
          double d12 = std::sqrt(
            std::pow(svc_data.coords[nn_idx1 * 2] - svc_data.coords[nn_idx2 * 2], 2) +
            std::pow(svc_data.coords[nn_idx1 * 2 + 1] - svc_data.coords[nn_idx2 * 2 + 1], 2)
          );
          C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, svc_data.cov_type);
        }
      }
    }

    // Solve C_mat * alpha = c_vec using Cholesky decomposition
    // Simple implementation for small matrices (nn typically 15-30)
    std::vector<double> L(n_neighbors * n_neighbors, 0.0);

    // Cholesky decomposition: C = L * L^T
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

    // Conditional mean: mu_i = c^T * C^{-1} * w_{N(i)} = c^T * alpha
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int nn_orig_idx = svc_data.nn_order[svc_data.nn_idx[i * nn + j] - 1];
      cond_mean += alpha[j] * w[nn_orig_idx];
    }

    // Conditional variance: var_i = sigma2 - c^T * C^{-1} * c
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
// SVC contribution to linear predictor
// -----------------------------------------------------------------------------

// Compute SVC contribution to linear predictor for all observations
// eta_svc[i] = sum_j X_svc[i,j] * w_j[i]
inline void compute_svc_eta(
    const std::vector<double>& w_flat,  // n_obs x n_svc flattened
    const SVCData& svc_data,
    std::vector<double>& eta_svc         // Output: length n_obs
) {
  int N = svc_data.n_obs;
  int n_svc = svc_data.n_svc;

  std::fill(eta_svc.begin(), eta_svc.end(), 0.0);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n_svc; j++) {
      // w_flat is stored as [w1[1..N], w2[1..N], ...]
      double w_ij = w_flat[j * N + i];
      double x_ij = svc_data.X_svc[i * n_svc + j];
      eta_svc[i] += x_ij * w_ij;
    }
  }
}

// -----------------------------------------------------------------------------
// Prior on GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for sigma2 (spatial variance): Half-Cauchy or exponential
inline double log_prior_sigma2(double sigma2, double scale) {
  // Half-Cauchy(0, scale): 2 / (pi * scale * (1 + (sigma2/scale)^2))
  // On log scale for sigma = sqrt(sigma2)
  double sigma = std::sqrt(sigma2);
  return std::log(2.0 / (M_PI * scale)) - std::log(1.0 + sigma * sigma / (scale * scale));
}

// Log prior for phi (range parameter): Uniform or exponential
inline double log_prior_phi(double phi, double lower, double upper) {
  // Uniform(lower, upper)
  if (phi < lower || phi > upper) return -INFINITY;
  return -std::log(upper - lower);
}

// Parse covariance type from string
inline CovType parse_cov_type(const std::string& cov_str) {
  if (cov_str == "exponential") return CovType::EXPONENTIAL;
  if (cov_str == "matern") return CovType::MATERN;
  if (cov_str == "gaussian") return CovType::GAUSSIAN;
  if (cov_str == "spherical") return CovType::SPHERICAL;
  return CovType::EXPONENTIAL;  // Default
}

} // namespace ratiod_svc

#endif // RATIOD_HMC_SVC_H
