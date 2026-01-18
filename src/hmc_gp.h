// hmc_gp.h
// Gaussian Process spatial effects with NNGP approximation
// Supports single-scale GP and multi-scale (local + regional) GP

#ifndef RATIOD_HMC_GP_H
#define RATIOD_HMC_GP_H

#include <vector>
#include <cmath>
#include <random>
#include <RcppEigen.h>
#include "hmc_svc.h"  // Reuse covariance functions and NNGP infrastructure

// Verbose debug output (set to false for production)
#define GP_DEBUG_BOUNDS false

namespace ratiod_gp {

using ratiod_svc::CovType;
using ratiod_svc::compute_cov;

// MSGP sampler strategies for advanced optimization
enum class MSGPSampler {
  AUTO,           // Default: use non-centered
  NONCENTERED,    // z ~ N(0,1), w = transform(z) - current default
  CENTERED,       // w ~ NNGP directly
  INTERWEAVED,    // Alternate between centered/non-centered each iteration
  ADAPTIVE,       // Per-parameter adaptive centering (alpha in [0,1])
  RIEMANNIAN,     // Simplified Riemannian HMC with diagonal Fisher metric
  LBFGS           // L-BFGS quasi-Newton mass matrix adaptation - O(md) per step
};

// Parse sampler string to enum
inline MSGPSampler parse_msgp_sampler(const std::string& s) {
  if (s == "noncentered" || s == "auto") return MSGPSampler::NONCENTERED;
  if (s == "centered") return MSGPSampler::CENTERED;
  if (s == "interweaved") return MSGPSampler::INTERWEAVED;
  if (s == "adaptive") return MSGPSampler::ADAPTIVE;
  if (s == "riemannian") return MSGPSampler::RIEMANNIAN;
  if (s == "lbfgs") return MSGPSampler::LBFGS;
  return MSGPSampler::NONCENTERED;  // Default fallback
}

// =============================================================================
// L-BFGS MASS MATRIX ADAPTATION
// =============================================================================
//
// L-BFGS approximates the inverse Hessian using limited memory:
//   H_k ≈ (I - ρ_k s_k y_k^T) H_{k-1} (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
//
// where:
//   s_k = q_k - q_{k-1}     (position difference)
//   y_k = g_k - g_{k-1}     (gradient difference)
//   ρ_k = 1 / (y_k^T s_k)
//
// Storage: O(md) where m = memory size (typically 5-20), d = dimension
// Compute H*v: O(md) via two-loop recursion
// =============================================================================

struct LBFGSState {
    int m;                                    // Memory size (number of pairs to store)
    int d;                                    // Dimension
    int k;                                    // Current iteration count
    std::vector<std::vector<double>> s_list; // Position differences (circular buffer)
    std::vector<std::vector<double>> y_list; // Gradient differences (circular buffer)
    std::vector<double> rho_list;            // 1 / (y^T s) values
    double gamma;                            // Scaling factor for initial H_0

    LBFGSState() : m(0), d(0), k(0), gamma(1.0) {}

    LBFGSState(int memory_size, int dimension)
        : m(memory_size), d(dimension), k(0), gamma(1.0) {
        s_list.reserve(m);
        y_list.reserve(m);
        rho_list.reserve(m);
    }

    // Add a new (s, y) pair from position and gradient differences
    void add_pair(const std::vector<double>& s, const std::vector<double>& y) {
        double ys = 0.0;
        double yy = 0.0;
        for (int i = 0; i < d; i++) {
            ys += y[i] * s[i];
            yy += y[i] * y[i];
        }

        // Skip if curvature condition not satisfied (ensures positive definiteness)
        if (ys < 1e-10) return;

        double rho = 1.0 / ys;

        // Update scaling factor: gamma = (s^T y) / (y^T y)
        if (yy > 1e-10) {
            gamma = ys / yy;
        }

        // Add to circular buffer
        if ((int)s_list.size() < m) {
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(rho);
        } else {
            // Circular replacement
            int idx = k % m;
            s_list[idx] = s;
            y_list[idx] = y;
            rho_list[idx] = rho;
        }
        k++;
    }

    // Two-loop recursion: compute H_k * v in O(md) time
    void multiply_H(const std::vector<double>& v, std::vector<double>& result) const {
        if (d <= 0 || (int)v.size() != d) {
            result = v;
            return;
        }

        result.resize(d);
        for (int i = 0; i < d; i++) {
            result[i] = v[i];
        }

        int n_stored = std::min(k, (int)s_list.size());
        n_stored = std::min(n_stored, m);

        if (n_stored == 0) {
            for (int i = 0; i < d; i++) {
                result[i] *= gamma;
            }
            return;
        }

        std::vector<double> alpha(n_stored);

        // First loop: from newest to oldest
        for (int i = n_stored - 1; i >= 0; i--) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                dot += s_list[idx][j] * result[j];
            }
            alpha[i] = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                result[j] -= alpha[i] * y_list[idx][j];
            }
        }

        // Apply initial Hessian: r = gamma * q
        for (int i = 0; i < d; i++) {
            result[i] *= gamma;
        }

        // Second loop: from oldest to newest
        for (int i = 0; i < n_stored; i++) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                dot += y_list[idx][j] * result[j];
            }
            double beta = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                result[j] += (alpha[i] - beta) * s_list[idx][j];
            }
        }
    }

    // Kinetic energy: K = 0.5 * p^T * H * p
    double kinetic_energy(const std::vector<double>& p) const {
        if ((int)p.size() != d) return 0.0;
        std::vector<double> Hp;
        multiply_H(p, Hp);
        double ke = 0.0;
        for (int i = 0; i < d; i++) {
            ke += p[i] * Hp[i];
        }
        return 0.5 * ke;
    }

    // Get diagonal of B for momentum sampling: sqrt(1/gamma)
    std::vector<double> get_sqrt_B_diag() const {
        std::vector<double> result(d);
        double sqrt_inv_gamma = std::sqrt(1.0 / gamma);
        for (int i = 0; i < d; i++) {
            result[i] = sqrt_inv_gamma;
        }
        return result;
    }
};

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

  // Advanced sampler strategy
  MSGPSampler sampler = MSGPSampler::NONCENTERED;
};

// -----------------------------------------------------------------------------
// Single-scale GP NNGP likelihood
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for single spatial field
// w: spatial effect values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
double gp_nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  // Bounds validation (always on - prevents UB from invalid data structures)
  if (gp_data.nn_order.size() < (size_t)N) return -INFINITY;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return -INFINITY;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return -INFINITY;  // Added: was missing
  if (w.size() < (size_t)N) return -INFINITY;
  if (gp_data.coords.size() < (size_t)(2 * N)) return -INFINITY;

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] gp_nngp_log_lik called: N=" << N << ", nn=" << nn << "\n";
  Rcpp::Rcout << "[GP_DEBUG] w.size()=" << w.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_order.size()=" << gp_data.nn_order.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_idx.size()=" << gp_data.nn_idx.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_dist.size()=" << gp_data.nn_dist.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] coords.size()=" << gp_data.coords.size() << "\n";

  // Validate sizes
  if (gp_data.nn_order.size() < (size_t)N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_order too small! size=" << gp_data.nn_order.size() << " < N=" << N << "\n";
    return -INFINITY;
  }
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_idx too small! size=" << gp_data.nn_idx.size() << " < N*nn=" << (N * nn) << "\n";
    return -INFINITY;
  }
  if (w.size() < (size_t)N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: w too small! size=" << w.size() << " < N=" << N << "\n";
    return -INFINITY;
  }
#endif

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] Accessing nn_order[0]...\n";
#endif
  int first_idx = gp_data.nn_order[0];

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] first_idx=" << first_idx << " (should be 0 to " << (N-1) << ")\n";
  if (first_idx < 0 || first_idx >= N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: first_idx out of bounds!\n";
    return -INFINITY;
  }
#endif

  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] First obs log_lik done, now processing remaining " << (N-1) << " observations\n";
#endif

  // Pre-allocate Eigen matrices/vectors for Cholesky solve
  // Using Eigen avoids hand-rolled linear algebra bugs and leverages SIMD
  Eigen::VectorXd c_vec(nn);
  Eigen::MatrixXd C_mat(nn, nn);
  Eigen::VectorXd alpha(nn);

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
#if GP_DEBUG_BOUNDS
    if (i < 5 || i == N-1) {
      Rcpp::Rcout << "[GP_DEBUG] Processing obs i=" << i << "\n";
    }
#endif

    int obs_idx = gp_data.nn_order[i];

    // Bounds check (always on)
    if (obs_idx < 0 || obs_idx >= N) return -INFINITY;

#if GP_DEBUG_BOUNDS
    if (obs_idx < 0 || obs_idx >= N) {
      Rcpp::Rcout << "[GP_DEBUG] ERROR: obs_idx=" << obs_idx << " out of bounds at i=" << i << "\n";
      return -INFINITY;
    }
#endif

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      // Bounds check (always on)
      if (nn_flat_idx < 0 || nn_flat_idx >= (int)gp_data.nn_idx.size()) return -INFINITY;
#if GP_DEBUG_BOUNDS
      if (nn_flat_idx < 0 || nn_flat_idx >= (int)gp_data.nn_idx.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_flat_idx=" << nn_flat_idx << " out of bounds (nn_idx.size=" << gp_data.nn_idx.size() << ")\n";
        return -INFINITY;
      }
#endif
      if (gp_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

#if GP_DEBUG_BOUNDS
    if (i < 5) {
      Rcpp::Rcout << "[GP_DEBUG]   n_neighbors=" << n_neighbors << "\n";
    }
#endif

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = gp_data.nn_dist[nn_flat_idx];
      c_vec(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int raw_nn_idx1 = gp_data.nn_idx[i * nn + j1];

      // Bounds check: nn_idx is 1-based from R, so subtract 1
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (i < 3 && j1 < 3) {
        Rcpp::Rcout << "[GP_DEBUG]   j1=" << j1 << " raw_nn_idx1=" << raw_nn_idx1 << "\n";
      }
      // nn_idx is 1-based from R, so subtract 1
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: raw_nn_idx1-1=" << (raw_nn_idx1 - 1) << " out of bounds for nn_order (size=" << gp_data.nn_order.size() << ")\n";
        return -INFINITY;
      }
#endif

      int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

      // Bounds check for coords access
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_idx1=" << nn_idx1 << " leads to coords out of bounds (coords.size=" << gp_data.coords.size() << ")\n";
        return -INFINITY;
      }
#endif

      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

        // Bounds check
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) {
          Rcpp::Rcout << "[GP_DEBUG] ERROR: raw_nn_idx2-1=" << (raw_nn_idx2 - 1) << " out of bounds for nn_order\n";
          return -INFINITY;
        }
#endif

        int nn_idx2 = gp_data.nn_order[raw_nn_idx2 - 1];

        if (j1 == j2) {
          C_mat(j1, j2) = sigma2;
        } else {
          double d12 = std::sqrt(
            std::pow(gp_data.coords[nn_idx1 * 2] - gp_data.coords[nn_idx2 * 2], 2) +
            std::pow(gp_data.coords[nn_idx1 * 2 + 1] - gp_data.coords[nn_idx2 * 2 + 1], 2)
          );
          C_mat(j1, j2) = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    // Solve C_mat * alpha = c_vec using Eigen's Cholesky (LLT) decomposition
    // This replaces hand-rolled Cholesky which had optimizer-related bugs on Windows
    Eigen::MatrixXd C_sub = C_mat.topLeftCorner(n_neighbors, n_neighbors);
    Eigen::VectorXd c_sub = c_vec.head(n_neighbors);

    // Add small jitter to diagonal for numerical stability
    // This prevents ill-conditioning when phi is very small or sigma2 is near zero
    for (int j = 0; j < n_neighbors; j++) {
      C_sub(j, j) += 1e-8;
    }

    Eigen::LLT<Eigen::MatrixXd> llt(C_sub);

    // Check if decomposition succeeded - this is the heisenbug fix!
    // Without this check, LLT silently returns garbage/NaN for ill-conditioned matrices,
    // causing unpredictable crashes that depend on parameter values during HMC exploration
    if (llt.info() != Eigen::Success) {
      // Matrix not positive definite - return -INFINITY to reject this parameter state
      return -INFINITY;
    }

    alpha.head(n_neighbors) = llt.solve(c_sub);

    // Conditional mean and variance
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int raw_nn_idx = gp_data.nn_idx[i * nn + j];

      // Bounds check
      if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: cond_mean raw_nn_idx-1=" << (raw_nn_idx - 1) << " out of bounds\n";
        return -INFINITY;
      }
#endif

      int nn_orig_idx = gp_data.nn_order[raw_nn_idx - 1];

      // Bounds check for w access
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_orig_idx=" << nn_orig_idx << " out of bounds for w (size=" << w.size() << ")\n";
        return -INFINITY;
      }
#endif

      cond_mean += alpha(j) * w[nn_orig_idx];
    }

    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec(j) * alpha(j);
    }
    double cond_var = std::max(1e-10, sigma2 - c_Cinv_c);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] gp_nngp_log_lik completed, log_lik=" << log_lik << "\n";
#endif

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

// Struct to hold NNGP gradient results (for hand-coded gradients)
struct NNGPGradients {
  std::vector<double> grad_w;         // Gradient w.r.t. spatial effects
  double grad_log_sigma2;             // Gradient w.r.t. log(sigma2)
  double grad_log_phi;                // Gradient w.r.t. log(phi)
};

// Analytical gradient of NNGP log-likelihood w.r.t. w (spatial effects)
// O(N * nn) complexity - much faster than numerical differentiation
// Returns gradients w.r.t. w only; sigma2/phi gradients computed numerically
inline void gp_nngp_gradient_w_analytical(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w  // Output: gradient (length n_obs)
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grad_w.assign(N, 0.0);

  // Validate input sizes
  if (gp_data.nn_order.size() < (size_t)N) return;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return;
  if (w.size() < (size_t)N) return;
  if (gp_data.coords.size() < (size_t)(2 * N)) return;

  // First observation: marginal N(0, sigma2)
  // ll_0 = -0.5*log(sigma2) - 0.5*w[first]^2/sigma2
  // d/dw[first] = -w[first]/sigma2
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  grad_w[first_idx] = -w[first_idx] / sigma2;

  // Process remaining observations in ordering
  for (int i = 1; i < N; i++) {
    int obs_idx = gp_data.nn_order[i];

    // Bounds check for obs_idx
    if (obs_idx < 0 || obs_idx >= N) continue;

    // Count actual neighbors
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      if (nn_flat_idx >= (int)gp_data.nn_idx.size()) break;
      if (gp_data.nn_idx[nn_flat_idx] > 0) n_neighbors++;
    }

    if (n_neighbors == 0) {
      // No neighbors: marginal - same as first obs
      grad_w[obs_idx] += -w[obs_idx] / sigma2;
      continue;
    }

    // Build c_vec (covariances between obs i and its neighbors)
    std::vector<double> c_vec(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = gp_data.nn_dist[nn_flat_idx];
      c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // Build C_mat (covariances among neighbors)
    std::vector<double> C_mat(n_neighbors * n_neighbors);
    std::vector<int> neighbor_orig_idx(n_neighbors);

    bool bounds_ok = true;
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int raw_nn_idx1 = gp_data.nn_idx[i * nn + j1];

      // Bounds check
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) {
        bounds_ok = false;
        break;
      }

      int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) {
        bounds_ok = false;
        break;
      }

      neighbor_orig_idx[j1] = nn_idx1;

      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) {
          bounds_ok = false;
          break;
        }

        int nn_idx2 = gp_data.nn_order[raw_nn_idx2 - 1];

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
      if (!bounds_ok) break;
    }

    if (!bounds_ok) {
      // Skip this observation - marginal fallback
      grad_w[obs_idx] += -w[obs_idx] / sigma2;
      continue;
    }

    // Cholesky decomposition of C_mat
    std::vector<double> L(n_neighbors * n_neighbors, 0.0);
    for (int j = 0; j < n_neighbors; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += L[j * n_neighbors + k] * L[j * n_neighbors + k];
      }
      double diag = C_mat[j * n_neighbors + j] - sum;
      if (diag <= 0) diag = 1e-10;
      L[j * n_neighbors + j] = std::sqrt(diag);

      for (int k = j + 1; k < n_neighbors; k++) {
        double s = 0.0;
        for (int m = 0; m < j; m++) {
          s += L[k * n_neighbors + m] * L[j * n_neighbors + m];
        }
        L[k * n_neighbors + j] = (C_mat[k * n_neighbors + j] - s) / L[j * n_neighbors + j];
      }
    }

    // Forward solve: L * y = c_vec
    std::vector<double> y(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
      double s = 0.0;
      for (int k = 0; k < j; k++) {
        s += L[j * n_neighbors + k] * y[k];
      }
      y[j] = (c_vec[j] - s) / L[j * n_neighbors + j];
    }

    // Backward solve: L^T * alpha = y
    std::vector<double> alpha(n_neighbors);
    for (int j = n_neighbors - 1; j >= 0; j--) {
      double s = 0.0;
      for (int k = j + 1; k < n_neighbors; k++) {
        s += L[k * n_neighbors + j] * alpha[k];
      }
      alpha[j] = (y[j] - s) / L[j * n_neighbors + j];
    }

    // Conditional mean: mu = alpha' * w_neighbors
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int nn_idx = neighbor_orig_idx[j];
      if (nn_idx >= 0 && nn_idx < N) {
        cond_mean += alpha[j] * w[nn_idx];
      }
    }

    // Conditional variance: sigma2_cond = sigma2 - c' * alpha
    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec[j] * alpha[j];
    }
    double cond_var = std::max(sigma2 - c_Cinv_c, 1e-10);

    // Residual
    double resid = w[obs_idx] - cond_mean;

    // Gradient w.r.t. w[obs_idx] (the target):
    // d(ll_i)/d(w_i) = -resid / cond_var
    grad_w[obs_idx] += -resid / cond_var;

    // Gradient w.r.t. neighbors w[neighbor_j]:
    // d(ll_i)/d(w_neighbor_j) = alpha_j * resid / cond_var
    // (because d(cond_mean)/d(w_neighbor_j) = alpha_j)
    for (int j = 0; j < n_neighbors; j++) {
      int nn_idx = neighbor_orig_idx[j];
      if (nn_idx >= 0 && nn_idx < N) {
        grad_w[nn_idx] += alpha[j] * resid / cond_var;
      }
    }
  }
}

// Full NNGP gradients including sigma2 and phi (using numerical diff for those)
inline void gp_nngp_gradients(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    NNGPGradients& grads,
    double epsilon = 1e-6
) {
  // Analytical gradient w.r.t. w
  gp_nngp_gradient_w_analytical(w, sigma2, phi, gp_data, grads.grad_w);

  // Numerical gradient w.r.t. log_sigma2 (only 1 parameter)
  double log_sigma2 = std::log(sigma2);
  double base_ll = gp_nngp_log_lik(w, sigma2, phi, gp_data);

  double log_sigma2_plus = log_sigma2 + epsilon;
  double sigma2_plus = std::exp(log_sigma2_plus);
  double ll_plus = gp_nngp_log_lik(w, sigma2_plus, phi, gp_data);
  grads.grad_log_sigma2 = (ll_plus - base_ll) / epsilon;

  // Numerical gradient w.r.t. log_phi (only 1 parameter)
  double log_phi = std::log(phi);
  double log_phi_plus = log_phi + epsilon;
  double phi_plus = std::exp(log_phi_plus);
  ll_plus = gp_nngp_log_lik(w, sigma2, phi_plus, gp_data);
  grads.grad_log_phi = (ll_plus - base_ll) / epsilon;
}

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
