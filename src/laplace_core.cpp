// laplace_core.cpp
// Core Laplace approximation engine for ratiod
// Implements nested Laplace approximation for latent Gaussian models

#include "laplace_core.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace ratiod {

// ---------------------------------------------------------------------
// Likelihood functions
// ---------------------------------------------------------------------

// Log-likelihood for binomial: y ~ Binomial(n, logit^{-1}(eta))
double log_lik_binomial(int y, int n, double eta) {
  // Numerically stable computation
  // log p = y * eta - n * log(1 + exp(eta)) + log(choose(n, y))
  double log_p;
  if (eta > 0) {
    log_p = y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    log_p = y * eta - n * std::log(1.0 + std::exp(eta));
  }
  // Omit log(choose(n,y)) as it's constant w.r.t. eta
  return log_p;
}

// Gradient: d/d(eta) log p(y | eta) = y - n * p where p = logit^{-1}(eta)
double grad_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return y - n * p;
}

// Negative Hessian: -d²/d(eta)² log p = n * p * (1-p)
double neg_hess_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return n * p * (1.0 - p);
}

// Log-likelihood for negative binomial: y ~ NegBin(mu = exp(eta), phi)
// Parameterization: E[Y] = mu, Var[Y] = mu + mu²/phi
double log_lik_negbin(int y, double eta, double phi) {
  double mu = ratiod_linalg::safe_exp(eta);
  // log p = lgamma(y + phi) - lgamma(phi) - lgamma(y+1)
  //       + phi * log(phi/(mu+phi)) + y * log(mu/(mu+phi))
  double log_p = R::lgammafn(y + phi) - R::lgammafn(phi) - R::lgammafn(y + 1.0)
               + phi * std::log(phi / (mu + phi))
               + y * std::log(mu / (mu + phi));
  return log_p;
}

// Gradient: d/d(eta) log p = y - (y + phi) * mu / (mu + phi)
//                          = y - (y + phi) * p where p = mu/(mu+phi)
double grad_log_lik_negbin(int y, double eta, double phi) {
  double mu = ratiod_linalg::safe_exp(eta);
  double p = mu / (mu + phi);
  return y - (y + phi) * p;
}

// Negative Hessian: (y + phi) * p * (1-p) * mu (chain rule factor)
// Actually: -d²/d(eta)² = (y + phi) * mu * phi / (mu + phi)²
double neg_hess_log_lik_negbin(int y, double eta, double phi) {
  double mu = ratiod_linalg::safe_exp(eta);
  double denom = mu + phi;
  return (y + phi) * mu * phi / (denom * denom);
}

// Log-likelihood for Poisson: y ~ Poisson(mu = exp(eta))
double log_lik_poisson(int y, double eta) {
  // log p = y * eta - exp(eta) - lgamma(y+1)
  return y * eta - ratiod_linalg::safe_exp(eta) - R::lgammafn(y + 1.0);
}

// Gradient: d/d(eta) log p = y - exp(eta)
double grad_log_lik_poisson(int y, double eta) {
  return y - ratiod_linalg::safe_exp(eta);
}

// Negative Hessian: exp(eta)
double neg_hess_log_lik_poisson(int y, double eta) {
  return ratiod_linalg::safe_exp(eta);
}

// ---------------------------------------------------------------------
// Helper functions for sparse matrix operations
// ---------------------------------------------------------------------

// Solve lower triangular system L x = b
void solve_lower(const std::vector<double>& L_vals,
                 const std::vector<int>& L_row,
                 const std::vector<int>& L_col_ptr,
                 const std::vector<double>& b,
                 std::vector<double>& x) {
  int n = b.size();
  x = b;  // Copy b to x

  for (int j = 0; j < n; j++) {
    x[j] /= L_vals[L_col_ptr[j]];  // Diagonal element
    for (int k = L_col_ptr[j] + 1; k < L_col_ptr[j + 1]; k++) {
      x[L_row[k]] -= L_vals[k] * x[j];
    }
  }
}

// Solve upper triangular system L' x = b
void solve_upper(const std::vector<double>& L_vals,
                 const std::vector<int>& L_row,
                 const std::vector<int>& L_col_ptr,
                 const std::vector<double>& b,
                 std::vector<double>& x) {
  int n = b.size();
  x = b;

  for (int j = n - 1; j >= 0; j--) {
    for (int k = L_col_ptr[j] + 1; k < L_col_ptr[j + 1]; k++) {
      x[j] -= L_vals[k] * x[L_row[k]];
    }
    x[j] /= L_vals[L_col_ptr[j]];
  }
}

// ---------------------------------------------------------------------
// Laplace mode finding
// ---------------------------------------------------------------------

// Simple dense implementation for now
// TODO: Add sparse implementation for large problems
LaplaceResult laplace_mode_dense(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_x = p + n_re_groups;  // Fixed effects + random effects

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);  // Hessian (dense for now)

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute linear predictor (parallelized)
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Add random effect
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;  // 0-based
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
    }

    // Compute gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions (parallelized with reduction)
    // Use thread-local storage for accumulation
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    {
      std::vector<double> grad_thread(n_x, 0.0);
      std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

      #pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        double g_i, h_i;

        if (family == "binomial") {
          g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
          h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        } else if (family == "negbin") {
          g_i = grad_log_lik_negbin(y[i], eta[i], phi);
          h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
        } else {  // poisson
          g_i = grad_log_lik_poisson(y[i], eta[i]);
          h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
        }

        // Gradient w.r.t. fixed effects
        for (int j = 0; j < p; j++) {
          grad_thread[j] += g_i * X(i, j);
          for (int k = 0; k < p; k++) {
            H_thread[j][k] += h_i * X(i, j) * X(i, k);
          }
        }

        // Gradient w.r.t. random effects
        if (n_re_groups > 0) {
          int g = (int)re_idx[i] - 1;
          if (g >= 0 && g < n_re_groups) {
            grad_thread[p + g] += g_i;
            H_thread[p + g][p + g] += h_i;

            // Cross terms
            for (int j = 0; j < p; j++) {
              H_thread[j][p + g] += h_i * X(i, j);
              H_thread[p + g][j] += h_i * X(i, j);
            }
          }
        }
      }

      // Reduce thread-local results
      #pragma omp critical
      {
        for (int j = 0; j < n_x; j++) {
          grad_local[j] += grad_thread[j];
          for (int k = 0; k < n_x; k++) {
            H_local[j][k] += H_thread[j][k];
          }
        }
      }
    }
    #else
    // Non-OpenMP fallback
    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {  // poisson
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      // Gradient w.r.t. fixed effects
      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      // Gradient w.r.t. random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;

          // Cross terms
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }
    }
    #endif

    // Copy local results to output
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Add prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // Add weak prior for fixed effects (regularization)
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step: solve H * delta = grad
    // For now, use simple Cholesky (dense)

    // Cholesky factorization of H
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;  // Ensure positive definite
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Solve L * z = grad
    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Solve L' * delta = z
    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update x with NaN/Inf checking
    double step_size = 1.0;
    double max_delta = 0.0;
    bool has_nan = false;
    for (int j = 0; j < n_x; j++) {
      if (!std::isfinite(delta[j])) {
        has_nan = true;
        break;
      }
      max_delta = std::max(max_delta, std::abs(delta[j]));
    }

    // If NaN detected, try smaller step or abort
    if (has_nan) {
      // Reset to previous iteration with smaller step
      step_size = 0.1;
      for (int j = 0; j < n_x; j++) {
        if (std::isfinite(delta[j])) {
          x[j] += step_size * delta[j];
        }
      }
      continue;  // Try next iteration
    }

    for (int j = 0; j < n_x; j++) {
      x[j] += step_size * delta[j];
    }

    // Check convergence
    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }

    result.n_iter = iter + 1;
  }

  // Store results
  result.mode = x;

  // Compute log determinant of H
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);

  // Recompute final Hessian at mode (parallelized)
  NumericVector eta_final(N);
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
  }

  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k < n_x; k++) {
      H(j, k) = 0.0;
    }
  }

  // Parallelize final Hessian computation
  std::vector<std::vector<double>> H_final_local(n_x, std::vector<double>(n_x, 0.0));

  #ifdef _OPENMP
  #pragma omp parallel
  {
    std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

    #pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      double h_i;
      if (family == "binomial") {
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta_final[i]);
      } else if (family == "negbin") {
        h_i = neg_hess_log_lik_negbin(y[i], eta_final[i], phi);
      } else {
        h_i = neg_hess_log_lik_poisson(y[i], eta_final[i]);
      }

      for (int j = 0; j < p; j++) {
        for (int k = 0; k < p; k++) {
          H_thread[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          H_thread[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_thread[j][p + g] += h_i * X(i, j);
            H_thread[p + g][j] += h_i * X(i, j);
          }
        }
      }
    }

    #pragma omp critical
    {
      for (int j = 0; j < n_x; j++) {
        for (int k = 0; k < n_x; k++) {
          H_final_local[j][k] += H_thread[j][k];
        }
      }
    }
  }
  #else
  for (int i = 0; i < N; i++) {
    double h_i;
    if (family == "binomial") {
      h_i = neg_hess_log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      h_i = neg_hess_log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      h_i = neg_hess_log_lik_poisson(y[i], eta_final[i]);
    }

    for (int j = 0; j < p; j++) {
      for (int k = 0; k < p; k++) {
        H_final_local[j][k] += h_i * X(i, j) * X(i, k);
      }
    }

    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        H_final_local[p + g][p + g] += h_i;
        for (int j = 0; j < p; j++) {
          H_final_local[j][p + g] += h_i * X(i, j);
          H_final_local[p + g][j] += h_i * X(i, j);
        }
      }
    }
  }
  #endif

  // Copy to H matrix
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k < n_x; k++) {
      H(j, k) = H_final_local[j][k];
    }
  }

  // Add prior
  for (int g = 0; g < n_re_groups; g++) {
    H(p + g, p + g) += tau_re;
  }
  for (int j = 0; j < p; j++) {
    H(j, j) += 1e-4;
  }

  // Cholesky and log det
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }

  result.log_det_Q = 2.0 * log_det;  // log|H| = 2 * sum(log(diag(L)))

  // Compute log marginal likelihood approximation (parallelized)
  double log_lik = 0.0;
  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:log_lik) schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += log_lik_poisson(y[i], eta_final[i]);
    }
  }

  // Log prior for RE
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));

  // Laplace approximation: log p(y|theta) ≈ log p(y,x*|theta) - 0.5*log|H| + (n_x/2)*log(2π)
  result.log_marginal = log_lik + log_prior_re - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// ---------------------------------------------------------------------
// Laplace mode finding WITH SPATIAL (ICAR)
// ---------------------------------------------------------------------

LaplaceResult laplace_mode_spatial(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const IntegerVector& spatial_idx,
    int n_spatial_units,
    const IntegerVector& adj_row_ptr,
    const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double tau_spatial,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  // Parameters: fixed effects + RE + spatial effects
  int n_x = p + n_re_groups + n_spatial_units;

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Indices
  int spatial_start = p + n_re_groups;

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute linear predictor (parallelized)
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      // Fixed effects
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // Spatial effects
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;  // 1-based to 0-based
        if (s >= 0 && s < n_spatial_units) {
          eta[i] += x[spatial_start + s];
        }
      }
    }

    // Initialize gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions (parallelized with thread-local storage)
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    {
      std::vector<double> grad_thread(n_x, 0.0);
      std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

      #pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        double g_i, h_i;

        if (family == "binomial") {
          g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
          h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        } else if (family == "negbin") {
          g_i = grad_log_lik_negbin(y[i], eta[i], phi);
          h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
        } else {
          g_i = grad_log_lik_poisson(y[i], eta[i]);
          h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
        }

        // Fixed effects contributions
        for (int j = 0; j < p; j++) {
          grad_thread[j] += g_i * X(i, j);
          for (int k = 0; k < p; k++) {
            H_thread[j][k] += h_i * X(i, j) * X(i, k);
          }
        }

        // Random effects contributions
        if (n_re_groups > 0) {
          int g = (int)re_idx[i] - 1;
          if (g >= 0 && g < n_re_groups) {
            grad_thread[p + g] += g_i;
            H_thread[p + g][p + g] += h_i;
            for (int j = 0; j < p; j++) {
              H_thread[j][p + g] += h_i * X(i, j);
              H_thread[p + g][j] += h_i * X(i, j);
            }
          }
        }

        // Spatial effects contributions
        if (n_spatial_units > 0) {
          int s = spatial_idx[i] - 1;
          if (s >= 0 && s < n_spatial_units) {
            int sp_idx = spatial_start + s;
            grad_thread[sp_idx] += g_i;
            H_thread[sp_idx][sp_idx] += h_i;

            // Cross terms with fixed effects
            for (int j = 0; j < p; j++) {
              H_thread[j][sp_idx] += h_i * X(i, j);
              H_thread[sp_idx][j] += h_i * X(i, j);
            }

            // Cross terms with random effects
            if (n_re_groups > 0) {
              int g = (int)re_idx[i] - 1;
              if (g >= 0 && g < n_re_groups) {
                H_thread[p + g][sp_idx] += h_i;
                H_thread[sp_idx][p + g] += h_i;
              }
            }
          }
        }
      }

      // Reduce
      #pragma omp critical
      {
        for (int j = 0; j < n_x; j++) {
          grad_local[j] += grad_thread[j];
          for (int k = 0; k < n_x; k++) {
            H_local[j][k] += H_thread[j][k];
          }
        }
      }
    }
    #else
    // Non-OpenMP fallback
    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }

      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          int sp_idx = spatial_start + s;
          grad_local[sp_idx] += g_i;
          H_local[sp_idx][sp_idx] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][sp_idx] += h_i * X(i, j);
            H_local[sp_idx][j] += h_i * X(i, j);
          }
          if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
              H_local[p + g][sp_idx] += h_i;
              H_local[sp_idx][p + g] += h_i;
            }
          }
        }
      }
    }
    #endif

    // Copy to output
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Add prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // Add ICAR prior contributions for spatial effects
    // ICAR prior: -tau/2 * sum_{i~j} (phi_i - phi_j)^2
    // Gradient: d/d(phi_i) = -tau * n_neighbors[i] * phi_i + tau * sum_{j~i} phi_j
    // Hessian diagonal: tau * n_neighbors[i]
    // Hessian off-diagonal (i~j): -tau
    for (int s = 0; s < n_spatial_units; s++) {
      int sp_idx = spatial_start + s;
      double phi_s = x[sp_idx];

      // Gradient from ICAR: -tau * (n_neighbors[s] * phi_s - sum of neighbors)
      double neighbor_sum = 0.0;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        neighbor_sum += x[spatial_start + neighbor];
      }
      grad[sp_idx] -= tau_spatial * (n_neighbors[s] * phi_s - neighbor_sum);

      // Hessian diagonal
      H(sp_idx, sp_idx) += tau_spatial * n_neighbors[s];

      // Hessian off-diagonal
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        int nb_idx = spatial_start + neighbor;
        H(sp_idx, nb_idx) -= tau_spatial;
      }
    }

    // Small regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step: solve H * delta = grad
    // Cholesky factorization
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Solve L * z = grad
    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Solve L' * delta = z
    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update x
    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    // Check convergence
    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }

    result.n_iter = iter + 1;
  }

  // Center spatial effects (soft sum-to-zero constraint)
  if (n_spatial_units > 0) {
    double mean_phi = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      mean_phi += x[spatial_start + s];
    }
    mean_phi /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      x[spatial_start + s] -= mean_phi;
    }
  }

  result.mode = x;

  // Compute log determinant and log marginal (simplified)
  // Recompute final Hessian at mode for log determinant
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);

  // Recompute H at mode (simplified - just use last H)
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }

  result.log_det_Q = 2.0 * log_det;

  // Compute log marginal (simplified approximation)
  double log_lik = 0.0;
  NumericVector eta_final(N);
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (n_spatial_units > 0) {
      int s = spatial_idx[i] - 1;
      if (s >= 0 && s < n_spatial_units) {
        eta_final[i] += x[spatial_start + s];
      }
    }
  }

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:log_lik) schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += log_lik_poisson(y[i], eta_final[i]);
    }
  }

  // Log prior for RE
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  // Log prior for spatial (ICAR) - simplified
  double log_prior_spatial = 0.0;
  if (n_spatial_units > 0) {
    double quad_form = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      double phi_s = x[spatial_start + s];
      quad_form += n_neighbors[s] * phi_s * phi_s;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        if (neighbor > s) {  // Count each edge once
          double phi_n = x[spatial_start + neighbor];
          quad_form -= 2.0 * phi_s * phi_n;
        }
      }
    }
    log_prior_spatial = -0.5 * tau_spatial * quad_form;
    // Add normalizing constant approximation
    log_prior_spatial += 0.5 * (n_spatial_units - 1) * std::log(tau_spatial / (2.0 * M_PI));
  }

  result.log_marginal = log_lik + log_prior_re + log_prior_spatial
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// ---------------------------------------------------------------------
// Laplace mode finding WITH SPATIAL (BYM2)
// BYM2: u = sigma * (sqrt(rho) * phi_scaled + sqrt(1-rho) * theta)
// phi_scaled ~ ICAR, theta ~ N(0, I)
// ---------------------------------------------------------------------

LaplaceResult laplace_mode_bym2(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const IntegerVector& spatial_idx,
    int n_spatial_units,
    const IntegerVector& adj_row_ptr,
    const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double sigma_spatial,    // Total spatial SD
    double rho,              // Mixing proportion (0 = all IID, 1 = all structured)
    double scale_factor,     // BYM2 scaling factor
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  // Parameters: fixed effects + RE + phi_scaled + theta
  int n_x = p + n_re_groups + 2 * n_spatial_units;

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Indices for BYM2 components
  int phi_start = p + n_re_groups;              // phi_scaled (structured)
  int theta_start = phi_start + n_spatial_units; // theta (unstructured)

  // Precompute BYM2 weights
  double sqrt_rho = std::sqrt(rho + 1e-10);
  double sqrt_1_rho = std::sqrt(1.0 - rho + 1e-10);

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute spatial effects and linear predictor
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      // Fixed effects
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // BYM2 spatial effects: sigma * (sqrt(rho) * phi * scale + sqrt(1-rho) * theta)
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          double phi_s = x[phi_start + s];
          double theta_s = x[theta_start + s];
          double spatial_effect = sigma_spatial * (
            sqrt_rho * phi_s * scale_factor + sqrt_1_rho * theta_s
          );
          eta[i] += spatial_effect;
        }
      }
    }

    // Initialize gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    {
      std::vector<double> grad_thread(n_x, 0.0);
      std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

      #pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        double g_i, h_i;

        if (family == "binomial") {
          g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
          h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        } else if (family == "negbin") {
          g_i = grad_log_lik_negbin(y[i], eta[i], phi);
          h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
        } else {
          g_i = grad_log_lik_poisson(y[i], eta[i]);
          h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
        }

        // Fixed effects
        for (int j = 0; j < p; j++) {
          grad_thread[j] += g_i * X(i, j);
          for (int k = 0; k < p; k++) {
            H_thread[j][k] += h_i * X(i, j) * X(i, k);
          }
        }

        // Random effects
        if (n_re_groups > 0) {
          int g = (int)re_idx[i] - 1;
          if (g >= 0 && g < n_re_groups) {
            grad_thread[p + g] += g_i;
            H_thread[p + g][p + g] += h_i;
            for (int j = 0; j < p; j++) {
              H_thread[j][p + g] += h_i * X(i, j);
              H_thread[p + g][j] += h_i * X(i, j);
            }
          }
        }

        // BYM2 spatial effects
        if (n_spatial_units > 0) {
          int s = spatial_idx[i] - 1;
          if (s >= 0 && s < n_spatial_units) {
            // Derivative factors for phi and theta
            double d_phi = sigma_spatial * sqrt_rho * scale_factor;
            double d_theta = sigma_spatial * sqrt_1_rho;

            int phi_idx = phi_start + s;
            int theta_idx = theta_start + s;

            // Gradient contributions
            grad_thread[phi_idx] += g_i * d_phi;
            grad_thread[theta_idx] += g_i * d_theta;

            // Hessian contributions
            H_thread[phi_idx][phi_idx] += h_i * d_phi * d_phi;
            H_thread[theta_idx][theta_idx] += h_i * d_theta * d_theta;
            H_thread[phi_idx][theta_idx] += h_i * d_phi * d_theta;
            H_thread[theta_idx][phi_idx] += h_i * d_phi * d_theta;

            // Cross terms with fixed effects
            for (int j = 0; j < p; j++) {
              H_thread[j][phi_idx] += h_i * X(i, j) * d_phi;
              H_thread[phi_idx][j] += h_i * X(i, j) * d_phi;
              H_thread[j][theta_idx] += h_i * X(i, j) * d_theta;
              H_thread[theta_idx][j] += h_i * X(i, j) * d_theta;
            }

            // Cross terms with random effects
            if (n_re_groups > 0) {
              int g = (int)re_idx[i] - 1;
              if (g >= 0 && g < n_re_groups) {
                H_thread[p + g][phi_idx] += h_i * d_phi;
                H_thread[phi_idx][p + g] += h_i * d_phi;
                H_thread[p + g][theta_idx] += h_i * d_theta;
                H_thread[theta_idx][p + g] += h_i * d_theta;
              }
            }
          }
        }
      }

      #pragma omp critical
      {
        for (int j = 0; j < n_x; j++) {
          grad_local[j] += grad_thread[j];
          for (int k = 0; k < n_x; k++) {
            H_local[j][k] += H_thread[j][k];
          }
        }
      }
    }
    #else
    // Non-OpenMP fallback (similar logic)
    for (int i = 0; i < N; i++) {
      double g_i, h_i;
      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }

      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          double d_phi = sigma_spatial * sqrt_rho * scale_factor;
          double d_theta = sigma_spatial * sqrt_1_rho;
          int phi_idx = phi_start + s;
          int theta_idx = theta_start + s;

          grad_local[phi_idx] += g_i * d_phi;
          grad_local[theta_idx] += g_i * d_theta;
          H_local[phi_idx][phi_idx] += h_i * d_phi * d_phi;
          H_local[theta_idx][theta_idx] += h_i * d_theta * d_theta;
          H_local[phi_idx][theta_idx] += h_i * d_phi * d_theta;
          H_local[theta_idx][phi_idx] += h_i * d_phi * d_theta;

          for (int j = 0; j < p; j++) {
            H_local[j][phi_idx] += h_i * X(i, j) * d_phi;
            H_local[phi_idx][j] += h_i * X(i, j) * d_phi;
            H_local[j][theta_idx] += h_i * X(i, j) * d_theta;
            H_local[theta_idx][j] += h_i * X(i, j) * d_theta;
          }

          if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
              H_local[p + g][phi_idx] += h_i * d_phi;
              H_local[phi_idx][p + g] += h_i * d_phi;
              H_local[p + g][theta_idx] += h_i * d_theta;
              H_local[theta_idx][p + g] += h_i * d_theta;
            }
          }
        }
      }
    }
    #endif

    // Copy to output
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // Prior for phi_scaled (ICAR with precision 1)
    for (int s = 0; s < n_spatial_units; s++) {
      int phi_idx = phi_start + s;
      double phi_s = x[phi_idx];

      double neighbor_sum = 0.0;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        neighbor_sum += x[phi_start + neighbor];
      }
      // ICAR with precision 1 (variance scaling handled by sigma_spatial)
      grad[phi_idx] -= (n_neighbors[s] * phi_s - neighbor_sum);
      H(phi_idx, phi_idx) += n_neighbors[s];
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        H(phi_idx, phi_start + neighbor) -= 1.0;
      }
    }

    // Prior for theta (IID N(0, 1))
    for (int s = 0; s < n_spatial_units; s++) {
      int theta_idx = theta_start + s;
      grad[theta_idx] -= x[theta_idx];
      H(theta_idx, theta_idx) += 1.0;
    }

    // Regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  // Center phi_scaled (sum-to-zero constraint)
  if (n_spatial_units > 0) {
    double mean_phi = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      mean_phi += x[phi_start + s];
    }
    mean_phi /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      x[phi_start + s] -= mean_phi;
    }
  }

  result.mode = x;

  // Compute log determinant
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;

  // Compute log marginal (simplified)
  double log_lik = 0.0;
  NumericVector eta_final(N);
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (n_spatial_units > 0) {
      int s = spatial_idx[i] - 1;
      if (s >= 0 && s < n_spatial_units) {
        double spatial_effect = sigma_spatial * (
          sqrt_rho * x[phi_start + s] * scale_factor +
          sqrt_1_rho * x[theta_start + s]
        );
        eta_final[i] += spatial_effect;
      }
    }
  }

  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += log_lik_poisson(y[i], eta_final[i]);
    }
  }

  // Log priors
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  // Log prior for theta (IID)
  double log_prior_theta = 0.0;
  for (int s = 0; s < n_spatial_units; s++) {
    log_prior_theta -= 0.5 * x[theta_start + s] * x[theta_start + s];
  }
  log_prior_theta -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);

  // Log prior for phi_scaled (ICAR) - simplified
  double log_prior_phi = 0.0;
  double quad_form = 0.0;
  for (int s = 0; s < n_spatial_units; s++) {
    double phi_s = x[phi_start + s];
    quad_form += n_neighbors[s] * phi_s * phi_s;
    for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
      int neighbor = adj_col_idx[k];  // Already 0-based
      if (neighbor > s) {
        quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
      }
    }
  }
  log_prior_phi = -0.5 * quad_form;

  result.log_marginal = log_lik + log_prior_re + log_prior_theta + log_prior_phi
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

} // namespace ratiod

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_dense(
    y, n, X, re_idx, n_re_groups, sigma_re, family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
int cpp_laplace_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_spatial(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double tau_spatial,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_spatial(
    y, n, X, re_idx, n_re_groups, sigma_re,
    spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors, tau_spatial,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_laplace_sample(
    Rcpp::NumericVector mode,
    Rcpp::NumericMatrix H,
    int n_samples
) {
  int n_x = mode.size();
  Rcpp::NumericMatrix samples(n_samples, n_x);

  // Cholesky of H^{-1} = L_inv L_inv'
  // We need to sample from N(mode, H^{-1})
  // Sample z ~ N(0, I), then x = mode + L_inv * z

  // First, Cholesky of H
  Rcpp::NumericMatrix L(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L(j, i) * L(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L(j, k) = std::sqrt(sum);
      } else {
        L(j, k) = sum / L(k, k);
      }
    }
  }

  // Sample
  for (int s = 0; s < n_samples; s++) {
    // z ~ N(0, I)
    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      z[j] = R::rnorm(0.0, 1.0);
    }

    // Solve L' x_centered = z (back substitution)
    Rcpp::NumericVector x_centered(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * x_centered[k];
      }
      x_centered[j] = sum / L(j, j);
    }

    // x = mode + x_centered
    for (int j = 0; j < n_x; j++) {
      samples(s, j) = mode[j] + x_centered[j];
    }
  }

  return samples;
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_bym2(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double sigma_spatial,
    double rho,
    double scale_factor,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_bym2(
    y, n, X, re_idx, n_re_groups, sigma_re,
    spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors,
    sigma_spatial, rho, scale_factor,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// ---------------------------------------------------------------------
// GP Spatial Laplace (NNGP approximation)
// ---------------------------------------------------------------------

// Helper: compute squared exponential covariance
inline double compute_cov_exp_laplace(double d, double sigma2, double phi) {
  if (d < 1e-10) return sigma2;
  return sigma2 * std::exp(-d / phi);
}

// Helper: compute Matern covariance (nu = 1.5)
inline double compute_cov_matern15_laplace(double d, double sigma2, double phi) {
  if (d < 1e-10) return sigma2;
  double x = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + x) * std::exp(-x);
}

// Helper: compute Matern covariance (nu = 2.5)
inline double compute_cov_matern25_laplace(double d, double sigma2, double phi) {
  if (d < 1e-10) return sigma2;
  double x = std::sqrt(5.0) * d / phi;
  return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

// Compute covariance based on type
inline double compute_cov_laplace(double d, double sigma2, double phi, int cov_type) {
  if (cov_type == 0) {
    return compute_cov_exp_laplace(d, sigma2, phi);
  } else if (cov_type == 1) {
    return compute_cov_matern15_laplace(d, sigma2, phi);
  } else {
    return compute_cov_matern25_laplace(d, sigma2, phi);
  }
}

// Compute NNGP conditional mean and variance for observation i
inline void nngp_conditional_laplace(
    int obs_idx,
    int i,
    const std::vector<double>& w,
    double sigma2,
    double phi,
    int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int nn,
    double& cond_mean,
    double& cond_var
) {
  int n_neighbors = 0;
  for (int j = 0; j < nn; j++) {
    if (nn_idx(i, j) > 0) n_neighbors++;
  }

  if (n_neighbors == 0) {
    cond_mean = 0.0;
    cond_var = sigma2;
    return;
  }

  std::vector<double> c_vec(n_neighbors);
  std::vector<double> C_mat(n_neighbors * n_neighbors);

  for (int j = 0; j < n_neighbors; j++) {
    c_vec[j] = compute_cov_laplace(nn_dist(i, j), sigma2, phi, cov_type);
  }

  for (int j1 = 0; j1 < n_neighbors; j1++) {
    int nn_orig1 = nn_order[nn_idx(i, j1) - 1];
    for (int j2 = 0; j2 < n_neighbors; j2++) {
      int nn_orig2 = nn_order[nn_idx(i, j2) - 1];
      if (j1 == j2) {
        C_mat[j1 * n_neighbors + j2] = sigma2;
      } else {
        double d12 = std::sqrt(
          std::pow(coords(nn_orig1, 0) - coords(nn_orig2, 0), 2) +
          std::pow(coords(nn_orig1, 1) - coords(nn_orig2, 1), 2)
        );
        C_mat[j1 * n_neighbors + j2] = compute_cov_laplace(d12, sigma2, phi, cov_type);
      }
    }
  }

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

  std::vector<double> y_solve(n_neighbors);
  for (int j = 0; j < n_neighbors; j++) {
    double sum = c_vec[j];
    for (int k = 0; k < j; k++) {
      sum -= L[j * n_neighbors + k] * y_solve[k];
    }
    y_solve[j] = sum / L[j * n_neighbors + j];
  }

  std::vector<double> alpha(n_neighbors);
  for (int j = n_neighbors - 1; j >= 0; j--) {
    double sum = y_solve[j];
    for (int k = j + 1; k < n_neighbors; k++) {
      sum -= L[k * n_neighbors + j] * alpha[k];
    }
    alpha[j] = sum / L[j * n_neighbors + j];
  }

  cond_mean = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    int nn_orig = nn_order[nn_idx(i, j) - 1];
    cond_mean += alpha[j] * w[nn_orig];
  }

  double c_Cinv_c = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    c_Cinv_c += c_vec[j] * alpha[j];
  }
  cond_var = std::max(1e-10, sigma2 - c_Cinv_c);
}

// Laplace mode finding with GP spatial (NNGP)
ratiod::LaplaceResult laplace_mode_gp(
    const Rcpp::IntegerVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial,
    int nn,
    double sigma2_gp,
    double phi_gp,
    int cov_type,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_x = p + n_re_groups + n_spatial;

  ratiod::LaplaceResult result;
  result.mode = Rcpp::NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  Rcpp::NumericVector x(n_x, 0.0);
  Rcpp::NumericVector grad(n_x);
  Rcpp::NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
  int gp_start = p + n_re_groups;

  std::vector<double> w(n_spatial);

  #ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  #endif

  for (int iter = 0; iter < max_iter; iter++) {
    for (int s = 0; s < n_spatial; s++) {
      w[s] = x[gp_start + s];
    }

    Rcpp::NumericVector eta(N);
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      if (i < n_spatial) {
        eta[i] += x[gp_start + i];
      }
    }

    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = ratiod::grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = ratiod::grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = ratiod::neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = ratiod::grad_log_lik_poisson(y[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H(j, k) += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad[p + g] += g_i;
          H(p + g, p + g) += h_i;
        }
      }

      if (i < n_spatial) {
        int gp_idx = gp_start + i;
        grad[gp_idx] += g_i;
        H(gp_idx, gp_idx) += h_i;
      }
    }

    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    for (int i = 0; i < n_spatial; i++) {
      int obs_idx = nn_order[i];
      int gp_idx = gp_start + obs_idx;

      double cond_mean, cond_var;
      nngp_conditional_laplace(obs_idx, i, w, sigma2_gp, phi_gp, cov_type,
                               coords, nn_idx, nn_dist, nn_order, nn,
                               cond_mean, cond_var);

      double tau_cond = 1.0 / cond_var;
      grad[gp_idx] -= tau_cond * (w[obs_idx] - cond_mean);
      H(gp_idx, gp_idx) += tau_cond;
    }

    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    Rcpp::NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int m = 0; m < k; m++) {
          sum -= L(j, m) * L(k, m);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    Rcpp::NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  result.mode = x;

  double log_det = 0.0;
  Rcpp::NumericMatrix L_final(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int m = 0; m < k; m++) {
        sum -= L_final(j, m) * L_final(k, m);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;

  double log_lik = 0.0;
  Rcpp::NumericVector eta_final(N);
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (i < n_spatial) {
      eta_final[i] += x[gp_start + i];
    }
  }

  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += ratiod::log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += ratiod::log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += ratiod::log_lik_poisson(y[i], eta_final[i]);
    }
  }

  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  double log_prior_gp = 0.0;
  for (int s = 0; s < n_spatial; s++) {
    w[s] = x[gp_start + s];
  }
  for (int i = 0; i < n_spatial; i++) {
    int obs_idx = nn_order[i];
    double cond_mean, cond_var;
    nngp_conditional_laplace(obs_idx, i, w, sigma2_gp, phi_gp, cov_type,
                             coords, nn_idx, nn_dist, nn_order, nn,
                             cond_mean, cond_var);
    double resid = w[obs_idx] - cond_mean;
    log_prior_gp += -0.5 * std::log(2.0 * M_PI * cond_var) -
                    0.5 * resid * resid / cond_var;
  }

  result.log_marginal = log_lik + log_prior_re + log_prior_gp
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// ---------------------------------------------------------------------
// Multiscale Temporal Laplace
// ---------------------------------------------------------------------

inline void add_rw1_precision_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n_times,
    double tau,
    bool cyclic
) {
  for (int t = 0; t < n_times; t++) {
    int idx = start_idx + t;

    if (t > 0) {
      double diff = x[idx] - x[idx - 1];
      grad[idx] -= tau * diff;
      grad[idx - 1] += tau * diff;
      H(idx, idx) += tau;
      H(idx - 1, idx - 1) += tau;
      H(idx, idx - 1) -= tau;
      H(idx - 1, idx) -= tau;
    }
  }

  if (cyclic && n_times > 1) {
    int idx_first = start_idx;
    int idx_last = start_idx + n_times - 1;
    double diff = x[idx_first] - x[idx_last];
    grad[idx_first] -= tau * diff;
    grad[idx_last] += tau * diff;
    H(idx_first, idx_first) += tau;
    H(idx_last, idx_last) += tau;
    H(idx_first, idx_last) -= tau;
    H(idx_last, idx_first) -= tau;
  }
}

inline void add_rw2_precision_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n_times,
    double tau,
    bool cyclic
) {
  if (n_times < 3) return;

  for (int t = 1; t < n_times - 1; t++) {
    int idx = start_idx + t;
    double diff2 = x[idx - 1] - 2.0 * x[idx] + x[idx + 1];

    grad[idx - 1] -= tau * diff2;
    grad[idx] += 2.0 * tau * diff2;
    grad[idx + 1] -= tau * diff2;

    H(idx - 1, idx - 1) += tau;
    H(idx, idx) += 4.0 * tau;
    H(idx + 1, idx + 1) += tau;
    H(idx - 1, idx) -= 2.0 * tau;
    H(idx, idx - 1) -= 2.0 * tau;
    H(idx, idx + 1) -= 2.0 * tau;
    H(idx + 1, idx) -= 2.0 * tau;
    H(idx - 1, idx + 1) += tau;
    H(idx + 1, idx - 1) += tau;
  }
}

inline void add_ar1_precision_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n_times,
    double tau,
    double rho
) {
  if (n_times < 1) return;

  double tau_marginal = tau * (1.0 - rho * rho);
  int idx0 = start_idx;
  grad[idx0] -= tau_marginal * x[idx0];
  H(idx0, idx0) += tau_marginal;

  for (int t = 1; t < n_times; t++) {
    int idx = start_idx + t;
    int idx_prev = start_idx + t - 1;
    double resid = x[idx] - rho * x[idx_prev];

    grad[idx] -= tau * resid;
    grad[idx_prev] += tau * rho * resid;

    H(idx, idx) += tau;
    H(idx_prev, idx_prev) += tau * rho * rho;
    H(idx, idx_prev) -= tau * rho;
    H(idx_prev, idx) -= tau * rho;
  }
}

ratiod::LaplaceResult laplace_mode_multiscale_temporal(
    const Rcpp::IntegerVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const Rcpp::IntegerVector& time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();

  int n_trend = (trend_type > 0) ? n_times : 0;
  int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
  int n_short = (short_type > 0) ? n_times : 0;

  int n_x = p + n_re_groups + n_trend + n_seasonal + n_short;

  ratiod::LaplaceResult result;
  result.mode = Rcpp::NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  Rcpp::NumericVector x(n_x, 0.0);
  Rcpp::NumericVector grad(n_x);
  Rcpp::NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
  double tau_trend = 1.0 / (sigma2_trend + 1e-10);
  double tau_seasonal = 1.0 / (sigma2_seasonal + 1e-10);
  double tau_short = 1.0 / (sigma2_short + 1e-10);

  int trend_start = p + n_re_groups;
  int seasonal_start = trend_start + n_trend;
  int short_start = seasonal_start + n_seasonal;

  #ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  #endif

  for (int iter = 0; iter < max_iter; iter++) {
    Rcpp::NumericVector eta(N);
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }

      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) {
          eta[i] += x[trend_start + t];
        }
        if (n_seasonal > 0) {
          int s = t % seasonal_period;
          if (s < n_seasonal) {
            eta[i] += x[seasonal_start + s];
          }
        }
        if (n_short > 0 && t < n_short) {
          eta[i] += x[short_start + t];
        }
      }
    }

    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = ratiod::grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = ratiod::grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = ratiod::neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = ratiod::grad_log_lik_poisson(y[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H(j, k) += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad[p + g] += g_i;
          H(p + g, p + g) += h_i;
        }
      }

      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) {
          int idx = trend_start + t;
          grad[idx] += g_i;
          H(idx, idx) += h_i;
        }
        if (n_seasonal > 0) {
          int s = t % seasonal_period;
          if (s < n_seasonal) {
            int idx = seasonal_start + s;
            grad[idx] += g_i;
            H(idx, idx) += h_i;
          }
        }
        if (n_short > 0 && t < n_short) {
          int idx = short_start + t;
          grad[idx] += g_i;
          H(idx, idx) += h_i;
        }
      }
    }

    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    if (trend_type == 1) {
      add_rw1_precision_laplace(grad, H, x, trend_start, n_trend, tau_trend, false);
    } else if (trend_type == 2) {
      add_rw2_precision_laplace(grad, H, x, trend_start, n_trend, tau_trend, false);
    }

    if (n_seasonal > 0) {
      add_rw1_precision_laplace(grad, H, x, seasonal_start, n_seasonal, tau_seasonal, true);
    }

    if (short_type == 1) {
      add_ar1_precision_laplace(grad, H, x, short_start, n_short, tau_short, rho_short);
    } else if (short_type == 2) {
      for (int t = 0; t < n_short; t++) {
        int idx = short_start + t;
        grad[idx] -= tau_short * x[idx];
        H(idx, idx) += tau_short;
      }
    }

    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    Rcpp::NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int m = 0; m < k; m++) {
          sum -= L(j, m) * L(k, m);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    Rcpp::NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  if (n_trend > 0) {
    double mean_trend = 0.0;
    for (int t = 0; t < n_trend; t++) mean_trend += x[trend_start + t];
    mean_trend /= n_trend;
    for (int t = 0; t < n_trend; t++) x[trend_start + t] -= mean_trend;
  }
  if (n_seasonal > 0) {
    double mean_seasonal = 0.0;
    for (int s = 0; s < n_seasonal; s++) mean_seasonal += x[seasonal_start + s];
    mean_seasonal /= n_seasonal;
    for (int s = 0; s < n_seasonal; s++) x[seasonal_start + s] -= mean_seasonal;
  }

  result.mode = x;

  double log_det = 0.0;
  Rcpp::NumericMatrix L_final(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int m = 0; m < k; m++) {
        sum -= L_final(j, m) * L_final(k, m);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;
  result.log_marginal = -0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial,
    int nn,
    double sigma2_gp,
    double phi_gp,
    int cov_type,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = laplace_mode_gp(
    y, n, X, re_idx, n_re_groups, sigma_re,
    coords, nn_idx, nn_dist, nn_order, n_spatial, nn,
    sigma2_gp, phi_gp, cov_type,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// Laplace approximation for multiscale GP (local + regional)
// This treats local and regional as independent GP components
namespace ratiod {
LaplaceResult laplace_mode_multiscale_gp(
    const Rcpp::IntegerVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx_local,
    const Rcpp::NumericMatrix& nn_dist_local,
    const Rcpp::IntegerVector& nn_order_local,
    int nn_local,
    const Rcpp::IntegerMatrix& nn_idx_regional,
    const Rcpp::NumericMatrix& nn_dist_regional,
    const Rcpp::IntegerVector& nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local,
    double phi_local,
    double sigma2_regional,
    double phi_regional,
    int cov_type,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  // n_spatial = n_obs for point-referenced data
  // Parameter layout: [beta(p)] [re(n_re_groups)] [w_local(n_spatial)] [w_regional(n_spatial)]
  int N = y.size();
  int p = X.ncol();

  int local_start = p + n_re_groups;
  int regional_start = local_start + n_spatial;
  int n_x = regional_start + n_spatial;

  ratiod::LaplaceResult result;
  result.mode = Rcpp::NumericVector(n_x, 0.0);
  result.log_det_Q = 0.0;
  result.log_marginal = 0.0;
  result.n_iter = 0;
  result.converged = false;

  // Initialize
  Rcpp::NumericVector x = Rcpp::clone(result.mode);

  // RE precision
  double tau_re = (sigma_re > 0) ? 1.0 / (sigma_re * sigma_re) : 0.01;

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute eta = X*beta + re + w_local + w_regional
    Rcpp::NumericVector eta(N, 0.0);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // Add local GP effect
      if (i < n_spatial) {
        eta[i] += x[local_start + i];
        eta[i] += x[regional_start + i];
      }
    }

    // Compute gradient and Hessian
    Rcpp::NumericVector grad(n_x, 0.0);
    Rcpp::NumericMatrix H(n_x, n_x);

    // Likelihood contributions
    for (int i = 0; i < N; i++) {
      double g_i, h_i;
      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      // Fixed effects
      for (int j = 0; j < p; j++) {
        grad[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H(j, k) += h_i * X(i, j) * X(i, k);
        }
      }

      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad[p + g] += g_i;
          H(p + g, p + g) += h_i;
        }
      }

      // Local GP effect
      if (i < n_spatial) {
        int idx_local = local_start + i;
        grad[idx_local] += g_i;
        H(idx_local, idx_local) += h_i;

        // Regional GP effect
        int idx_regional = regional_start + i;
        grad[idx_regional] += g_i;
        H(idx_regional, idx_regional) += h_i;

        // Cross terms between local and regional
        H(idx_local, idx_regional) += h_i;
        H(idx_regional, idx_local) += h_i;
      }
    }

    // Prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // GP prior contribution - local (sparse NNGP)
    double tau_local = 1.0 / sigma2_local;
    for (int i = 0; i < n_spatial; i++) {
      int idx = local_start + i;

      // Compute conditional mean from neighbors
      double cond_mean = 0.0;
      double cond_prec = tau_local;
      int n_neighbors = 0;

      for (int k = 0; k < nn_local; k++) {
        int neighbor = nn_idx_local(i, k) - 1;
        if (neighbor >= 0 && neighbor < n_spatial) {
          double dist = nn_dist_local(i, k);
          double cov_val = std::exp(-dist / phi_local);  // Exponential covariance
          cond_mean += cov_val * x[local_start + neighbor];
          n_neighbors++;
        }
      }
      if (n_neighbors > 0) {
        cond_mean *= tau_local;
      }

      grad[idx] -= tau_local * x[idx] - cond_mean;
      H(idx, idx) += tau_local;
    }

    // GP prior contribution - regional (sparse NNGP)
    double tau_regional = 1.0 / sigma2_regional;
    for (int i = 0; i < n_spatial; i++) {
      int idx = regional_start + i;

      double cond_mean = 0.0;
      double cond_prec = tau_regional;
      int n_neighbors = 0;

      for (int k = 0; k < nn_regional; k++) {
        int neighbor = nn_idx_regional(i, k) - 1;
        if (neighbor >= 0 && neighbor < n_spatial) {
          double dist = nn_dist_regional(i, k);
          double cov_val = std::exp(-dist / phi_regional);  // Exponential covariance
          cond_mean += cov_val * x[regional_start + neighbor];
          n_neighbors++;
        }
      }
      if (n_neighbors > 0) {
        cond_mean *= tau_regional;
      }

      grad[idx] -= tau_regional * x[idx] - cond_mean;
      H(idx, idx) += tau_regional;
    }

    // Regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step (using Cholesky decomposition)
    Rcpp::NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Forward substitution
    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Back substitution
    Rcpp::NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update
    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  // Center GP effects (sum-to-zero)
  double mean_local = 0.0, mean_regional = 0.0;
  for (int i = 0; i < n_spatial; i++) {
    mean_local += x[local_start + i];
    mean_regional += x[regional_start + i];
  }
  mean_local /= n_spatial;
  mean_regional /= n_spatial;
  for (int i = 0; i < n_spatial; i++) {
    x[local_start + i] -= mean_local;
    x[regional_start + i] -= mean_regional;
  }

  result.mode = x;
  return result;
}

// ---------------------------------------------------------------------
// Laplace mode finding with RSR (Restricted Spatial Regression)
// RSR projects spatial effects to be orthogonal to covariate space
// w_rsr = P_perp * w where P_perp = I - X(X'X)^{-1}X'
// ---------------------------------------------------------------------

LaplaceResult laplace_mode_rsr(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const IntegerVector& spatial_idx,
    int n_spatial_units,
    const IntegerVector& adj_row_ptr,
    const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double tau_spatial,
    const NumericVector& rsr_projection,  // P_perp matrix (n_spatial x n_spatial, row-major)
    int rsr_n,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  // Parameters: fixed effects + RE + spatial effects
  int n_x = p + n_re_groups + n_spatial_units;

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Indices
  int spatial_start = p + n_re_groups;

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute projected spatial effects: w_proj = P_perp * w
    NumericVector w_proj(n_spatial_units, 0.0);
    for (int i = 0; i < n_spatial_units; i++) {
      for (int j = 0; j < n_spatial_units; j++) {
        // P_perp is stored row-major
        w_proj[i] += rsr_projection[i * rsr_n + j] * x[spatial_start + j];
      }
    }

    // Compute linear predictor with projected spatial effects
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      // Fixed effects
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // RSR projected spatial effects
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;  // 1-based to 0-based
        if (s >= 0 && s < n_spatial_units) {
          eta[i] += w_proj[s];  // Use projected spatial effect
        }
      }
    }

    // Initialize gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions
    // The chain rule for RSR: d/dw = P_perp' * d/dw_proj = P_perp' * d/deta
    // (since P_perp is symmetric, P_perp' = P_perp)
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    // First compute gradient w.r.t. projected spatial effects
    std::vector<double> grad_w_proj(n_spatial_units, 0.0);
    std::vector<double> H_w_proj_diag(n_spatial_units, 0.0);

    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      // Fixed effects
      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }

      // Gradient w.r.t. projected spatial effect
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          grad_w_proj[s] += g_i;
          H_w_proj_diag[s] += h_i;
        }
      }
    }

    // Transform gradients and Hessian through RSR projection
    // grad_w = P_perp' * grad_w_proj (P_perp is symmetric)
    for (int i = 0; i < n_spatial_units; i++) {
      for (int j = 0; j < n_spatial_units; j++) {
        grad_local[spatial_start + i] += rsr_projection[i * rsr_n + j] * grad_w_proj[j];
      }
    }

    // Hessian for spatial: H_w = P_perp' * diag(H_w_proj) * P_perp
    // Since H_w_proj is diagonal, this becomes:
    // H_w[i,j] = sum_k P_perp[k,i] * H_w_proj[k] * P_perp[k,j]
    for (int i = 0; i < n_spatial_units; i++) {
      for (int j = 0; j <= i; j++) {
        double sum = 0.0;
        for (int k = 0; k < n_spatial_units; k++) {
          // P_perp is symmetric, so P_perp[k,i] = P_perp[i,k]
          sum += rsr_projection[k * rsr_n + i] * H_w_proj_diag[k] * rsr_projection[k * rsr_n + j];
        }
        H_local[spatial_start + i][spatial_start + j] = sum;
        if (i != j) {
          H_local[spatial_start + j][spatial_start + i] = sum;
        }
      }
    }

    // Cross-terms between fixed effects and projected spatial effects
    for (int i = 0; i < N; i++) {
      double h_i;
      if (family == "binomial") {
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          // Cross-terms with fixed effects through projection
          for (int j = 0; j < p; j++) {
            for (int k = 0; k < n_spatial_units; k++) {
              double P_ks = rsr_projection[s * rsr_n + k];  // P_perp[s,k]
              H_local[j][spatial_start + k] += h_i * X(i, j) * P_ks;
              H_local[spatial_start + k][j] += h_i * X(i, j) * P_ks;
            }
          }

          // Cross-terms with random effects through projection
          if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
              for (int k = 0; k < n_spatial_units; k++) {
                double P_ks = rsr_projection[s * rsr_n + k];
                H_local[p + g][spatial_start + k] += h_i * P_ks;
                H_local[spatial_start + k][p + g] += h_i * P_ks;
              }
            }
          }
        }
      }
    }

    // Copy to main gradient and Hessian
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Prior contributions
    // Random effects prior
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // ICAR prior on unprojected spatial effects
    for (int s = 0; s < n_spatial_units; s++) {
      int sp_idx = spatial_start + s;
      double phi_s = x[sp_idx];

      double neighbor_sum = 0.0;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        neighbor_sum += x[spatial_start + neighbor];
      }
      grad[sp_idx] -= tau_spatial * (n_neighbors[s] * phi_s - neighbor_sum);
      H(sp_idx, sp_idx) += tau_spatial * n_neighbors[s];

      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        int nb_idx = spatial_start + neighbor;
        H(sp_idx, nb_idx) -= tau_spatial;
      }
    }

    // Regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Forward substitution
    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Back substitution
    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update
    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  // Center spatial effects (soft sum-to-zero constraint)
  if (n_spatial_units > 0) {
    double mean_phi = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      mean_phi += x[spatial_start + s];
    }
    mean_phi /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      x[spatial_start + s] -= mean_phi;
    }
  }

  result.mode = x;

  // Compute log determinant and log marginal
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);

  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;

  // Compute final projected spatial effects
  NumericVector w_proj_final(n_spatial_units, 0.0);
  for (int i = 0; i < n_spatial_units; i++) {
    for (int j = 0; j < n_spatial_units; j++) {
      w_proj_final[i] += rsr_projection[i * rsr_n + j] * x[spatial_start + j];
    }
  }

  // Compute log likelihood at mode
  double log_lik = 0.0;
  NumericVector eta_final(N);
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (n_spatial_units > 0) {
      int s = spatial_idx[i] - 1;
      if (s >= 0 && s < n_spatial_units) {
        eta_final[i] += w_proj_final[s];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += log_lik_poisson(y[i], eta_final[i]);
    }
  }

  // Log prior for RE
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  // Log prior for spatial (ICAR)
  double log_prior_spatial = 0.0;
  if (n_spatial_units > 0) {
    double quad_form = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      double phi_s = x[spatial_start + s];
      quad_form += n_neighbors[s] * phi_s * phi_s;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        if (neighbor > s) {
          double phi_n = x[spatial_start + neighbor];
          quad_form -= 2.0 * phi_s * phi_n;
        }
      }
    }
    log_prior_spatial = -0.5 * tau_spatial * quad_form;
    log_prior_spatial += 0.5 * (n_spatial_units - 1) * std::log(tau_spatial / (2.0 * M_PI));
  }

  result.log_marginal = log_lik + log_prior_re + log_prior_spatial
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

} // namespace ratiod

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_rsr(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double tau_spatial,
    Rcpp::NumericVector rsr_projection,
    int rsr_n,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_rsr(
    y, n, X, re_idx, n_re_groups, sigma_re,
    spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors,
    tau_spatial, rsr_projection, rsr_n,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multiscale_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx_local,
    Rcpp::NumericMatrix nn_dist_local,
    Rcpp::IntegerVector nn_order_local,
    int nn_local,
    Rcpp::IntegerMatrix nn_idx_regional,
    Rcpp::NumericMatrix nn_dist_regional,
    Rcpp::IntegerVector nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local,
    double phi_local,
    double sigma2_regional,
    double phi_regional,
    int cov_type,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_multiscale_gp(
    y, n, X, re_idx, n_re_groups, sigma_re,
    coords, nn_idx_local, nn_dist_local, nn_order_local, nn_local,
    nn_idx_regional, nn_dist_regional, nn_order_regional, nn_regional,
    n_spatial,
    sigma2_local, phi_local, sigma2_regional, phi_regional,
    cov_type, family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multiscale_temporal(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = laplace_mode_multiscale_temporal(
    y, n, X, re_idx, n_re_groups, sigma_re,
    time_idx, n_times, seasonal_period, trend_type, short_type,
    sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}
