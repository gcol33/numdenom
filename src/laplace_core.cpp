// laplace_core.cpp
// Core Laplace approximation engine for quotr
// Implements nested Laplace approximation for latent Gaussian models

#include "laplace_core.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace quotr {

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
  double mu = std::exp(eta);
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
  double mu = std::exp(eta);
  double p = mu / (mu + phi);
  return y - (y + phi) * p;
}

// Negative Hessian: (y + phi) * p * (1-p) * mu (chain rule factor)
// Actually: -d²/d(eta)² = (y + phi) * mu * phi / (mu + phi)²
double neg_hess_log_lik_negbin(int y, double eta, double phi) {
  double mu = std::exp(eta);
  double denom = mu + phi;
  return (y + phi) * mu * phi / (denom * denom);
}

// Log-likelihood for Poisson: y ~ Poisson(mu = exp(eta))
double log_lik_poisson(int y, double eta) {
  // log p = y * eta - exp(eta) - lgamma(y+1)
  return y * eta - std::exp(eta) - R::lgammafn(y + 1.0);
}

// Gradient: d/d(eta) log p = y - exp(eta)
double grad_log_lik_poisson(int y, double eta) {
  return y - std::exp(eta);
}

// Negative Hessian: exp(eta)
double neg_hess_log_lik_poisson(int y, double eta) {
  return std::exp(eta);
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

    // Update x
    double step_size = 1.0;
    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
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
        int neighbor = adj_col_idx[k] - 1;  // 1-based to 0-based
        neighbor_sum += x[spatial_start + neighbor];
      }
      grad[sp_idx] -= tau_spatial * (n_neighbors[s] * phi_s - neighbor_sum);

      // Hessian diagonal
      H(sp_idx, sp_idx) += tau_spatial * n_neighbors[s];

      // Hessian off-diagonal
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k] - 1;
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
        int neighbor = adj_col_idx[k] - 1;
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
        int neighbor = adj_col_idx[k] - 1;
        neighbor_sum += x[phi_start + neighbor];
      }
      // ICAR with precision 1 (variance scaling handled by sigma_spatial)
      grad[phi_idx] -= (n_neighbors[s] * phi_s - neighbor_sum);
      H(phi_idx, phi_idx) += n_neighbors[s];
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k] - 1;
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
      int neighbor = adj_col_idx[k] - 1;
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

} // namespace quotr

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
  quotr::LaplaceResult result = quotr::laplace_mode_dense(
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
  quotr::LaplaceResult result = quotr::laplace_mode_spatial(
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
  quotr::LaplaceResult result = quotr::laplace_mode_bym2(
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
