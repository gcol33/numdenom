// pg_binomial.cpp
// Pólya-Gamma Gibbs sampler for binomial models with random effects
// Based on Polson, Scott & Windle (2013) JASA

#include "pg_binomial.h"
#include "pg_rng.h"
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
// Helper: solve linear system with diagonal covariance
// For posterior: (X'WX + D^{-1})^{-1} X'W kappa
// where W = diag(omega), D = prior precision
// ---------------------------------------------------------------------

// Compute X'WX + prior_prec * I and X'W * (kappa - offset)
// Returns list with posterior mean and precision matrix
List compute_posterior_normal(
    const NumericMatrix& X,
    const NumericVector& omega,
    const NumericVector& kappa,
    const NumericVector& offset,
    double prior_prec
) {
  int n = X.nrow();
  int p = X.ncol();

  // X'WX + prior_prec * I
  NumericMatrix XWX(p, p);
  NumericVector XWkappa(p);

  for (int j = 0; j < p; j++) {
    for (int k = j; k < p; k++) {
      double sum = 0.0;
      for (int i = 0; i < n; i++) {
        sum += X(i, j) * omega[i] * X(i, k);
      }
      XWX(j, k) = sum;
      if (j != k) XWX(k, j) = sum;
    }
    // Add prior precision to diagonal
    XWX(j, j) += prior_prec;

    // X'W(kappa - offset)
    double sum_kappa = 0.0;
    for (int i = 0; i < n; i++) {
      sum_kappa += X(i, j) * omega[i] * (kappa[i] / omega[i] - offset[i]);
    }
    XWkappa[j] = sum_kappa;
  }

  return List::create(
    Named("XWX") = XWX,
    Named("XWkappa") = XWkappa
  );
}

// Cholesky solve: solve (L L') x = b
NumericVector chol_solve(const NumericMatrix& L, const NumericVector& b) {
  int p = L.ncol();
  NumericVector y(p), x(p);

  // Forward substitution: L y = b
  for (int i = 0; i < p; i++) {
    double sum = b[i];
    for (int j = 0; j < i; j++) {
      sum -= L(i, j) * y[j];
    }
    y[i] = sum / L(i, i);
  }

  // Back substitution: L' x = y
  for (int i = p - 1; i >= 0; i--) {
    double sum = y[i];
    for (int j = i + 1; j < p; j++) {
      sum -= L(j, i) * x[j];
    }
    x[i] = sum / L(i, i);
  }

  return x;
}

// Cholesky decomposition (lower triangular)
NumericMatrix chol_decomp(const NumericMatrix& A) {
  int n = A.nrow();
  NumericMatrix L(n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += L(i, k) * L(j, k);
      }
      if (i == j) {
        L(i, j) = std::sqrt(std::max(A(i, i) - sum, 1e-10));
      } else {
        L(i, j) = (A(i, j) - sum) / L(j, j);
      }
    }
  }
  return L;
}

// Sample from multivariate normal using Cholesky
NumericVector rmvnorm_chol(const NumericVector& mean, const NumericMatrix& L) {
  int p = mean.size();
  NumericVector z(p), x(p);

  // Sample standard normal
  for (int i = 0; i < p; i++) {
    z[i] = R::rnorm(0.0, 1.0);
  }

  // x = mean + L * z
  for (int i = 0; i < p; i++) {
    x[i] = mean[i];
    for (int j = 0; j <= i; j++) {
      x[i] += L(i, j) * z[j];
    }
  }

  return x;
}

// ---------------------------------------------------------------------
// Update functions
// ---------------------------------------------------------------------

// Update beta (fixed effects)
NumericVector update_beta(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericMatrix& X,
    const NumericVector& re_contrib,
    double prior_sd
) {
  int n = X.nrow();
  int p = X.ncol();
  double prior_prec = 1.0 / (prior_sd * prior_sd);

  // Compute posterior parameters
  List post = compute_posterior_normal(X, omega, kappa, re_contrib, prior_prec);
  NumericMatrix XWX = post["XWX"];
  NumericVector XWkappa = post["XWkappa"];

  // Cholesky decomposition
  NumericMatrix L = chol_decomp(XWX);

  // Posterior mean: solve XWX * mean = XWkappa
  NumericVector post_mean = chol_solve(L, XWkappa);

  // Sample from posterior
  // Need to sample from N(post_mean, XWX^{-1})
  // XWX^{-1} = (L L')^{-1} = L'^{-1} L^{-1}
  // So sample z ~ N(0, I), compute L^{-1} z, add to mean

  // More efficient: solve L z_star = z for z_star, then x = mean + z_star
  NumericVector z(p);
  for (int i = 0; i < p; i++) {
    z[i] = R::rnorm(0.0, 1.0);
  }

  // Solve L z_star = z (forward substitution only)
  NumericVector z_star(p);
  for (int i = 0; i < p; i++) {
    double sum = z[i];
    for (int j = 0; j < i; j++) {
      sum -= L(i, j) * z_star[j];
    }
    z_star[i] = sum / L(i, i);
  }

  // x = mean + z_star
  NumericVector beta(p);
  for (int i = 0; i < p; i++) {
    beta[i] = post_mean[i] + z_star[i];
  }

  return beta;
}

// Update random effects (blocked by group)
NumericVector update_re(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& X_beta,
    const IntegerVector& group,
    int n_groups,
    double sigma_re
) {
  int n = kappa.size();
  NumericVector re(n_groups);

  // For each group, compute posterior
  // Observations in group g: Y_g = X_beta_g + b_g + error
  // With PG augmentation: kappa_g/omega_g = X_beta_g + b_g + N(0, 1/omega_g)
  // Prior: b_g ~ N(0, sigma_re^2)
  //
  // Posterior: b_g | ... ~ N(m_g, v_g)
  // v_g = (sum(omega_g) + 1/sigma_re^2)^{-1}
  // m_g = v_g * sum(kappa_g - omega_g * X_beta_g)

  double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Accumulate sufficient statistics by group
  NumericVector sum_omega(n_groups);
  NumericVector sum_resid(n_groups);

  for (int i = 0; i < n; i++) {
    int g = group[i] - 1;  // Convert to 0-based
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * X_beta[i];
  }

  // Sample from posterior
  for (int g = 0; g < n_groups; g++) {
    double post_var = 1.0 / (sum_omega[g] + prior_prec);
    double post_mean = post_var * sum_resid[g];
    re[g] = R::rnorm(post_mean, std::sqrt(post_var));
  }

  return re;
}

// Update sigma_re with half-Cauchy prior
// p(sigma | b) ∝ p(b | sigma) * p(sigma)
// Using auxiliary variable approach for half-Cauchy
double update_sigma_re(
    const NumericVector& re,
    double scale
) {
  int J = re.size();

  // Sum of squared random effects
  double ss = 0.0;
  for (int j = 0; j < J; j++) {
    ss += re[j] * re[j];
  }

  // Half-Cauchy prior: sample via inverse-gamma auxiliary
  // sigma^2 | b, aux ~ IG((J+1)/2, ss/2 + 1/aux)
  // aux | sigma^2 ~ IG(1, 1/scale^2 + 1/sigma^2)

  // For simplicity, use direct sampling with truncated normal on log scale
  // or inverse gamma approximation

  // Simple approach: use conjugate inverse-gamma with weakly informative params
  // sigma^2 ~ IG(0.5, 0.5 * scale^2) approximately matches half-Cauchy for moderate scale

  // Posterior: sigma^2 | b ~ IG((J + 1)/2, ss/2 + 0.5 * scale^2)
  double shape = (J + 1.0) / 2.0;
  double rate = ss / 2.0 + 0.5 * scale * scale;

  double sigma_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
  return std::sqrt(sigma_sq);
}

// ---------------------------------------------------------------------
// Main Gibbs sampler
// ---------------------------------------------------------------------

List pg_binomial_gibbs_impl(
    IntegerVector y,
    IntegerVector n,
    NumericMatrix X,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_groups);
  NumericVector sigma_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Initialize
  NumericVector beta(p, 0.0);
  NumericVector re(n_groups, 0.0);
  double sigma_re = 1.0;
  NumericVector omega(N, 1.0);  // PG draws
  NumericVector kappa(N);       // y - n/2
  NumericVector eta(N);         // Linear predictor
  NumericVector X_beta(N);      // X * beta
  NumericVector re_contrib(N);  // Random effects contribution

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = re[group[i] - 1];  // group is 1-based
      eta[i] = X_beta[i] + re_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta) (parallelized)
    // Note: PG sampling is independent across observations
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      omega[i] = rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, y
    beta = update_beta(kappa, omega, X, re_contrib, prior_beta_sd);

    // 4. Recompute X_beta after beta update (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, sigma_re
    if (n_groups > 0) {
      re = update_re(kappa, omega, X_beta, group, n_groups, sigma_re);

      // 6. Update sigma_re | re
      sigma_re = update_sigma_re(re, prior_sigma_scale);
    }

    // Save draws (after warmup, respecting thinning)
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_draws[save_idx] = sigma_re;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  List result = List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("sigma_re") = sigma_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  return result;
}

} // namespace quotr

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_scale = 2.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  return quotr::pg_binomial_gibbs_impl(
    y, n, X, group, n_groups,
    n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_scale,
    store_eta, verbose, n_threads
  );
}

// [[Rcpp::export]]
int cpp_pg_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// Forward declaration
namespace quotr {
  Rcpp::NumericVector update_spatial_icar(
      const Rcpp::NumericVector& kappa,
      const Rcpp::NumericVector& omega,
      const Rcpp::NumericVector& offset,
      const Rcpp::IntegerVector& group,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      double tau
  );

  double update_tau_icar(
      const Rcpp::NumericVector& phi,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      double prior_shape,
      double prior_rate
  );
}

// Binomial Gibbs sampler with random effects AND spatial effects (ICAR)
// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_spatial(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_tau_shape = 1.0,
    double prior_tau_rate = 0.01,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix spatial_draws(n_save, n_spatial_units);
  Rcpp::NumericVector tau_draws(n_save);
  Rcpp::NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = Rcpp::NumericMatrix(n_save, N);
  }

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi(n_spatial_units, 0.0);  // Spatial effects
  double tau = 1.0;  // Spatial precision
  Rcpp::NumericVector omega(N, 1.0);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector spatial_contrib(N);
  Rcpp::NumericVector offset(N);

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
      spatial_contrib[i] = phi[spatial_group[i] - 1];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta) (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      omega[i] = quotr::rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, phi, y
    // Offset for beta update = re + phi (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = quotr::update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, phi, sigma_re
    if (n_re_groups > 0) {
      // Offset for RE update = X*beta + phi (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + spatial_contrib[i];
      }
      re = quotr::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);

      // Update sigma_re
      sigma_re = quotr::update_sigma_re(re, prior_sigma_re_scale);

      // Update re_contrib (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 6. Update spatial effects | omega, beta, re, tau
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }
    phi = quotr::update_spatial_icar(kappa, omega, offset, spatial_group, adj_list, n_neighbors, tau);

    // 7. Update tau (spatial precision)
    tau = quotr::update_tau_icar(phi, adj_list, n_neighbors, prior_tau_shape, prior_tau_rate);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_draws(save_idx, s) = phi[s];
      }
      tau_draws[save_idx] = tau;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("spatial") = spatial_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  return result;
}

// ---------------------------------------------------------------------
// BYM2 Spatial Gibbs sampler
// ---------------------------------------------------------------------

// Forward declaration for BYM2 functions
namespace quotr {
  Rcpp::NumericVector update_spatial_bym2(
      const Rcpp::NumericVector& kappa,
      const Rcpp::NumericVector& omega,
      const Rcpp::NumericVector& offset,
      const Rcpp::IntegerVector& group,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      Rcpp::NumericVector& phi_scaled,
      Rcpp::NumericVector& theta,
      double sigma_spatial,
      double rho,
      double scale_factor
  );

  double update_sigma_spatial(
      const Rcpp::NumericVector& u,
      double scale
  );

  double update_rho_bym2(
      const Rcpp::NumericVector& phi_scaled,
      const Rcpp::NumericVector& theta,
      double sigma_spatial,
      double scale_factor,
      const Rcpp::NumericVector& sum_omega,
      const Rcpp::NumericVector& sum_resid,
      double alpha,
      double beta
  );
}

// Binomial Gibbs sampler with random effects AND spatial effects (BYM2)
// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_bym2(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_spatial_scale = 2.5,
    double prior_rho_alpha = 0.5,
    double prior_rho_beta = 0.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix phi_scaled_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix theta_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix u_draws(n_save, n_spatial_units);  // Combined spatial effect
  Rcpp::NumericVector sigma_spatial_draws(n_save);
  Rcpp::NumericVector rho_draws(n_save);
  Rcpp::NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = Rcpp::NumericMatrix(n_save, N);
  }

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi_scaled(n_spatial_units, 0.0);  // Structured component
  Rcpp::NumericVector theta(n_spatial_units, 0.0);       // Unstructured component
  Rcpp::NumericVector u(n_spatial_units, 0.0);           // Combined effect
  double sigma_spatial = 1.0;
  double rho = 0.5;  // Start with equal mix
  Rcpp::NumericVector omega(N, 1.0);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector spatial_contrib(N);
  Rcpp::NumericVector offset(N);

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
      spatial_contrib[i] = u[spatial_group[i] - 1];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta) (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      omega[i] = quotr::rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, u, y
    // Offset for beta update = re + u (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = quotr::update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, u, sigma_re
    if (n_re_groups > 0) {
      // Offset for RE update = X*beta + u (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + spatial_contrib[i];
      }
      re = quotr::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);

      // Update sigma_re
      sigma_re = quotr::update_sigma_re(re, prior_sigma_re_scale);

      // Update re_contrib (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 6. Update BYM2 spatial effects | omega, beta, re, sigma_spatial, rho
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }
    u = quotr::update_spatial_bym2(kappa, omega, offset, spatial_group, adj_list, n_neighbors,
                                   phi_scaled, theta, sigma_spatial, rho, scale_factor);

    // 7. Update sigma_spatial
    sigma_spatial = quotr::update_sigma_spatial(u, prior_sigma_spatial_scale);

    // 8. Update rho (mixing proportion)
    // Need to compute sum_omega and sum_resid for rho update
    Rcpp::NumericVector sum_omega_s(n_spatial_units, 0.0);
    Rcpp::NumericVector sum_resid_s(n_spatial_units, 0.0);
    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      sum_omega_s[s] += omega[i];
      sum_resid_s[s] += kappa[i] - omega[i] * offset[i];
    }
    rho = quotr::update_rho_bym2(phi_scaled, theta, sigma_spatial, scale_factor,
                                  sum_omega_s, sum_resid_s, prior_rho_alpha, prior_rho_beta);

    // Update spatial contributions (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int s = 0; s < n_spatial_units; s++) {
        phi_scaled_draws(save_idx, s) = phi_scaled[s];
        theta_draws(save_idx, s) = theta[s];
        u_draws(save_idx, s) = u[s];
      }
      sigma_spatial_draws[save_idx] = sigma_spatial;
      rho_draws[save_idx] = rho;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("phi_scaled") = phi_scaled_draws,
    Rcpp::Named("theta") = theta_draws,
    Rcpp::Named("spatial") = u_draws,
    Rcpp::Named("sigma_spatial") = sigma_spatial_draws,
    Rcpp::Named("rho") = rho_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  return result;
}
