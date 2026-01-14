// hmc_sampler.cpp
// Full HMC/NUTS backend with spatial, temporal, and ZI support
// Provides Stan-free Bayesian inference for all ratiod models

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include "hmc_progress.h"
#include "autodiff.h"
#include "autodiff_utils.h"
#include <Rcpp.h>

// Include log_post_impl.h AFTER hmc_sampler.h so types are defined
#include "log_post_impl.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace ratiod_hmc {

// =====================================================================
// Parameter layout computation
// =====================================================================

ParamLayout compute_param_layout(const ModelData& data) {
  ParamLayout layout;
  int idx = 0;

  // Fixed effects numerator
  layout.beta_num_start = idx;
  idx += data.p_num;
  layout.beta_num_end = idx;

  // Fixed effects denominator
  layout.beta_denom_start = idx;
  idx += data.p_denom;
  layout.beta_denom_end = idx;

  // Random effects (supports multiple crossed RE terms with slopes)
  layout.has_re = (data.n_re_groups > 0 || data.total_re_groups > 0);
  layout.has_re_slopes = data.has_re_slopes;
  layout.has_re_correlated_slopes = data.has_re_correlated_slopes;

  if (data.has_re_slopes && data.n_re_terms > 0) {
    // Random slopes case: need sigma per coefficient type + Cholesky params + RE effects
    int n_terms = data.n_re_terms;

    layout.log_sigma_re_multi.resize(n_terms);
    layout.log_sigma_re_slopes.resize(n_terms);
    layout.re_start_multi.resize(n_terms);
    layout.re_end_multi.resize(n_terms);
    layout.re_n_coefs_multi.resize(n_terms);
    layout.re_correlated_multi.resize(n_terms);
    layout.chol_re_start_multi.resize(n_terms);
    layout.chol_re_end_multi.resize(n_terms);

    // First pass: allocate sigma parameters for each term
    for (int t = 0; t < n_terms; t++) {
      int n_coefs = data.re_n_coefs[t];
      layout.re_n_coefs_multi[t] = n_coefs;
      layout.re_correlated_multi[t] = data.re_correlated[t];

      // Allocate log_sigma for each coefficient type (intercept, slopes)
      layout.log_sigma_re_slopes[t].resize(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        layout.log_sigma_re_slopes[t][c] = idx++;
      }
      // Legacy: point to first sigma for backwards compat
      layout.log_sigma_re_multi[t] = layout.log_sigma_re_slopes[t][0];
    }

    // Second pass: allocate Cholesky parameters for correlated terms
    for (int t = 0; t < n_terms; t++) {
      int n_chol = data.re_n_chol[t];  // k*(k-1)/2 for correlated, 0 otherwise
      if (n_chol > 0) {
        layout.chol_re_start_multi[t] = idx;
        idx += n_chol;
        layout.chol_re_end_multi[t] = idx;
      } else {
        layout.chol_re_start_multi[t] = -1;
        layout.chol_re_end_multi[t] = -1;
      }
    }

    // Third pass: allocate RE effects for each term
    for (int t = 0; t < n_terms; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = data.re_n_coefs[t];

      layout.re_start_multi[t] = idx;
      idx += n_groups * n_coefs;  // Each group has n_coefs parameters
      layout.re_end_multi[t] = idx;
    }

    // Legacy fields: point to first term
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end = layout.re_end_multi[0];

  } else if (data.n_re_terms > 1) {
    // Multiple RE terms (intercept only): allocate sigma and RE for each term
    layout.log_sigma_re_multi.resize(data.n_re_terms);
    layout.re_start_multi.resize(data.n_re_terms);
    layout.re_end_multi.resize(data.n_re_terms);
    layout.re_n_coefs_multi.resize(data.n_re_terms, 1);  // All intercept-only
    layout.re_correlated_multi.resize(data.n_re_terms, false);
    layout.chol_re_start_multi.resize(data.n_re_terms, -1);
    layout.chol_re_end_multi.resize(data.n_re_terms, -1);

    for (int t = 0; t < data.n_re_terms; t++) {
      layout.log_sigma_re_multi[t] = idx++;
    }
    for (int t = 0; t < data.n_re_terms; t++) {
      layout.re_start_multi[t] = idx;
      idx += data.re_n_groups_multi[t];
      layout.re_end_multi[t] = idx;
    }

    // Set legacy fields to first term for backwards compatibility
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end = layout.re_end_multi[0];
  } else if (layout.has_re) {
    // Single RE term (intercept only)
    layout.log_sigma_re_idx = idx++;
    layout.re_start = idx;
    idx += data.n_re_groups;
    layout.re_end = idx;

    // Also set multi arrays for consistency
    layout.log_sigma_re_multi.resize(1);
    layout.log_sigma_re_multi[0] = layout.log_sigma_re_idx;
    layout.re_start_multi.resize(1);
    layout.re_start_multi[0] = layout.re_start;
    layout.re_end_multi.resize(1);
    layout.re_end_multi[0] = layout.re_end;
    layout.re_n_coefs_multi.resize(1, 1);
    layout.re_correlated_multi.resize(1, false);
    layout.chol_re_start_multi.resize(1, -1);
    layout.chol_re_end_multi.resize(1, -1);
  } else {
    layout.log_sigma_re_idx = -1;
    layout.re_start = layout.re_end = -1;
  }

  // Overdispersion
  layout.has_phi_num = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                        data.model_type == ModelType::POISSON_GAMMA);
  layout.has_phi_denom = (data.model_type == ModelType::NEGBIN_NEGBIN);

  if (layout.has_phi_num) {
    layout.log_phi_num_idx = idx++;
  } else {
    layout.log_phi_num_idx = -1;
  }
  if (layout.has_phi_denom) {
    layout.log_phi_denom_idx = idx++;
  } else {
    layout.log_phi_denom_idx = -1;
  }

  // Spatial effects (ICAR/BYM2 only - GP handled separately below)
  layout.has_spatial = (data.spatial_type == SpatialType::ICAR ||
                        data.spatial_type == SpatialType::BYM2);
  layout.is_bym2 = (data.spatial_type == SpatialType::BYM2);

  if (layout.has_spatial) {
    if (layout.is_bym2) {
      // BYM2: log_sigma, logit_rho, phi_scaled, theta
      layout.log_sigma_bym2_idx = idx++;
      layout.logit_rho_bym2_idx = idx++;
      layout.spatial_start = idx;
      idx += data.n_spatial_units;  // phi_scaled (structured)
      layout.spatial_end = idx;
      layout.theta_bym2_start = idx;
      idx += data.n_spatial_units;  // theta (unstructured)
      layout.theta_bym2_end = idx;
      layout.log_tau_spatial_idx = -1;
    } else {
      // ICAR: log_tau, phi
      layout.log_tau_spatial_idx = idx++;
      layout.spatial_start = idx;
      idx += data.n_spatial_units;
      layout.spatial_end = idx;
      layout.log_sigma_bym2_idx = -1;
      layout.logit_rho_bym2_idx = -1;
      layout.theta_bym2_start = layout.theta_bym2_end = -1;
    }
  } else {
    layout.log_tau_spatial_idx = -1;
    layout.spatial_start = layout.spatial_end = -1;
    layout.log_sigma_bym2_idx = -1;
    layout.logit_rho_bym2_idx = -1;
    layout.theta_bym2_start = layout.theta_bym2_end = -1;
  }

  // Temporal effects
  layout.has_temporal = (data.temporal_type != TemporalType::NONE);
  layout.is_ar1 = (data.temporal_type == TemporalType::AR1);

  if (layout.has_temporal) {
    // log_tau for temporal precision
    layout.log_tau_temporal_idx = idx++;

    // AR1 also has rho parameter
    if (layout.is_ar1) {
      layout.logit_rho_ar1_idx = idx++;
    } else {
      layout.logit_rho_ar1_idx = -1;
    }

    // Temporal effects: n_times * n_groups parameters
    layout.temporal_start = idx;
    idx += data.n_temporal_params;
    layout.temporal_end = idx;
  } else {
    layout.log_tau_temporal_idx = -1;
    layout.logit_rho_ar1_idx = -1;
    layout.temporal_start = layout.temporal_end = -1;
  }

  // Zero-inflation parameters
  layout.has_zi = (data.zi_type != ZIType::NONE);

  if (layout.has_zi) {
    layout.beta_zi_start = idx;
    idx += data.p_zi;
    layout.beta_zi_end = idx;
  } else {
    layout.beta_zi_start = layout.beta_zi_end = -1;
  }

  // GP spatial parameters
  layout.is_gp = (data.spatial_type == SpatialType::GP);
  layout.is_multiscale_gp = (data.spatial_type == SpatialType::MULTISCALE_GP);

  if (layout.is_gp && data.has_gp) {
    layout.log_sigma2_gp_idx = idx++;
    layout.log_phi_gp_idx = idx++;
    layout.gp_w_start = idx;
    idx += data.gp_data.n_obs;
    layout.gp_w_end = idx;
  } else {
    layout.log_sigma2_gp_idx = -1;
    layout.log_phi_gp_idx = -1;
    layout.gp_w_start = layout.gp_w_end = -1;
  }

  // Multi-scale GP parameters
  if (layout.is_multiscale_gp && data.has_multiscale_gp) {
    // Local scale
    layout.log_sigma2_gp_local_idx = idx++;
    layout.log_phi_gp_local_idx = idx++;
    layout.gp_local_start = idx;
    idx += data.multiscale_gp_data.n_obs;
    layout.gp_local_end = idx;

    // Regional scale
    layout.log_sigma2_gp_regional_idx = idx++;
    layout.log_phi_gp_regional_idx = idx++;
    layout.gp_regional_start = idx;
    idx += data.multiscale_gp_data.n_obs;
    layout.gp_regional_end = idx;
  } else {
    layout.log_sigma2_gp_local_idx = -1;
    layout.log_phi_gp_local_idx = -1;
    layout.gp_local_start = layout.gp_local_end = -1;
    layout.log_sigma2_gp_regional_idx = -1;
    layout.log_phi_gp_regional_idx = -1;
    layout.gp_regional_start = layout.gp_regional_end = -1;
  }

  // Multi-scale temporal parameters
  layout.has_multiscale_temporal = data.has_multiscale_temporal;

  if (layout.has_multiscale_temporal) {
    // Trend component
    if (data.multiscale_temporal_data.trend_type != ratiod_temporal::TemporalType::NONE) {
      layout.log_sigma2_trend_idx = idx++;
      layout.trend_start = idx;
      idx += data.multiscale_temporal_data.n_times;
      layout.trend_end = idx;
    } else {
      layout.log_sigma2_trend_idx = -1;
      layout.trend_start = layout.trend_end = -1;
    }

    // Seasonal component
    if (data.multiscale_temporal_data.seasonal_period > 0) {
      layout.log_sigma2_seasonal_idx = idx++;
      layout.seasonal_start = idx;
      idx += data.multiscale_temporal_data.seasonal_period;
      layout.seasonal_end = idx;
    } else {
      layout.log_sigma2_seasonal_idx = -1;
      layout.seasonal_start = layout.seasonal_end = -1;
    }

    // Short-term component
    if (data.multiscale_temporal_data.short_term_type != ratiod_temporal::TemporalType::NONE) {
      layout.log_sigma2_short_idx = idx++;
      if (data.multiscale_temporal_data.short_term_type == ratiod_temporal::TemporalType::AR1) {
        layout.logit_rho_short_idx = idx++;
      } else {
        layout.logit_rho_short_idx = -1;
      }
      layout.short_term_start = idx;
      idx += data.multiscale_temporal_data.n_times;
      layout.short_term_end = idx;
    } else {
      layout.log_sigma2_short_idx = -1;
      layout.logit_rho_short_idx = -1;
      layout.short_term_start = layout.short_term_end = -1;
    }
  } else {
    layout.log_sigma2_trend_idx = -1;
    layout.trend_start = layout.trend_end = -1;
    layout.log_sigma2_seasonal_idx = -1;
    layout.seasonal_start = layout.seasonal_end = -1;
    layout.log_sigma2_short_idx = -1;
    layout.logit_rho_short_idx = -1;
    layout.short_term_start = layout.short_term_end = -1;
  }

  // SVC not yet integrated into GP interface
  layout.has_svc = data.has_svc;
  layout.log_sigma2_svc_start = layout.log_sigma2_svc_end = -1;
  layout.log_phi_svc_start = layout.log_phi_svc_end = -1;
  layout.svc_w_start = layout.svc_w_end = -1;

  // Latent factors for unmeasured confounders
  layout.has_latent = data.has_latent;
  if (layout.has_latent && data.latent_n_factors > 0) {
    // Log sigma for each factor
    layout.log_sigma_latent_start = idx;
    idx += data.latent_n_factors;
    layout.log_sigma_latent_end = idx;

    // Factor scores (N x K)
    layout.latent_factor_start = idx;
    idx += data.N * data.latent_n_factors;
    layout.latent_factor_end = idx;
  } else {
    layout.log_sigma_latent_start = layout.log_sigma_latent_end = -1;
    layout.latent_factor_start = layout.latent_factor_end = -1;
  }

  // Spatiotemporal interaction
  layout.has_spatiotemporal = data.has_spatiotemporal;
  layout.is_st_gp = (data.has_spatiotemporal &&
                     (data.spatiotemporal_data.type == STType::SEPARABLE ||
                      data.spatiotemporal_data.type == STType::NONSEP_GP));

  if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
    // log_tau for interaction precision
    layout.log_tau_st_idx = idx++;

    // Second precision for Type IV (Kronecker)
    if (data.spatiotemporal_data.type == STType::TYPE_IV) {
      layout.log_tau_st2_idx = idx++;
    } else {
      layout.log_tau_st2_idx = -1;
    }

    // AR1 rho if temporal uses AR1
    if (data.spatiotemporal_data.temporal_type == TemporalType::AR1) {
      layout.logit_rho_st_idx = idx++;
    } else {
      layout.logit_rho_st_idx = -1;
    }

    // GP range parameters (for separable/non-separable GP)
    if (layout.is_st_gp) {
      layout.log_phi_st_space_idx = idx++;
      layout.log_phi_st_time_idx = idx++;
    } else {
      layout.log_phi_st_space_idx = -1;
      layout.log_phi_st_time_idx = -1;
    }

    // Spatiotemporal interaction effects
    layout.st_delta_start = idx;
    idx += data.spatiotemporal_data.n_params;
    layout.st_delta_end = idx;
  } else {
    layout.log_tau_st_idx = -1;
    layout.log_tau_st2_idx = -1;
    layout.logit_rho_st_idx = -1;
    layout.log_phi_st_space_idx = -1;
    layout.log_phi_st_time_idx = -1;
    layout.st_delta_start = layout.st_delta_end = -1;
  }

  layout.total_params = idx;
  return layout;
}

int get_n_params(const ModelData& data) {
  ParamLayout layout = compute_param_layout(data);
  return layout.total_params;
}

// =====================================================================
// Likelihood functions
// =====================================================================

inline double log_lik_binomial(int y, int n, double eta) {
  // Numerically stable binomial log-likelihood
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    return y * eta - n * std::log(1.0 + std::exp(eta));
  }
}

inline double log_lik_negbin(int y, double mu, double phi) {
  if (mu <= 0 || phi <= 0) return -1e10;
  return R::lgammafn(y + phi) - R::lgammafn(phi) - R::lgammafn(y + 1.0)
       + phi * std::log(phi / (mu + phi))
       + y * std::log(mu / (mu + phi));
}

inline double log_lik_poisson(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - R::lgammafn(y + 1.0);
}

inline double log_lik_gamma(double y, double shape, double mu) {
  if (y <= 0 || shape <= 0 || mu <= 0) return -1e10;
  double rate = shape / mu;
  return shape * std::log(rate) + (shape - 1.0) * std::log(y)
       - rate * y - R::lgammafn(shape);
}

// =====================================================================
// ICAR quadratic form: phi' Q phi
// =====================================================================

double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
) {
  double quad_form = 0.0;
  int J = data.n_spatial_units;

  for (int i = 0; i < J; i++) {
    // Diagonal: n_i * phi_i^2
    quad_form += data.n_neighbors[i] * phi[i] * phi[i];

    // Off-diagonal: -2 * sum over neighbors (count each edge once)
    int row_start = data.adj_row_ptr[i];
    int row_end = data.adj_row_ptr[i + 1];
    for (int k = row_start; k < row_end; k++) {
      int j = data.adj_col_idx[k];
      if (j > i) {  // Count each edge once
        quad_form -= 2.0 * phi[i] * phi[j];
      }
    }
  }

  return quad_form;
}

// =====================================================================
// Log-posterior computation with OpenMP parallelization
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  // Extract parameters
  const double* beta_num = &params[layout.beta_num_start];
  const double* beta_denom = &params[layout.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, phi_denom = 1.0;
  double log_phi_num = 0.0, log_phi_denom = 0.0;
  if (layout.has_phi_num) {
    log_phi_num = params[layout.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.has_phi_denom) {
    log_phi_denom = params[layout.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
  }

  // Spatial parameters
  double tau_spatial = 1.0, log_tau_spatial = 0.0;
  double sigma_bym2 = 1.0, rho_bym2 = 0.5;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;

  if (layout.has_spatial) {
    phi_spatial = &params[layout.spatial_start];
    if (layout.is_bym2) {
      sigma_bym2 = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      theta_bym2 = &params[layout.theta_bym2_start];
    } else {
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);
    }
  }

  double log_post = 0.0;

  // ============ PRIORS ============

  // Fixed effects: N(0, sigma_beta^2)
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.p_num; j++) {
    log_post -= 0.5 * tau_beta * beta_num[j] * beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    log_post -= 0.5 * tau_beta * beta_denom[j] * beta_denom[j];
  }

  // Random effects priors (supports multiple crossed RE terms with slopes)
  if (layout.has_re) {
    int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      int n_groups = (n_terms > 1 || data.n_re_terms > 0) ? data.re_n_groups_multi[t] : data.n_re_groups;
      int n_coefs = layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
      bool is_correlated = layout.has_re_slopes && layout.re_correlated_multi[t];

      // Extract sigma parameters for this term
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx;
        if (layout.has_re_slopes) {
          log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        } else if (n_terms > 1) {
          log_sigma_idx = layout.log_sigma_re_multi[t];
        } else {
          log_sigma_idx = layout.log_sigma_re_idx;
        }
        double log_sigma = params[log_sigma_idx];
        sigmas[c] = std::exp(log_sigma);

        // Half-Cauchy(0, scale) prior for each sigma
        double ratio = sigmas[c] / data.sigma_re_scale;
        log_post -= std::log(1.0 + ratio * ratio);
        log_post += log_sigma;  // Jacobian
      }

      // For correlated slopes: LKJ prior on correlation matrix via Cholesky
      // Parameterization: Sigma = diag(sigma) * L * L' * diag(sigma)
      // where L is lower-triangular with unit diagonal
      std::vector<double> L_flat;  // Lower triangular Cholesky factor (full, including diagonal)
      if (is_correlated && n_coefs > 1) {
        int chol_start = layout.chol_re_start_multi[t];
        int n_chol = n_coefs * (n_coefs - 1) / 2;

        // Build L matrix: diagonal = 1, off-diagonal from params
        // We use a spherical parameterization for numerical stability
        // L[i,i] = sqrt(1 - sum(L[i,j]^2 for j<i)) for i>0, L[0,0] = 1
        L_flat.resize(n_coefs * n_coefs, 0.0);

        int chol_idx = 0;
        for (int i = 0; i < n_coefs; i++) {
          double row_sum_sq = 0.0;
          for (int j = 0; j < i; j++) {
            // Off-diagonal elements: unconstrained parameters
            double l_ij = params[chol_start + chol_idx];
            // Transform to ensure positive definiteness: use tanh to bound
            // Actually, for LKJ we use a different parameterization
            // Use partial correlations / spherical coordinates
            L_flat[i * n_coefs + j] = l_ij;
            row_sum_sq += l_ij * l_ij;
            chol_idx++;
          }
          // Diagonal: ensure positive definiteness
          // L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2), but need to handle numerically
          double diag_sq = 1.0 - row_sum_sq;
          if (diag_sq < 1e-10) {
            // Invalid: correlation matrix not positive definite
            return -std::numeric_limits<double>::infinity();
          }
          L_flat[i * n_coefs + i] = std::sqrt(diag_sq);
        }

        // LKJ(eta) prior: p(L) propto det(L*L')^(eta-1)
        // For eta=1 (uniform): log_prior = 0
        // For eta=2 (weakly informative): log_prior = sum_k (n_coefs - k) * log(L[k,k])
        double eta = 2.0;  // LKJ concentration parameter (2 = weakly informative)
        for (int k = 0; k < n_coefs; k++) {
          double L_kk = L_flat[k * n_coefs + k];
          log_post += (eta - 1.0 + (n_coefs - k - 1) / 2.0) * 2.0 * std::log(L_kk);
        }

        // Jacobian for the transformation from Cholesky to correlation
        for (int k = 1; k < n_coefs; k++) {
          double L_kk = L_flat[k * n_coefs + k];
          log_post += (n_coefs - k) * std::log(L_kk);
        }
      }

      // Get RE parameters for this term
      int re_start = (n_terms > 1 || layout.has_re_slopes) ? layout.re_start_multi[t] : layout.re_start;

      // Random effects prior: depends on correlation structure
      if (is_correlated && n_coefs > 1) {
        // Multivariate normal with covariance Sigma = diag(sigma) * L * L' * diag(sigma)
        // For efficiency, we work with standardized RE: z = diag(1/sigma) * L^{-1} * re
        // and put N(0, I) prior on z

        for (int g = 0; g < n_groups; g++) {
          // Extract RE vector for this group
          std::vector<double> re_g(n_coefs);
          for (int c = 0; c < n_coefs; c++) {
            re_g[c] = params[re_start + g * n_coefs + c];
          }

          // Compute z = L^{-1} * diag(1/sigma) * re using forward substitution
          std::vector<double> scaled_re(n_coefs);
          for (int c = 0; c < n_coefs; c++) {
            scaled_re[c] = re_g[c] / sigmas[c];
          }

          std::vector<double> z(n_coefs);
          for (int i = 0; i < n_coefs; i++) {
            double sum = scaled_re[i];
            for (int j = 0; j < i; j++) {
              sum -= L_flat[i * n_coefs + j] * z[j];
            }
            z[i] = sum / L_flat[i * n_coefs + i];
          }

          // N(0, I) prior on z
          for (int c = 0; c < n_coefs; c++) {
            log_post -= 0.5 * z[c] * z[c];
          }
        }

        // Log-determinant contribution: |Sigma|^{-n_groups/2}
        // |Sigma| = prod(sigma_c^2) * |L|^2 = prod(sigma_c^2) * prod(L_kk)^2
        double log_det = 0.0;
        for (int c = 0; c < n_coefs; c++) {
          log_det += 2.0 * std::log(sigmas[c]);
        }
        for (int k = 0; k < n_coefs; k++) {
          log_det += 2.0 * std::log(L_flat[k * n_coefs + k]);
        }
        log_post -= 0.5 * n_groups * log_det;
        log_post -= 0.5 * n_groups * n_coefs * std::log(2.0 * M_PI);

      } else {
        // Uncorrelated case: independent N(0, sigma_c^2) for each coefficient type
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double re_val = params[re_start + g * n_coefs + c];
            double tau_re = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
            log_post -= 0.5 * tau_re * re_val * re_val;
            log_post += 0.5 * std::log(tau_re);
          }
        }
        log_post -= 0.5 * n_groups * n_coefs * std::log(2.0 * M_PI);
      }
    }
  }

  // Overdispersion: Gamma prior
  if (layout.has_phi_num) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_num
              - data.phi_prior_rate * phi_num + log_phi_num;
  }
  if (layout.has_phi_denom) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_denom
              - data.phi_prior_rate * phi_denom + log_phi_denom;
  }

  // Spatial priors
  if (layout.has_spatial) {
    int J = data.n_spatial_units;

    if (layout.is_bym2) {
      // BYM2 prior
      // sigma ~ Half-Cauchy (Jacobian for log transform)
      double sigma_ratio = sigma_bym2 / data.sigma_re_scale;
      log_post -= std::log(1.0 + sigma_ratio * sigma_ratio);
      log_post += params[layout.log_sigma_bym2_idx];

      // rho ~ Beta(0.5, 0.5) (Jacobian for logit transform)
      log_post += -0.5 * std::log(rho_bym2) - 0.5 * std::log(1.0 - rho_bym2);
      log_post += std::log(rho_bym2) + std::log(1.0 - rho_bym2);  // Jacobian

      // phi_scaled ~ N(0, Q^{-1}) with soft sum-to-zero
      std::vector<double> phi_vec(phi_spatial, phi_spatial + J);
      double quad = icar_quadratic_form(phi_vec, data);
      log_post -= 0.5 * quad;

      // theta ~ N(0, I)
      for (int j = 0; j < J; j++) {
        log_post -= 0.5 * theta_bym2[j] * theta_bym2[j];
      }
    } else {
      // ICAR prior
      // tau ~ Gamma(shape, rate)
      log_post += (data.tau_spatial_shape - 1.0) * log_tau_spatial
                - data.tau_spatial_rate * tau_spatial + log_tau_spatial;

      // phi ~ ICAR(tau): p(phi|tau) propto tau^{(J-1)/2} exp(-0.5 * tau * phi'Qphi)
      std::vector<double> phi_vec(phi_spatial, phi_spatial + J);
      double quad = icar_quadratic_form(phi_vec, data);
      log_post += 0.5 * (J - 1) * log_tau_spatial - 0.5 * tau_spatial * quad;
    }
  }

  // ZI coefficient priors
  const double* beta_zi = nullptr;
  if (layout.has_zi) {
    beta_zi = &params[layout.beta_zi_start];
    // N(0, zi_prior_sd^2) prior on ZI coefficients
    double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    for (int j = 0; j < data.p_zi; j++) {
      log_post -= 0.5 * tau_zi * beta_zi[j] * beta_zi[j];
    }
  }

  // Temporal priors
  double tau_temporal = 1.0, log_tau_temporal = 0.0;
  double rho_ar1 = 0.5;
  const double* phi_temporal = nullptr;

  if (layout.has_temporal) {
    log_tau_temporal = params[layout.log_tau_temporal_idx];
    tau_temporal = std::exp(log_tau_temporal);
    phi_temporal = &params[layout.temporal_start];

    // tau ~ Gamma(shape, rate) with Jacobian
    log_post += (data.tau_temporal_shape - 1.0) * log_tau_temporal
              - data.tau_temporal_rate * tau_temporal + log_tau_temporal;

    // AR1: also estimate rho
    if (layout.is_ar1) {
      double logit_rho = params[layout.logit_rho_ar1_idx];
      rho_ar1 = 1.0 / (1.0 + std::exp(-logit_rho));

      // rho ~ Uniform(-1, 1) via transformed prior
      // On logit((rho+1)/2) scale, Jacobian is rho*(1-rho)/2
      // Simplified: uniform on rho -> Jacobian for logit transform
      log_post += std::log(rho_ar1 + 1.0) + std::log(1.0 - rho_ar1);
    }

    // Temporal effects prior (per group)
    int T = data.n_times;
    for (int g = 0; g < data.n_temporal_groups; g++) {
      const double* phi_g = phi_temporal + g * T;

      if (data.temporal_type == TemporalType::RW1) {
        double quad = ratiod_temporal::rw1_quadratic_form(phi_g, T, data.temporal_cyclic);
        int rank = data.temporal_cyclic ? T : T - 1;
        log_post += 0.5 * rank * log_tau_temporal - 0.5 * tau_temporal * quad;
        // Soft sum-to-zero constraint
        log_post += ratiod_temporal::sum_to_zero_penalty(phi_g, T, 0.001);

      } else if (data.temporal_type == TemporalType::RW2) {
        double quad = ratiod_temporal::rw2_quadratic_form(phi_g, T, data.temporal_cyclic);
        int rank = data.temporal_cyclic ? T : T - 2;
        log_post += 0.5 * rank * log_tau_temporal - 0.5 * tau_temporal * quad;
        // Soft sum-to-zero constraint
        log_post += ratiod_temporal::sum_to_zero_penalty(phi_g, T, 0.001);

      } else if (data.temporal_type == TemporalType::AR1) {
        log_post += ratiod_temporal::ar1_log_density(phi_g, T, rho_ar1, tau_temporal);
      }
    }
  }

  // GP spatial priors
  double sigma2_gp = 1.0, phi_gp = 1.0;
  const double* gp_w = nullptr;

  if (layout.is_gp && data.has_gp) {
    double log_sigma2_gp = params[layout.log_sigma2_gp_idx];
    double log_phi_gp = params[layout.log_phi_gp_idx];
    sigma2_gp = std::exp(log_sigma2_gp);
    phi_gp = std::exp(log_phi_gp);
    gp_w = &params[layout.gp_w_start];

    // PC prior on sigma2 (favor smaller variance)
    log_post += ratiod_gp::log_prior_sigma2_pc(sigma2_gp, data.gp_sigma2_prior_U,
                                                data.gp_sigma2_prior_alpha);
    log_post += log_sigma2_gp;  // Jacobian for log transform

    // Uniform prior on phi within bounds
    log_post += ratiod_gp::log_prior_phi_uniform(phi_gp, data.gp_phi_prior_lower,
                                                  data.gp_phi_prior_upper);
    log_post += log_phi_gp;  // Jacobian

    // NNGP prior on spatial effects
    // Bounds check: ensure we don't read past end of params vector
    if (layout.gp_w_start + data.gp_data.n_obs > (int)params.size()) {
      return -INFINITY;  // Invalid parameter layout
    }
    std::vector<double> w_vec(gp_w, gp_w + data.gp_data.n_obs);

    // Apply RSR projection if enabled
    if (data.has_rsr && !data.rsr_projection.empty()) {
      std::vector<double> w_projected(data.rsr_n, 0.0);
      for (int i = 0; i < data.rsr_n; i++) {
        for (int j = 0; j < data.rsr_n; j++) {
          w_projected[i] += data.rsr_projection[i * data.rsr_n + j] * w_vec[j];
        }
      }
      w_vec = w_projected;
    }

    double gp_ll = ratiod_gp::gp_nngp_log_lik(w_vec, sigma2_gp, phi_gp, data.gp_data);
    log_post += gp_ll;
  }

  // Multi-scale GP spatial priors
  double sigma2_local = 1.0, phi_local = 1.0;
  double sigma2_regional = 1.0, phi_regional = 1.0;
  const double* gp_local = nullptr;
  const double* gp_regional = nullptr;

  if (layout.is_multiscale_gp && data.has_multiscale_gp) {
    // Local scale parameters
    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_phi_local = params[layout.log_phi_gp_local_idx];
    sigma2_local = std::exp(log_sigma2_local);
    phi_local = std::exp(log_phi_local);
    gp_local = &params[layout.gp_local_start];

    // Regional scale parameters
    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_phi_regional = params[layout.log_phi_gp_regional_idx];
    sigma2_regional = std::exp(log_sigma2_regional);
    phi_regional = std::exp(log_phi_regional);
    gp_regional = &params[layout.gp_regional_start];

    // PC priors on variances
    log_post += ratiod_gp::log_prior_sigma2_pc(sigma2_local, data.ms_sigma2_local_prior_U,
                                                data.ms_sigma2_local_prior_alpha);
    log_post += log_sigma2_local;

    log_post += ratiod_gp::log_prior_sigma2_pc(sigma2_regional, data.ms_sigma2_regional_prior_U,
                                                data.ms_sigma2_regional_prior_alpha);
    log_post += log_sigma2_regional;

    // Range priors (uniform within bounds)
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper) {
      return -std::numeric_limits<double>::infinity();
    }
    log_post += log_phi_local;

    if (phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
      return -std::numeric_limits<double>::infinity();
    }
    log_post += log_phi_regional;

    // NNGP likelihood for each scale
    std::vector<double> w_local_vec(gp_local, gp_local + data.multiscale_gp_data.n_obs);
    std::vector<double> w_regional_vec(gp_regional, gp_regional + data.multiscale_gp_data.n_obs);

    // Apply RSR projection if enabled
    if (data.has_rsr && !data.rsr_projection.empty()) {
      std::vector<double> local_proj(data.rsr_n, 0.0);
      std::vector<double> regional_proj(data.rsr_n, 0.0);
      for (int i = 0; i < data.rsr_n; i++) {
        for (int j = 0; j < data.rsr_n; j++) {
          local_proj[i] += data.rsr_projection[i * data.rsr_n + j] * w_local_vec[j];
          regional_proj[i] += data.rsr_projection[i * data.rsr_n + j] * w_regional_vec[j];
        }
      }
      w_local_vec = local_proj;
      w_regional_vec = regional_proj;
    }

    log_post += ratiod_gp::multiscale_gp_log_lik(w_local_vec, w_regional_vec,
                                                  sigma2_local, phi_local,
                                                  sigma2_regional, phi_regional,
                                                  data.multiscale_gp_data);
  }

  // Multi-scale temporal priors
  double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
  double rho_short = 0.5;
  const double* trend = nullptr;
  const double* seasonal = nullptr;
  const double* short_term = nullptr;

  if (layout.has_multiscale_temporal) {
    std::vector<double> trend_vec, seasonal_vec, short_term_vec;

    // Trend component
    if (layout.log_sigma2_trend_idx >= 0) {
      double log_sigma2_trend = params[layout.log_sigma2_trend_idx];
      sigma2_trend = std::exp(log_sigma2_trend);
      trend = &params[layout.trend_start];
      trend_vec.assign(trend, trend + data.multiscale_temporal_data.n_times);

      // PC prior
      log_post += ratiod_temporal::log_prior_sigma2_temporal_pc(
        sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
      log_post += log_sigma2_trend;
    }

    // Seasonal component
    if (layout.log_sigma2_seasonal_idx >= 0) {
      double log_sigma2_seasonal = params[layout.log_sigma2_seasonal_idx];
      sigma2_seasonal = std::exp(log_sigma2_seasonal);
      seasonal = &params[layout.seasonal_start];
      seasonal_vec.assign(seasonal, seasonal + data.multiscale_temporal_data.seasonal_period);

      // PC prior
      log_post += ratiod_temporal::log_prior_sigma2_temporal_pc(
        sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
      log_post += log_sigma2_seasonal;
    }

    // Short-term component
    if (layout.log_sigma2_short_idx >= 0) {
      double log_sigma2_short = params[layout.log_sigma2_short_idx];
      sigma2_short = std::exp(log_sigma2_short);
      short_term = &params[layout.short_term_start];
      short_term_vec.assign(short_term, short_term + data.multiscale_temporal_data.n_times);

      // PC prior
      log_post += ratiod_temporal::log_prior_sigma2_temporal_pc(
        sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
      log_post += log_sigma2_short;

      // AR1 rho parameter
      if (layout.logit_rho_short_idx >= 0) {
        double logit_rho_short = params[layout.logit_rho_short_idx];
        rho_short = 2.0 / (1.0 + std::exp(-logit_rho_short)) - 1.0;  // Map to (-1, 1)

        // Prior on rho (Beta(2,2) on transformed scale)
        log_post += ratiod_temporal::log_prior_rho(rho_short, 2.0, 2.0);
        // Jacobian for logit transform
        double x = (rho_short + 1.0) / 2.0;
        log_post += std::log(x) + std::log(1.0 - x);
      }
    }

    // Multi-scale temporal log-likelihood
    log_post += ratiod_temporal::multiscale_temporal_log_lik(
      trend_vec, seasonal_vec, short_term_vec,
      sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
      data.multiscale_temporal_data);
  }

  // Latent factor priors
  std::vector<double> latent_sigma;
  std::vector<double> latent_factors_vec;
  if (layout.has_latent && data.latent_n_factors > 0) {
    int K = data.latent_n_factors;
    int N = data.N;

    // Extract log_sigma parameters
    std::vector<double> log_sigma_latent(K);
    for (int k = 0; k < K; k++) {
      log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
    }

    // Extract sigma (exponentiated)
    latent_sigma.resize(K);
    for (int k = 0; k < K; k++) {
      latent_sigma[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factor scores (N x K, row-major)
    int n_factor_params = N * K;
    latent_factors_vec.resize(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
      latent_factors_vec[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint (sum-to-zero or first-zero)
    if (data.latent_constraint == 0) {  // sum_to_zero
      ratiod_latent::apply_sum_to_zero(latent_factors_vec, N, K);
    } else {  // first_zero
      ratiod_latent::apply_first_zero(latent_factors_vec, N, K);
    }

    // PC prior on sigma (exponential prior with Jacobian)
    log_post += ratiod_latent::latent_sigma_log_prior(log_sigma_latent,
                                                       data.latent_sigma_prior_rate);

    // Standard normal prior on factor scores
    ratiod_latent::LatentConstraint constraint =
      (data.latent_constraint == 0) ? ratiod_latent::LatentConstraint::SUM_TO_ZERO
                                    : ratiod_latent::LatentConstraint::FIRST_ZERO;
    log_post += ratiod_latent::latent_factor_log_prior(latent_factors_vec, N, K, constraint);
  }

  // Spatiotemporal interaction priors
  double tau_st = 1.0, tau_st2 = 1.0, rho_st = 0.0;
  double phi_st_space = 1.0, phi_st_time = 1.0;
  const double* st_delta = nullptr;

  if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
    // Extract precision parameter
    double log_tau_st = params[layout.log_tau_st_idx];
    tau_st = std::exp(log_tau_st);

    // PC prior on tau (exponential on sigma = 1/sqrt(tau))
    double sigma_st = 1.0 / std::sqrt(tau_st);
    double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
    log_post += std::log(lambda) - lambda * sigma_st - std::log(2.0 * sigma_st);
    log_post += log_tau_st;  // Jacobian for log transform

    // Second precision for Type IV
    if (layout.log_tau_st2_idx >= 0) {
      double log_tau_st2 = params[layout.log_tau_st2_idx];
      tau_st2 = std::exp(log_tau_st2);

      // PC prior on tau2
      double sigma_st2 = 1.0 / std::sqrt(tau_st2);
      log_post += std::log(lambda) - lambda * sigma_st2 - std::log(2.0 * sigma_st2);
      log_post += log_tau_st2;  // Jacobian
    }

    // AR1 rho parameter
    if (layout.logit_rho_st_idx >= 0) {
      double logit_rho_st = params[layout.logit_rho_st_idx];
      rho_st = 2.0 / (1.0 + std::exp(-logit_rho_st)) - 1.0;  // Map to (-1, 1)

      // Uniform(-1, 1) prior on rho
      // Jacobian for logit((rho+1)/2) transform
      double x = (rho_st + 1.0) / 2.0;
      log_post += std::log(x) + std::log(1.0 - x);
    }

    // GP range parameters
    if (layout.is_st_gp) {
      double log_phi_space = params[layout.log_phi_st_space_idx];
      double log_phi_time = params[layout.log_phi_st_time_idx];
      phi_st_space = std::exp(log_phi_space);
      phi_st_time = std::exp(log_phi_time);

      // Uniform prior within bounds
      if (phi_st_space < data.st_phi_space_prior_lower ||
          phi_st_space > data.st_phi_space_prior_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      if (phi_st_time < data.st_phi_time_prior_lower ||
          phi_st_time > data.st_phi_time_prior_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      log_post += log_phi_space + log_phi_time;  // Jacobians
    }

    // Spatiotemporal interaction effects
    st_delta = &params[layout.st_delta_start];

    // Apply spatiotemporal log-prior from hmc_spatiotemporal.h
    log_post += ratiod_spatiotemporal::spatiotemporal_log_prior(
      st_delta, tau_st, tau_st2, rho_st, phi_st_space, phi_st_time,
      data.spatiotemporal_data
    );

    // Soft sum-to-zero constraint for identifiability
    int S = data.spatiotemporal_data.n_spatial;
    int T = data.spatiotemporal_data.n_times;
    log_post += ratiod_spatiotemporal::st_sum_to_zero_penalty(
      st_delta, S, T, 0.001, true, true
    );
  }

  // ============ LIKELIHOOD (parallelized) ============

  double log_lik = 0.0;

  // NOTE: Disable OpenMP for GP models to avoid race conditions
  // The GP NNGP likelihood accesses shared data structures that may not be thread-safe
  #ifdef _OPENMP
  int use_threads = (layout.is_gp || layout.is_multiscale_gp) ? 1 : data.n_threads;
  #pragma omp parallel for reduction(+:log_lik) schedule(static) \
          num_threads(use_threads)
  #endif
  for (int i = 0; i < data.N; i++) {
    // Linear predictor for numerator (using optimized dot product)
    double eta_num = ratiod_linalg::dot_product(
        &data.X_num_flat[i * data.p_num], beta_num, data.p_num);

    // Linear predictor for denominator (using optimized dot product)
    double eta_denom = ratiod_linalg::dot_product(
        &data.X_denom_flat[i * data.p_denom], beta_denom, data.p_denom);

    // Add random effects (shared between num and denom)
    // Supports multiple crossed RE terms with slopes
    if (layout.has_re) {
      int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

      if (layout.has_re_slopes) {
        // Random slopes case
        for (int t = 0; t < n_terms; t++) {
          int group_idx = data.re_group_multi[t][i];
          if (group_idx > 0) {
            int g = group_idx - 1;
            int n_coefs = layout.re_n_coefs_multi[t];
            int re_base = layout.re_start_multi[t] + g * n_coefs;

            // Intercept contribution (coefficient 0)
            double re_contrib = params[re_base];

            // Slope contributions (coefficients 1, 2, ...)
            int n_slopes = n_coefs - 1;  // Assuming intercept is always first
            if (n_slopes > 0 && !data.re_slope_matrices[t].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                // Slope design matrix is stored as [N x n_slopes] flattened row-major
                double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                double re_slope = params[re_base + 1 + s];
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            eta_denom += re_contrib;
          }
        }
      } else if (n_terms > 1) {
        // Multiple RE terms (intercept only)
        for (int t = 0; t < n_terms; t++) {
          int group_idx = data.re_group_multi[t][i];
          if (group_idx > 0) {
            int g = group_idx - 1;
            double re_val = params[layout.re_start_multi[t] + g];
            eta_num += re_val;
            eta_denom += re_val;
          }
        }
      } else {
        // Single RE term (legacy path)
        if (data.re_group[i] > 0) {
          int g = data.re_group[i] - 1;
          eta_num += re[g];
          eta_denom += re[g];
        }
      }
    }

    // Add spatial effect (ICAR/BYM2, not GP which is handled separately)
    if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
      int s = data.spatial_group[i] - 1;
      double spatial_effect;

      if (layout.is_bym2) {
        // BYM2: u = sigma * (sqrt(rho)*phi_scaled*scale + sqrt(1-rho)*theta)
        double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
        spatial_effect = sigma_bym2 * (
          std::sqrt(rho_bym2) * scaled_phi +
          std::sqrt(1.0 - rho_bym2) * theta_bym2[s]
        );
      } else {
        spatial_effect = phi_spatial[s];
      }

      eta_num += spatial_effect;
      eta_denom += spatial_effect;
    }

    // Add temporal effect (shared between num and denom by default)
    if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
      int t = data.temporal_time_idx[i] - 1;  // Time index (0-based)
      int g = data.temporal_group_idx[i] - 1;  // Group index (0-based)
      int T = data.n_times;

      // Get temporal effect for this observation
      double temporal_effect = phi_temporal[g * T + t];

      // Add to both linear predictors (shared structure)
      if (data.temporal_shared) {
        eta_num += temporal_effect;
        eta_denom += temporal_effect;
      } else {
        // If not shared, only add to numerator (or we'd need separate params)
        eta_num += temporal_effect;
      }
    }

    // Add GP spatial effect (observation-level)
    if (layout.is_gp && data.has_gp && gp_w != nullptr) {
      double gp_effect = gp_w[i];
      if (data.gp_data.shared) {
        eta_num += gp_effect;
        eta_denom += gp_effect;
      } else {
        eta_num += gp_effect;
      }
    }

    // Add multi-scale GP spatial effects (observation-level)
    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
      double local_effect = gp_local[i];
      double regional_effect = gp_regional[i];
      double ms_spatial_effect = local_effect + regional_effect;

      if (data.multiscale_gp_data.shared) {
        eta_num += ms_spatial_effect;
        eta_denom += ms_spatial_effect;
      } else {
        eta_num += ms_spatial_effect;
      }
    }

    // Add multi-scale temporal effect
    if (layout.has_multiscale_temporal) {
      double ms_temporal_effect = 0.0;
      int t_idx = data.multiscale_temporal_data.time_index[i] - 1;  // 0-based

      // Trend component
      if (trend != nullptr && t_idx >= 0 &&
          t_idx < static_cast<int>(data.multiscale_temporal_data.n_times)) {
        ms_temporal_effect += trend[t_idx];
      }

      // Seasonal component
      if (seasonal != nullptr && data.multiscale_temporal_data.seasonal_period > 0) {
        int s_idx = t_idx % data.multiscale_temporal_data.seasonal_period;
        ms_temporal_effect += seasonal[s_idx];
      }

      // Short-term component
      if (short_term != nullptr && t_idx >= 0 &&
          t_idx < static_cast<int>(data.multiscale_temporal_data.n_times)) {
        ms_temporal_effect += short_term[t_idx];
      }

      if (data.multiscale_temporal_data.shared) {
        eta_num += ms_temporal_effect;
        eta_denom += ms_temporal_effect;
      } else {
        eta_num += ms_temporal_effect;
      }
    }

    // Add latent factor effect
    if (layout.has_latent && data.latent_n_factors > 0 && !latent_factors_vec.empty()) {
      int K = data.latent_n_factors;
      double latent_effect = 0.0;
      for (int k = 0; k < K; k++) {
        latent_effect += latent_factors_vec[i * K + k] * latent_sigma[k];
      }
      if (data.latent_shared) {
        eta_num += latent_effect;
        eta_denom += latent_effect;
      } else {
        eta_num += latent_effect;
      }
    }

    // Add spatiotemporal interaction effect
    if (layout.has_spatiotemporal && st_delta != nullptr) {
      // Get spatiotemporal index for this observation
      int st_idx = data.spatiotemporal_data.st_flat[i];
      if (st_idx > 0) {
        double st_effect = st_delta[st_idx - 1];  // Convert to 0-based

        if (data.spatiotemporal_data.shared) {
          eta_num += st_effect;
          eta_denom += st_effect;
        } else {
          eta_num += st_effect;
        }
      }
    }

    // Compute ZI linear predictor if applicable (using optimized dot product)
    double logit_zi = 0.0;
    if (layout.has_zi) {
      logit_zi = ratiod_linalg::dot_product(
          &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
    }

    // Likelihood contribution
    double ll_i = 0.0;
    if (data.model_type == ModelType::BINOMIAL) {
      ll_i = log_lik_binomial(data.y_num[i], data.y_denom[i], eta_num);
    } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);

      // Check for zero-inflation on numerator
      if (layout.has_zi) {
        ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                           logit_zi, data.zi_type);
      } else {
        ll_i = log_lik_negbin(data.y_num[i], mu_num, phi_num);
      }
      // Denominator is always standard (not zero-inflated)
      ll_i += log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);

    } else {  // POISSON_GAMMA
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);

      // Check for zero-inflation on numerator
      if (layout.has_zi) {
        ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                           logit_zi, data.zi_type);
      } else {
        ll_i = log_lik_poisson(data.y_num[i], mu_num);
      }
      // Denominator is gamma (continuous)
      ll_i += log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
    }

    log_lik += ll_i;
  }

  log_post += log_lik;
  return log_post;
}

// =====================================================================
// Analytical gradient for simple Poisson-Gamma models
// O(n) instead of O(n*p) - huge speedup for typical models
// =====================================================================

bool can_use_analytical_gradient(const ModelData& data, const ParamLayout& layout) {
  // Only for simple models without complex structure
  return (data.model_type == ModelType::POISSON_GAMMA &&
          !layout.is_gp && !layout.is_multiscale_gp &&
          !layout.has_spatial && !layout.has_temporal &&
          !layout.has_zi && !layout.has_re_slopes &&
          !layout.has_latent && !layout.has_spatiotemporal &&
          !layout.has_multiscale_temporal &&
          data.n_re_terms <= 1);  // Single or no RE term
}

void compute_gradient_analytical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Extract parameters
  const double* beta_num = &params[layout.beta_num_start];
  const double* beta_denom = &params[layout.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0, tau_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, log_phi_num = 0.0;
  if (layout.has_phi_num) {
    log_phi_num = params[layout.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }

  // ============ Prior gradients (cheap) ============

  // Beta priors: N(0, sigma_beta^2)
  // d/d(beta) = -tau_beta * beta where tau_beta = 1/sigma_beta^2
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.p_num; j++) {
    grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];
  }

  // sigma_re: Half-Cauchy prior with scale = data.sigma_re_scale (via log transform)
  // log_post = -log(1 + (sigma/scale)^2) + log(sigma) (Jacobian)
  // d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
  if (layout.has_re) {
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
  }

  // phi_num: Gamma(2, 0.1) prior (via log transform)
  // d/d(log_phi) = (shape-1) - rate*phi + 1 (Jacobian)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] = (data.phi_prior_shape - 1.0)
                                   - data.phi_prior_rate * phi_num + 1.0;
  }

  // RE prior: N(0, sigma_re^2)
  // d/d(re[g]) = -tau_re * re[g]
  // Also accumulates contribution to sigma_re gradient
  double re_prior_grad_sigma = 0.0;
  if (layout.has_re) {
    for (int g = 0; g < data.n_re_groups; g++) {
      double re_g = re[g];
      grad[layout.re_start + g] = -tau_re * re_g;
      // Contribution to sigma gradient: d/d(log_sigma) of -0.5*tau*re^2 + 0.5*log(tau)
      // = tau*re^2 - 1 (after chain rule for log transform)
      re_prior_grad_sigma += tau_re * re_g * re_g - 1.0;
    }
    grad[layout.log_sigma_re_idx] += re_prior_grad_sigma;
  }

  // ============ Likelihood gradients (O(n)) ============

  // Accumulators for beta gradients (will be added via X' * residual)
  std::vector<double> grad_beta_num(data.p_num, 0.0);
  std::vector<double> grad_beta_denom(data.p_denom, 0.0);
  double grad_phi_lik = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel
  {
    std::vector<double> local_grad_beta_num(data.p_num, 0.0);
    std::vector<double> local_grad_beta_denom(data.p_denom, 0.0);
    double local_grad_phi = 0.0;
    std::vector<double> local_grad_re(layout.has_re ? data.n_re_groups : 0, 0.0);

    #pragma omp for schedule(static)
    for (int i = 0; i < data.N; i++) {
  #else
    for (int i = 0; i < data.N; i++) {
  #endif
      // Compute linear predictors
      double eta_num = ratiod_linalg::dot_product(
          &data.X_num_flat[i * data.p_num], beta_num, data.p_num);
      double eta_denom = ratiod_linalg::dot_product(
          &data.X_denom_flat[i * data.p_denom], beta_denom, data.p_denom);

      // Add RE if present
      int re_idx = -1;
      if (layout.has_re && data.re_group[i] > 0) {
        re_idx = data.re_group[i] - 1;
        eta_num += re[re_idx];
        eta_denom += re[re_idx];
      }

      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);

      // Poisson residual: d(LL)/d(eta_num) = y - mu
      double resid_num = data.y_num[i] - mu_num;

      // Gamma residual: d(LL)/d(eta_denom) = phi * (y/mu - 1)
      double y_denom = data.y_denom_cont[i];
      double resid_denom = phi_num * (y_denom / mu_denom - 1.0);

      // Accumulate beta gradients: grad += X[i,:] * resid
      for (int j = 0; j < data.p_num; j++) {
        #ifdef _OPENMP
        local_grad_beta_num[j] += data.X_num_flat[i * data.p_num + j] * resid_num;
        #else
        grad_beta_num[j] += data.X_num_flat[i * data.p_num + j] * resid_num;
        #endif
      }
      for (int j = 0; j < data.p_denom; j++) {
        #ifdef _OPENMP
        local_grad_beta_denom[j] += data.X_denom_flat[i * data.p_denom + j] * resid_denom;
        #else
        grad_beta_denom[j] += data.X_denom_flat[i * data.p_denom + j] * resid_denom;
        #endif
      }

      // Accumulate RE gradient (shared between num and denom)
      if (re_idx >= 0) {
        #ifdef _OPENMP
        local_grad_re[re_idx] += resid_num + resid_denom;
        #else
        grad[layout.re_start + re_idx] += resid_num + resid_denom;
        #endif
      }

      // Phi gradient from Gamma likelihood
      // d(LL)/d(phi) = log(phi/mu) + 1 - digamma(phi) + digamma(phi) contribution...
      // Simplified: d(LL)/d(log_phi) = phi * [log(phi*y/mu) - y/mu + 1 - digamma(phi) + log(phi)]
      // This is complex, use numerical for phi for now
      double rate = phi_num / mu_denom;
      #ifdef _OPENMP
      local_grad_phi += std::log(rate) + 1.0 + std::log(y_denom) - R::digamma(phi_num) - rate * y_denom / phi_num;
      #else
      grad_phi_lik += std::log(rate) + 1.0 + std::log(y_denom) - R::digamma(phi_num) - rate * y_denom / phi_num;
      #endif
    }

  #ifdef _OPENMP
    // Reduce local accumulators
    #pragma omp critical
    {
      for (int j = 0; j < data.p_num; j++) {
        grad_beta_num[j] += local_grad_beta_num[j];
      }
      for (int j = 0; j < data.p_denom; j++) {
        grad_beta_denom[j] += local_grad_beta_denom[j];
      }
      grad_phi_lik += local_grad_phi;
      for (int g = 0; g < (int)local_grad_re.size(); g++) {
        grad[layout.re_start + g] += local_grad_re[g];
      }
    }
  }  // end parallel
  #endif

  // Add likelihood gradients to total
  for (int j = 0; j < data.p_num; j++) {
    grad[layout.beta_num_start + j] += grad_beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    grad[layout.beta_denom_start + j] += grad_beta_denom[j];
  }

  // Phi gradient (Jacobian already included in prior gradient)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] += phi_num * grad_phi_lik;
  }
}

// =====================================================================
// Numerical gradient (fallback for complex models)
// =====================================================================

void compute_gradient_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
  int n = params.size();
  grad.resize(n);

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = compute_log_post(params_plus, data, layout);
    double f_minus = compute_log_post(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Unified gradient interface
// =====================================================================

// Debug: compare analytical vs numerical gradients
bool verify_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol = 1e-4
) {
  std::vector<double> grad_analytical, grad_numerical;
  compute_gradient_analytical(params, data, layout, grad_analytical);
  compute_gradient_numerical(params, data, layout, grad_numerical);

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_analytical.size(); i++) {
    double diff = std::abs(grad_analytical[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_analytical[i]), std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    Rcpp::Rcerr << "Gradient mismatch! Max rel diff: " << max_diff
                << " at param " << worst_idx
                << " (analytical: " << grad_analytical[worst_idx]
                << ", numerical: " << grad_numerical[worst_idx] << ")\n";
    return false;
  }
  return true;
}

// =====================================================================
// Autodiff gradient (O(n) - works for ALL models)
// =====================================================================

void compute_gradient_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace ratiod::ad;

    // Initialize autodiff tape
    init_tape();

    // Create autodiff variables from parameters
    std::vector<Var> params_ad = make_vars(params);

    // Compute log posterior using templated implementation
    Var log_post = ratiod::compute_log_post_impl(params_ad, data, layout);

    // Backward pass to compute gradients
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // Clean up tape
    clear_tape();
}

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    // Use hand-coded analytical gradients for simple Poisson-Gamma (fastest, 9x)
    if (can_use_analytical_gradient(data, layout)) {
        compute_gradient_analytical(params, data, layout, grad);
    } else if (layout.is_gp || layout.is_multiscale_gp) {
        // GP models not yet supported in autodiff (log_post_impl.h missing GP)
        // Fall back to numerical gradients until GP autodiff is implemented
        compute_gradient_numerical(params, data, layout, grad);
    } else {
        // Use autodiff for all other models (fast, 3-5x)
        compute_gradient_autodiff(params, data, layout, grad);
    }
}

// =====================================================================
// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, int n_params)
  : mu(std::log(epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(compute_target(n_params)), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: 1e-6 for very challenging posteriors
  // Upper bound: 0.05 (conservative to avoid divergences)
  log_epsilon = std::max(-14.0, std::min(log_epsilon, -3.0));
  double epsilon = std::exp(log_epsilon);
  double m_w = std::pow((double)m, -kappa);
  log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
  return epsilon;
}

double DualAveraging::final_epsilon() const {
  return std::exp(log_epsilon_bar);
}

// =====================================================================
// Welford's online algorithm for mean and variance
// Used for diagonal mass matrix estimation during warmup
// =====================================================================

class WelfordStats {
public:
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // Sum of squared differences from mean

  WelfordStats(int dim) : n(0), mean(dim, 0.0), M2(dim, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    for (size_t i = 0; i < x.size(); i++) {
      double delta = x[i] - mean[i];
      mean[i] += delta / n;
      double delta2 = x[i] - mean[i];
      M2[i] += delta * delta2;
    }
  }

  std::vector<double> variance() const {
    std::vector<double> var(mean.size());
    if (n < 2) {
      // Return unit variance if not enough samples
      std::fill(var.begin(), var.end(), 1.0);
    } else {
      for (size_t i = 0; i < mean.size(); i++) {
        var[i] = M2[i] / (n - 1);
        // Ensure minimum variance to avoid numerical issues
        if (var[i] < 1e-6) var[i] = 1e-6;
      }
    }
    return var;
  }

  // Get inverse mass matrix (= variance, clamped for stability)
  // For HMC: M = diag(1/var), so M^{-1} = diag(var)
  // High variance parameters should move faster in position space
  std::vector<double> inv_mass() const {
    auto var = variance();
    for (size_t i = 0; i < var.size(); i++) {
      // Clamp variance to [1e-3, 1e3]
      var[i] = std::max(1e-3, std::min(var[i], 1e3));
    }
    return var;
  }

  // Get sqrt of mass matrix for momentum sampling
  // Since M = diag(1/var), sqrt(M) = diag(1/sqrt(var))
  // p ~ N(0, M), so p_i = z_i / sqrt(var_i)
  std::vector<double> sqrt_mass() const {
    std::vector<double> sqrt_m(mean.size());
    auto var = variance();
    for (size_t i = 0; i < var.size(); i++) {
      double clamped_var = std::max(1e-3, std::min(var[i], 1e3));
      sqrt_m[i] = 1.0 / std::sqrt(clamped_var);
    }
    return sqrt_m;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

// =====================================================================
// Leapfrog integrator
// =====================================================================

LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  for (int i = 0; i < n; i++) {
    result.q[i] += epsilon * result.p[i];
  }

  // Half step for momentum
  result.log_prob = compute_log_post(result.q, data, layout);
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }

  // Also check for extreme parameter values
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

// =====================================================================
// Find reasonable initial step size
// =====================================================================

// Compute diagonal mass matrix from gradient magnitudes
// This provides automatic scaling for poorly-conditioned posteriors
std::vector<double> compute_diagonal_mass(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  std::vector<double> grad(n);
  compute_gradient(q, data, layout, grad);

  std::vector<double> mass(n);
  for (int i = 0; i < n; i++) {
    // Use gradient magnitude as rough estimate of curvature
    // Mass ~ 1/variance, so larger gradient -> larger mass -> smaller step in that direction
    double abs_grad = std::abs(grad[i]);
    // Clamp to reasonable range [1, 1000]
    mass[i] = std::max(1.0, std::min(abs_grad, 1000.0));
  }

  return mass;
}

// Leapfrog step with diagonal mass matrix
LeapfrogResult leapfrog_step_mass(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const std::vector<double>& inv_mass,  // inverse mass (1/M)
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position (scaled by inverse mass)
  for (int i = 0; i < n; i++) {
    result.q[i] += epsilon * inv_mass[i] * result.p[i];
  }

  // Half step for momentum
  result.log_prob = compute_log_post(result.q, data, layout);
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }

  // Check for extreme parameter values
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
) {
  int n = q.size();

  // Compute gradient norm to scale step size
  std::vector<double> grad(n);
  compute_gradient(q, data, layout, grad);
  double grad_norm = 0.0;
  double max_grad = 0.0;
  for (int i = 0; i < n; i++) {
    grad_norm += grad[i] * grad[i];
    max_grad = std::max(max_grad, std::abs(grad[i]));
  }
  grad_norm = std::sqrt(grad_norm);

  // Conservative step size based on both gradient norm and max gradient
  // This ensures we don't take too large steps in any direction
  double eps_by_norm = 1.0 / (grad_norm + 1.0);
  double eps_by_max = 0.5 / (max_grad + 1.0);
  double eps_max = std::min(eps_by_norm, eps_by_max);
  eps_max = std::max(1e-5, std::min(eps_max, 0.01));  // Clamp to [1e-5, 0.01]

  // Start from a conservative initial epsilon
  double epsilon = eps_max;

  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<double> p(n);
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng);
  }

  double log_prob_init = compute_log_post(q, data, layout);
  double kinetic_init = 0.5 * ratiod_linalg::norm_squared(p.data(), n);
  double H_init = -log_prob_init + kinetic_init;

  LeapfrogResult lf = leapfrog_step(q, p, epsilon, data, layout);
  double kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
  double H_new = -lf.log_prob + kinetic_new;

  // If initial step is OK, try to increase epsilon
  double delta_H = H_new - H_init;
  if (std::isfinite(H_new) && delta_H < std::log(0.5)) {
    // Can potentially increase step size
    for (int iter = 0; iter < 10; iter++) {
      double new_eps = epsilon * 2.0;
      if (new_eps > eps_max * 10) break;  // Don't go too far above eps_max

      lf = leapfrog_step(q, p, new_eps, data, layout);
      if (!std::isfinite(lf.log_prob)) break;

      kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
      H_new = -lf.log_prob + kinetic_new;
      delta_H = H_new - H_init;

      if (delta_H > std::log(0.5)) break;  // Too large
      epsilon = new_eps;
    }
  } else {
    // Need to decrease step size
    for (int iter = 0; iter < 20; iter++) {
      epsilon *= 0.5;
      if (epsilon < 1e-6) break;

      lf = leapfrog_step(q, p, epsilon, data, layout);
      if (!std::isfinite(lf.log_prob)) continue;

      kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
      H_new = -lf.log_prob + kinetic_new;
      delta_H = H_new - H_init;

      if (delta_H < std::log(0.5)) break;  // Found good step size
    }
  }

  // Final clamp to ensure numerical stability
  return std::max(1e-6, std::min(epsilon, 0.01));
}

// =====================================================================
// Run single HMC chain
// =====================================================================

// Pure C++ version - safe for OpenMP parallel regions
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;

  HMCResultCpp result;
  result.samples.resize(n_sample, std::vector<double>(n_params));
  result.log_prob.resize(n_sample);
  result.accept_prob.resize(n_sample);
  result.n_leapfrog.resize(n_sample, L);
  result.divergent.resize(n_sample, 0);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;
  result.chain_id = chain_id;

  std::mt19937 rng(seed + chain_id * 12345);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  std::vector<double> q = q_init;

  double log_prob_current = compute_log_post(q, data, layout);

  double epsilon = find_reasonable_epsilon(q, data, layout, rng);
  DualAveraging da(epsilon, n_params);

  // Mass matrix adaptation
  // For M = diag(1/var), inv_mass = M^{-1} = diag(var)
  // sqrt_mass_diag = sqrt(M) = diag(1/sqrt(var)) for momentum sampling
  std::vector<double> inv_mass(n_params, 1.0);  // Starts as identity
  std::vector<double> sqrt_mass_diag(n_params, 1.0);  // For p ~ N(0, M)
  WelfordStats mass_stats(n_params);
  bool use_mass_matrix = false;

  // Warmup windows for mass matrix adaptation
  // Window 1 (0 to 50%): collect samples with identity mass
  // Window 2 (50% to 75%): use estimated mass, continue adapting
  // Window 3 (75% to 100%): final mass, adapt epsilon only
  int warmup_window1 = n_warmup / 2;
  int warmup_window2 = (3 * n_warmup) / 4;

  int sample_idx = 0;
  int n_accept = 0;
  int n_divergent = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);

    // Check if we should update mass matrix
    if (iter == warmup_window1 && mass_stats.n >= 10) {
      // End of window 1: compute initial mass matrix
      inv_mass = mass_stats.inv_mass();
      sqrt_mass_diag = mass_stats.sqrt_mass();
      use_mass_matrix = true;
      mass_stats.reset();  // Reset for next window
      // Reinitialize epsilon for new mass matrix
      epsilon = find_reasonable_epsilon(q, data, layout, rng);
      da = DualAveraging(epsilon, n_params);
    } else if (iter == warmup_window2 && mass_stats.n >= 10) {
      // End of window 2: final mass matrix update
      inv_mass = mass_stats.inv_mass();
      sqrt_mass_diag = mass_stats.sqrt_mass();
      // Reinitialize epsilon for final mass matrix
      epsilon = find_reasonable_epsilon(q, data, layout, rng);
      da = DualAveraging(epsilon, n_params);
    }

    // Sample momentum (scaled by sqrt of mass matrix)
    // p ~ N(0, M) where M = diag(1/var), so p_i = z_i * sqrt(1/var_i)
    std::vector<double> p(n_params);
    for (int i = 0; i < n_params; i++) {
      p[i] = normal(rng) * sqrt_mass_diag[i];
    }

    // Current Hamiltonian with mass matrix
    // K = 0.5 * p' * M^{-1} * p = 0.5 * sum(p[i]^2 * inv_mass[i])
    double kinetic_current = 0.0;
    for (int i = 0; i < n_params; i++) {
      kinetic_current += p[i] * p[i] * inv_mass[i];
    }
    kinetic_current *= 0.5;
    double H_current = -log_prob_current + kinetic_current;

    // Leapfrog integration (with or without mass matrix)
    std::vector<double> q_prop = q;
    std::vector<double> p_prop = p;
    bool divergent = false;

    for (int l = 0; l < L; l++) {
      LeapfrogResult lf;
      if (use_mass_matrix) {
        lf = leapfrog_step_mass(q_prop, p_prop, epsilon, inv_mass, data, layout);
      } else {
        lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout);
      }
      q_prop = lf.q;
      p_prop = lf.p;
      if (lf.divergent) {
        divergent = true;
        break;
      }
    }

    double log_prob_prop = compute_log_post(q_prop, data, layout);
    double kinetic_prop = 0.0;
    for (int i = 0; i < n_params; i++) {
      kinetic_prop += p_prop[i] * p_prop[i] * inv_mass[i];
    }
    kinetic_prop *= 0.5;
    double H_prop = -log_prob_prop + kinetic_prop;

    // Metropolis accept/reject
    double alpha = std::min(1.0, std::exp(H_current - H_prop));
    if (!std::isfinite(alpha)) alpha = 0.0;

    bool accepted = (unif(rng) < alpha) && !divergent;
    if (accepted) {
      q = q_prop;
      log_prob_current = log_prob_prop;
      n_accept++;
    }
    if (divergent) n_divergent++;

    // Adaptation during warmup
    if (is_warmup) {
      epsilon = da.update(alpha);
      // Collect samples for mass matrix estimation
      mass_stats.update(q);
    }

    // Store sample
    if (!is_warmup) {
      for (int i = 0; i < n_params; i++) {
        result.samples[sample_idx][i] = q[i];
      }
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      sample_idx++;
    }

    // Note: verbose output disabled in parallel - not thread-safe
    // Progress will be reported after parallel region
  }

  result.epsilon = da.final_epsilon();

  return result;
}

// R wrapper version - for single chain or non-parallel use
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose
) {
  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcerr << "Chain " << (chain_id + 1) << " complete. "
                << "Divergent: " << n_div << std::endl;
  }

  return result;
}

// =====================================================================
// Run multiple chains in parallel using OpenMP
// =====================================================================

std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose
) {
  ParamLayout layout = compute_param_layout(data);
  int n_params = layout.total_params;

  // Use pure C++ containers in parallel region
  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Check if we can safely parallelize
  // Autodiff uses a global tape that is NOT thread-safe, so we must run
  // chains sequentially for models that require autodiff gradients
  bool can_parallelize = can_use_analytical_gradient(data, layout) ||
                         layout.is_gp || layout.is_multiscale_gp;  // GP uses numerical

#ifdef _OPENMP
  if (can_parallelize && n_chains > 1) {
    // Run chains in parallel - only for analytical/numerical gradient models
    #pragma omp parallel for schedule(static) num_threads(n_chains)
    for (int c = 0; c < n_chains; c++) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init, data, layout,
        n_iter, n_warmup, L, c, seed, false  // verbose=false in parallel
      );
    }
  } else {
    // Sequential execution for autodiff models (global tape not thread-safe)
    for (int c = 0; c < n_chains; c++) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init, data, layout,
        n_iter, n_warmup, L, c, seed, verbose
      );
    }
  }
#else
  // Sequential fallback when OpenMP not available
  for (int c = 0; c < n_chains; c++) {
    cpp_results[c] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, c, seed, verbose
    );
  }
#endif

  // Convert to R objects outside parallel region (single-threaded)
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);

    if (verbose && can_parallelize) {
      // Only print here if we ran in parallel (otherwise already printed)
      int n_div = 0;
      for (int i = 0; i < cpp_results[c].n_sample; i++) {
        n_div += cpp_results[c].divergent[i];
      }
      Rcpp::Rcerr << "Chain " << (c + 1) << " complete. "
                  << "Divergent: " << n_div << std::endl;
    }
  }

  return results;
}

} // namespace ratiod_hmc

// =====================================================================
// R EXPORTS
// =====================================================================

// HMC sampler with bundled list arguments to avoid R's 65-arg limit for .Call
// Parameters are bundled into logical groups:
//   re_params: random effects (group, n_groups, n_terms, group_matrix, slopes, etc.)
//   spatial_params: spatial structure (type, group, adjacency, etc.)
//   temporal_params: temporal structure (type, time_idx, group_idx, etc.)
//   prior_params: prior hyperparameters
//   zi_params: zero-inflation (type, X_zi, prior_sd)
//   latent_params: latent factors
//   st_params: spatiotemporal interaction
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose
) {
  using namespace ratiod_hmc;

  // =========================================================================
  // Extract bundled parameters from lists with defensive checks
  // =========================================================================

  // Random effects parameters
  // Use eager deep copies to prevent R GC from invalidating memory during HMC
  std::vector<int> re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
  int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  int n_re_terms = Rcpp::as<int>(re_params["n_terms"]);

  // Handle group_matrix which may be numeric or integer
  Rcpp::IntegerMatrix re_group_matrix;
  SEXP group_mat_sexp = re_params["group_matrix"];
  if (Rf_isMatrix(group_mat_sexp)) {
    if (TYPEOF(group_mat_sexp) == INTSXP) {
      re_group_matrix = Rcpp::as<Rcpp::IntegerMatrix>(group_mat_sexp);
    } else {
      // Convert numeric to integer
      Rcpp::NumericMatrix nm(group_mat_sexp);
      re_group_matrix = Rcpp::IntegerMatrix(nm.nrow(), nm.ncol());
      for (int i = 0; i < nm.nrow(); i++) {
        for (int j = 0; j < nm.ncol(); j++) {
          re_group_matrix(i, j) = static_cast<int>(nm(i, j));
        }
      }
    }
  } else {
    // Create dummy matrix
    re_group_matrix = Rcpp::IntegerMatrix(1, 1);
    re_group_matrix(0, 0) = 0;
  }

  std::vector<int> re_n_groups_vec = Rcpp::as<std::vector<int>>(re_params["n_groups_vec"]);
  bool has_re_slopes = Rcpp::as<bool>(re_params["has_slopes"]);
  bool has_re_correlated_slopes = Rcpp::as<bool>(re_params["has_correlated_slopes"]);
  std::vector<int> re_n_coefs_vec = Rcpp::as<std::vector<int>>(re_params["n_coefs_vec"]);
  std::vector<int> re_correlated_vec = Rcpp::as<std::vector<int>>(re_params["correlated_vec"]);
  std::vector<int> re_n_chol_vec = Rcpp::as<std::vector<int>>(re_params["n_chol_vec"]);
  Rcpp::List slope_matrices_list = re_params["slope_matrices"];

  // Spatial parameters (eager deep copies)
  std::string spatial_type_str = Rcpp::as<std::string>(spatial_params["type"]);
  std::vector<int> spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
  int n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  std::vector<int> adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
  std::vector<int> adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
  std::vector<int> n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
  double bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);

  // Temporal parameters (eager deep copies)
  std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
  std::vector<int> temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
  std::vector<int> temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
  int n_times = Rcpp::as<int>(temporal_params["n_times"]);
  int n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
  int n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
  bool temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
  bool temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
  double tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
  double tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);

  // Prior parameters
  double sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  double phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  double tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
  double tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);

  // Zero-inflation parameters
  std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);

  // Handle X_zi which may be numeric matrix or NULL
  Rcpp::NumericMatrix X_zi;
  SEXP xzi_sexp = zi_params["X"];
  if (!Rf_isNull(xzi_sexp) && Rf_isMatrix(xzi_sexp)) {
    X_zi = Rcpp::as<Rcpp::NumericMatrix>(xzi_sexp);
  } else {
    // Create empty dummy matrix
    X_zi = Rcpp::NumericMatrix(1, 1);
    X_zi(0, 0) = 1.0;
  }
  double zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);

  // Latent factor parameters
  bool has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  int latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
  bool latent_shared = Rcpp::as<bool>(latent_params["shared"]);
  bool latent_scale = Rcpp::as<bool>(latent_params["scale"]);
  int latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
  double latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);

  // =========================================================================
  // Set up model data
  // =========================================================================
  ModelData data;

  // Copy response data
  data.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());

  // Flatten design matrices for cache efficiency
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();
  data.N = y_num.size();

  data.X_num_flat.resize(data.N * data.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_num; j++) {
      data.X_num_flat[i * data.p_num + j] = X_num(i, j);
    }
  }

  data.X_denom_flat.resize(data.N * data.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_denom; j++) {
      data.X_denom_flat[i * data.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects (already deep copied above)
  data.re_group = re_group;
  data.n_re_groups = n_re_groups;
  data.n_re_terms = n_re_terms;

  // Random slopes flags
  data.has_re_slopes = has_re_slopes;
  data.has_re_correlated_slopes = has_re_correlated_slopes;

  if (n_re_terms > 0) {
    // Multi-term RE structure (with or without slopes)
    data.re_group_multi.resize(n_re_terms);
    data.re_n_groups_multi.resize(n_re_terms);
    data.re_offsets.resize(n_re_terms);
    data.re_n_coefs.resize(n_re_terms);
    data.re_correlated.resize(n_re_terms);
    data.re_n_chol.resize(n_re_terms);
    data.re_n_slopes.resize(n_re_terms);
    data.re_slope_matrices.resize(n_re_terms);

    int offset = 0;
    int total_re_params = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;

    for (int t = 0; t < n_re_terms; t++) {
      data.re_n_groups_multi[t] = re_n_groups_vec[t];
      data.re_offsets[t] = offset;

      // Slopes metadata
      int n_coefs = has_re_slopes ? re_n_coefs_vec[t] : 1;
      data.re_n_coefs[t] = n_coefs;
      data.re_correlated[t] = has_re_slopes ? (re_correlated_vec[t] != 0) : false;
      data.re_n_chol[t] = has_re_slopes ? re_n_chol_vec[t] : 0;
      data.re_n_slopes[t] = n_coefs - 1;  // Number of slopes (excluding intercept)

      // Process slope matrix for this term
      if (has_re_slopes && data.re_n_slopes[t] > 0 && slope_matrices_list.size() > t) {
        SEXP mat_sexp = slope_matrices_list[t];
        if (!Rf_isNull(mat_sexp)) {
          Rcpp::NumericMatrix slope_mat(mat_sexp);
          int n_rows = slope_mat.nrow();
          int n_cols = slope_mat.ncol();
          data.re_slope_matrices[t].resize(n_rows * n_cols);
          for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
              data.re_slope_matrices[t][i * n_cols + j] = slope_mat(i, j);
            }
          }
        }
      }

      offset += re_n_groups_vec[t];
      total_re_params += re_n_groups_vec[t] * n_coefs;
      total_sigma_params += n_coefs;
      total_chol_params += data.re_n_chol[t];

      // Extract column t from re_group_matrix
      data.re_group_multi[t].resize(data.N);
      for (int i = 0; i < data.N; i++) {
        data.re_group_multi[t][i] = re_group_matrix(i, t);
      }
    }
    data.total_re_groups = offset;
    data.total_re_params = total_re_params;
    data.total_sigma_params = total_sigma_params;
    data.total_chol_params = total_chol_params;
  } else {
    // No RE terms
    data.total_re_groups = n_re_groups;
    data.total_re_params = n_re_groups;
    data.total_sigma_params = (n_re_groups > 0) ? 1 : 0;
    data.total_chol_params = 0;
  }

  // Model type
  if (model_type_str == "binomial") {
    data.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.model_type = ModelType::NEGBIN_NEGBIN;
  } else {
    data.model_type = ModelType::POISSON_GAMMA;
  }

  // Spatial structure
  if (spatial_type_str == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (spatial_type_str == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  data.spatial_group = spatial_group;  // Already deep copied above
  data.n_spatial_units = n_spatial_units;
  data.adj_row_ptr = adj_row_ptr;
  data.adj_col_idx = adj_col_idx;
  data.n_neighbors = n_neighbors;
  data.bym2_scale_factor = bym2_scale_factor;

  // Temporal structure
  if (temporal_type_str == "rw1") {
    data.temporal_type = TemporalType::RW1;
  } else if (temporal_type_str == "rw2") {
    data.temporal_type = TemporalType::RW2;
  } else if (temporal_type_str == "ar1") {
    data.temporal_type = TemporalType::AR1;
  } else {
    data.temporal_type = TemporalType::NONE;
  }

  data.temporal_time_idx = temporal_time_idx;  // Already deep copied above
  data.temporal_group_idx = temporal_group_idx;
  data.n_times = n_times;
  data.n_temporal_groups = n_temporal_groups;
  data.n_temporal_params = n_temporal_params;
  data.temporal_cyclic = temporal_cyclic;
  data.temporal_shared = temporal_shared;
  data.tau_temporal_shape = tau_temporal_shape;
  data.tau_temporal_rate = tau_temporal_rate;

  // Zero-inflation structure
  data.zi_type = ratiod_zi::parse_zi_type(zi_type_str);
  data.p_zi = X_zi.ncol();
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // Priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = tau_spatial_shape;
  data.tau_spatial_rate = tau_spatial_rate;

  // Parallelization
  data.n_threads = n_threads;

  // Initialize feature flags that are not used in cpp_hmc_fit (only in cpp_hmc_fit_gp)
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_multiscale_temporal = false;
  data.has_rsr = false;
  data.has_svc = false;

  // Latent factors
  data.has_latent = has_latent;
  data.latent_n_factors = latent_n_factors;
  data.latent_shared = latent_shared;
  data.latent_scale = latent_scale;
  data.latent_constraint = latent_constraint;
  data.latent_sigma_prior_rate = latent_sigma_prior_rate;

  // Spatiotemporal interaction - extract from list
  bool has_spatiotemporal = st_params.size() > 0 && Rcpp::as<bool>(st_params["has_spatiotemporal"]);
  data.has_spatiotemporal = has_spatiotemporal;
  if (has_spatiotemporal) {
    // Extract parameters from list (eager deep copies to prevent R GC issues)
    std::string st_type_str = Rcpp::as<std::string>(st_params["type"]);
    bool st_shared = Rcpp::as<bool>(st_params["shared"]);
    int st_n_spatial = Rcpp::as<int>(st_params["n_spatial"]);
    int st_n_times = Rcpp::as<int>(st_params["n_times"]);
    int st_n_params = Rcpp::as<int>(st_params["n_params"]);
    std::vector<int> st_s_idx = Rcpp::as<std::vector<int>>(st_params["s_idx"]);
    std::vector<int> st_t_idx = Rcpp::as<std::vector<int>>(st_params["t_idx"]);
    std::vector<int> st_flat = Rcpp::as<std::vector<int>>(st_params["st_flat"]);
    std::string st_temporal_type_str = Rcpp::as<std::string>(st_params["temporal_type"]);
    bool st_temporal_cyclic = Rcpp::as<bool>(st_params["temporal_cyclic"]);
    std::vector<int> st_adj_row_ptr = Rcpp::as<std::vector<int>>(st_params["adj_row_ptr"]);
    std::vector<int> st_adj_col_idx = Rcpp::as<std::vector<int>>(st_params["adj_col_idx"]);
    double st_sigma2_prior_U = Rcpp::as<double>(st_params["sigma2_prior_U"]);
    double st_sigma2_prior_alpha = Rcpp::as<double>(st_params["sigma2_prior_alpha"]);

    // Parse ST type
    if (st_type_str == "type_i") {
      data.spatiotemporal_data.type = STType::TYPE_I;
    } else if (st_type_str == "type_ii") {
      data.spatiotemporal_data.type = STType::TYPE_II;
    } else if (st_type_str == "type_iii") {
      data.spatiotemporal_data.type = STType::TYPE_III;
    } else if (st_type_str == "type_iv") {
      data.spatiotemporal_data.type = STType::TYPE_IV;
    } else if (st_type_str == "separable") {
      data.spatiotemporal_data.type = STType::SEPARABLE;
    } else if (st_type_str == "nonsep_gp") {
      data.spatiotemporal_data.type = STType::NONSEP_GP;
    } else {
      data.spatiotemporal_data.type = STType::NONE;
    }

    data.spatiotemporal_data.shared = st_shared;
    data.spatiotemporal_data.n_spatial = st_n_spatial;
    data.spatiotemporal_data.n_times = st_n_times;
    data.spatiotemporal_data.n_params = st_n_params;

    // Observation indexing (already deep copied above)
    data.spatiotemporal_data.s_idx = st_s_idx;
    data.spatiotemporal_data.t_idx = st_t_idx;
    data.spatiotemporal_data.st_flat = st_flat;

    // Temporal type for Type II/IV
    if (st_temporal_type_str == "rw1") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;
    } else if (st_temporal_type_str == "rw2") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW2;
    } else if (st_temporal_type_str == "ar1") {
      data.spatiotemporal_data.temporal_type = TemporalType::AR1;
    } else {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;  // Default
    }
    data.spatiotemporal_data.temporal_cyclic = st_temporal_cyclic;

    // Spatial adjacency for Type III/IV (already deep copied above)
    data.spatiotemporal_data.adj_row_ptr = st_adj_row_ptr;
    data.spatiotemporal_data.adj_col_idx = st_adj_col_idx;

    // Prior parameters
    data.st_sigma2_prior_U = st_sigma2_prior_U;
    data.st_sigma2_prior_alpha = st_sigma2_prior_alpha;
  } else {
    data.spatiotemporal_data.type = STType::NONE;
  }

  // Initialize parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Memory barrier to ensure all copies complete before HMC execution
  // This prevents R GC from invalidating memory during sampling
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains
    );
  }
}

// [[Rcpp::export]]
int cpp_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// HMC sampler for GP-based spatial models
// Parameters are bundled into lists to avoid R's .Call argument limit:
//   gp_params: GP spatial parameters
//   ms_gp_params: multiscale GP parameters
//   ms_temporal_params: multiscale temporal parameters
//   rsr_params: RSR parameters
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    std::string model_type_str,
    Rcpp::List gp_params,
    Rcpp::List ms_gp_params,
    Rcpp::List ms_temporal_params,
    Rcpp::List rsr_params,
    double sigma_beta,
    double sigma_re_scale,
    double phi_prior_shape,
    double phi_prior_rate,
    std::string zi_type_str,
    Rcpp::NumericMatrix X_zi,
    double zi_prior_sd,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose
) {
  // Force all Rcpp parameter extractions into eagerly-copied std::vectors FIRST
  // This prevents R garbage collection from invalidating lazy Rcpp views during C++ execution
  // The original debug output workaround worked because I/O forced R to sync; this achieves
  // the same effect through explicit eager copying without visible output.

  // Extract GP parameters - convert to native C++ types immediately
  std::string gp_type_str = Rcpp::as<std::string>(gp_params["gp_type"]);

  // Force eager copy into std::vectors (not Rcpp views that could be GC'd)
  std::vector<double> coords_vec = Rcpp::as<std::vector<double>>(gp_params["coords"]);
  std::vector<int> nn_idx_vec = Rcpp::as<std::vector<int>>(gp_params["nn_idx"]);
  std::vector<double> nn_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_dist"]);
  std::vector<int> nn_order_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order"]);
  std::vector<int> nn_order_inv_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order_inv"]);

  int nn = Rcpp::as<int>(gp_params["nn"]);
  std::string cov_type_str = Rcpp::as<std::string>(gp_params["cov_type"]);
  double nu = Rcpp::as<double>(gp_params["nu"]);
  bool gp_shared = Rcpp::as<bool>(gp_params["shared"]);
  double gp_sigma2_prior_U = Rcpp::as<double>(gp_params["sigma2_prior_U"]);
  double gp_sigma2_prior_alpha = Rcpp::as<double>(gp_params["sigma2_prior_alpha"]);
  double gp_phi_prior_lower = Rcpp::as<double>(gp_params["phi_prior_lower"]);
  double gp_phi_prior_upper = Rcpp::as<double>(gp_params["phi_prior_upper"]);

  // Memory barrier to ensure all extractions complete before proceeding
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Extract multiscale GP parameters - eager copy to std::vectors
  std::vector<int> nn_idx_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_local"]);
  std::vector<double> nn_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_local"]);
  std::vector<int> nn_order_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_local"]);
  std::vector<int> nn_order_inv_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_local"]);
  int nn_local = Rcpp::as<int>(ms_gp_params["nn_local"]);
  std::vector<int> nn_idx_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_regional"]);
  std::vector<double> nn_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_regional"]);
  std::vector<int> nn_order_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_regional"]);
  std::vector<int> nn_order_inv_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_regional"]);
  int nn_regional = Rcpp::as<int>(ms_gp_params["nn_regional"]);
  double range_local_lower = Rcpp::as<double>(ms_gp_params["range_local_lower"]);
  double range_local_upper = Rcpp::as<double>(ms_gp_params["range_local_upper"]);
  double range_regional_lower = Rcpp::as<double>(ms_gp_params["range_regional_lower"]);
  double range_regional_upper = Rcpp::as<double>(ms_gp_params["range_regional_upper"]);
  double ms_sigma2_local_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_U"]);
  double ms_sigma2_local_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_alpha"]);
  double ms_sigma2_regional_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_U"]);
  double ms_sigma2_regional_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_alpha"]);

  // Extract multiscale temporal parameters - eager copy
  std::string ms_temporal_type_str = Rcpp::as<std::string>(ms_temporal_params["type"]);
  std::vector<int> ms_time_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["time_index"]);
  std::vector<int> ms_group_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["group_index"]);
  int ms_n_times = Rcpp::as<int>(ms_temporal_params["n_times"]);
  int ms_n_groups = Rcpp::as<int>(ms_temporal_params["n_groups"]);
  std::string trend_type_str = Rcpp::as<std::string>(ms_temporal_params["trend_type"]);
  int seasonal_period = Rcpp::as<int>(ms_temporal_params["seasonal_period"]);
  std::string short_term_type_str = Rcpp::as<std::string>(ms_temporal_params["short_term_type"]);
  bool ms_temporal_shared = Rcpp::as<bool>(ms_temporal_params["shared"]);
  double ms_sigma2_trend_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_U"]);
  double ms_sigma2_trend_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_alpha"]);
  double ms_sigma2_seasonal_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_U"]);
  double ms_sigma2_seasonal_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_alpha"]);
  double ms_sigma2_short_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_U"]);
  double ms_sigma2_short_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_alpha"]);

  // Extract RSR parameters - eager copy
  bool has_rsr = Rcpp::as<bool>(rsr_params["has_rsr"]);
  std::vector<double> rsr_projection_vec = Rcpp::as<std::vector<double>>(rsr_params["projection"]);
  int rsr_n = Rcpp::as<int>(rsr_params["n"]);

  // Second memory barrier after all Rcpp extractions
  std::atomic_thread_fence(std::memory_order_seq_cst);
  using namespace ratiod_hmc;

  // Set up model data
  ModelData data;

  // Copy response data
  data.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());
  data.N = y_num.size();

  // Flatten design matrices
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();

  data.X_num_flat.resize(data.N * data.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_num; j++) {
      data.X_num_flat[i * data.p_num + j] = X_num(i, j);
    }
  }

  data.X_denom_flat.resize(data.N * data.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_denom; j++) {
      data.X_denom_flat[i * data.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects (single-term legacy path)
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;

  // Initialize multi-term RE fields to indicate single-term mode
  data.n_re_terms = 0;  // 0 means use legacy single-term path
  data.total_re_groups = n_re_groups;
  data.has_re_slopes = false;  // GP interface doesn't support random slopes
  data.has_re_correlated_slopes = false;

  // Model type
  if (model_type_str == "binomial") {
    data.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.model_type = ModelType::NEGBIN_NEGBIN;
  } else {
    data.model_type = ModelType::POISSON_GAMMA;
  }

  // Covariance type
  ratiod_gp::CovType cov_type;
  if (cov_type_str == "exponential") {
    cov_type = ratiod_gp::CovType::EXPONENTIAL;
  } else if (cov_type_str == "matern") {
    cov_type = ratiod_gp::CovType::MATERN;
  } else if (cov_type_str == "gaussian") {
    cov_type = ratiod_gp::CovType::GAUSSIAN;
  } else {
    cov_type = ratiod_gp::CovType::SPHERICAL;
  }

  // GP spatial structure
  if (gp_type_str == "gp") {
    data.spatial_type = SpatialType::GP;
    data.has_gp = true;
    data.has_multiscale_gp = false;

    data.gp_data.n_obs = data.N;
    data.gp_data.nn = nn;
    data.gp_data.coords = coords_vec;  // Already std::vector from eager copy
    data.gp_data.nn_idx = nn_idx_vec;
    data.gp_data.nn_dist = nn_dist_vec;
    // Convert from R's 1-based to C++'s 0-based indexing
    data.gp_data.nn_order.resize(nn_order_vec.size());
    for (size_t i = 0; i < nn_order_vec.size(); i++) {
      data.gp_data.nn_order[i] = nn_order_vec[i] - 1;
    }
    data.gp_data.nn_order_inv.resize(nn_order_inv_vec.size());
    for (size_t i = 0; i < nn_order_inv_vec.size(); i++) {
      data.gp_data.nn_order_inv[i] = nn_order_inv_vec[i] - 1;
    }
    data.gp_data.cov_type = cov_type;
    data.gp_data.nu = nu;
    data.gp_data.shared = gp_shared;

    data.gp_sigma2_prior_U = gp_sigma2_prior_U;
    data.gp_sigma2_prior_alpha = gp_sigma2_prior_alpha;
    data.gp_phi_prior_lower = gp_phi_prior_lower;
    data.gp_phi_prior_upper = gp_phi_prior_upper;

  } else if (gp_type_str == "multiscale_gp") {
    data.spatial_type = SpatialType::MULTISCALE_GP;
    data.has_gp = false;
    data.has_multiscale_gp = true;

    data.multiscale_gp_data.n_obs = data.N;
    data.multiscale_gp_data.coords = coords_vec;  // Already std::vector from eager copy

    // Local scale - use pre-copied std::vectors
    data.multiscale_gp_data.nn_local = nn_local;
    data.multiscale_gp_data.nn_idx_local = nn_idx_local_vec;
    data.multiscale_gp_data.nn_dist_local = nn_dist_local_vec;
    // Convert from R's 1-based to C++'s 0-based indexing
    data.multiscale_gp_data.nn_order_local.resize(nn_order_local_vec.size());
    for (size_t i = 0; i < nn_order_local_vec.size(); i++) {
      data.multiscale_gp_data.nn_order_local[i] = nn_order_local_vec[i] - 1;
    }
    data.multiscale_gp_data.nn_order_inv_local.resize(nn_order_inv_local_vec.size());
    for (size_t i = 0; i < nn_order_inv_local_vec.size(); i++) {
      data.multiscale_gp_data.nn_order_inv_local[i] = nn_order_inv_local_vec[i] - 1;
    }

    // Regional scale - use pre-copied std::vectors
    data.multiscale_gp_data.nn_regional = nn_regional;
    data.multiscale_gp_data.nn_idx_regional = nn_idx_regional_vec;
    data.multiscale_gp_data.nn_dist_regional = nn_dist_regional_vec;
    // Convert from R's 1-based to C++'s 0-based indexing
    data.multiscale_gp_data.nn_order_regional.resize(nn_order_regional_vec.size());
    for (size_t i = 0; i < nn_order_regional_vec.size(); i++) {
      data.multiscale_gp_data.nn_order_regional[i] = nn_order_regional_vec[i] - 1;
    }
    data.multiscale_gp_data.nn_order_inv_regional.resize(nn_order_inv_regional_vec.size());
    for (size_t i = 0; i < nn_order_inv_regional_vec.size(); i++) {
      data.multiscale_gp_data.nn_order_inv_regional[i] = nn_order_inv_regional_vec[i] - 1;
    }

    // Range constraints
    data.multiscale_gp_data.range_local_lower = range_local_lower;
    data.multiscale_gp_data.range_local_upper = range_local_upper;
    data.multiscale_gp_data.range_regional_lower = range_regional_lower;
    data.multiscale_gp_data.range_regional_upper = range_regional_upper;

    data.multiscale_gp_data.cov_type = cov_type;
    data.multiscale_gp_data.nu = nu;
    data.multiscale_gp_data.shared = gp_shared;

    data.ms_sigma2_local_prior_U = ms_sigma2_local_prior_U;
    data.ms_sigma2_local_prior_alpha = ms_sigma2_local_prior_alpha;
    data.ms_sigma2_regional_prior_U = ms_sigma2_regional_prior_U;
    data.ms_sigma2_regional_prior_alpha = ms_sigma2_regional_prior_alpha;

  } else {
    data.spatial_type = SpatialType::NONE;
    data.has_gp = false;
    data.has_multiscale_gp = false;
  }

  // Initialize adjacency for ICAR/BYM2 (not used with GP)
  data.n_spatial_units = 0;
  data.bym2_scale_factor = 1.0;

  // Multi-scale temporal structure
  if (ms_temporal_type_str == "multiscale") {
    data.has_multiscale_temporal = true;

    data.multiscale_temporal_data.n_times = ms_n_times;
    data.multiscale_temporal_data.n_groups = ms_n_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = ms_time_index_vec;  // Already std::vector from eager copy
    data.multiscale_temporal_data.group_index = ms_group_index_vec;
    data.multiscale_temporal_data.shared = ms_temporal_shared;
    data.multiscale_temporal_data.seasonal_period = seasonal_period;

    // Parse temporal component types
    data.multiscale_temporal_data.trend_type = ratiod_temporal::parse_temporal_type(trend_type_str);
    data.multiscale_temporal_data.short_term_type = ratiod_temporal::parse_temporal_type(short_term_type_str);

    data.ms_sigma2_trend_prior_U = ms_sigma2_trend_prior_U;
    data.ms_sigma2_trend_prior_alpha = ms_sigma2_trend_prior_alpha;
    data.ms_sigma2_seasonal_prior_U = ms_sigma2_seasonal_prior_U;
    data.ms_sigma2_seasonal_prior_alpha = ms_sigma2_seasonal_prior_alpha;
    data.ms_sigma2_short_prior_U = ms_sigma2_short_prior_U;
    data.ms_sigma2_short_prior_alpha = ms_sigma2_short_prior_alpha;

  } else {
    data.has_multiscale_temporal = false;
    data.multiscale_temporal_data.trend_type = ratiod_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.short_term_type = ratiod_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.seasonal_period = 0;
  }

  // Legacy temporal (not used with multiscale)
  data.temporal_type = TemporalType::NONE;
  data.n_times = 0;
  data.n_temporal_groups = 0;
  data.n_temporal_params = 0;
  data.temporal_cyclic = false;
  data.temporal_shared = false;
  data.tau_temporal_shape = 1.0;
  data.tau_temporal_rate = 0.01;

  // Multi-term RE structure (not used in GP interface - single term only)
  data.total_re_params = 0;
  data.total_sigma_params = 0;
  data.total_chol_params = 0;

  // RSR structure - use pre-copied std::vector
  data.has_rsr = has_rsr;
  if (has_rsr && !rsr_projection_vec.empty()) {
    data.rsr_projection = rsr_projection_vec;
    data.rsr_n = rsr_n;
  } else {
    data.rsr_n = 0;
  }

  // Zero-inflation structure
  data.zi_type = ratiod_zi::parse_zi_type(zi_type_str);
  data.p_zi = X_zi.ncol();
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // Standard priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = 1.0;
  data.tau_spatial_rate = 0.01;

  // SVC not used in GP interface
  data.has_svc = false;

  // Latent factors not used in GP interface
  data.has_latent = false;
  data.latent_n_factors = 0;
  data.latent_shared = false;
  data.latent_scale = false;
  data.latent_constraint = 0;
  data.latent_sigma_prior_rate = 1.0;

  // Spatiotemporal not used in GP interface
  data.has_spatiotemporal = false;
  data.spatiotemporal_data.type = STType::NONE;

  // Parallelization
  data.n_threads = n_threads;

  // Final memory barrier before HMC execution
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Initialize parameters - use explicit std::vector copy from Rcpp
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains
    );
  }
}

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp_v2(Rcpp::List args) {
  // O2-safe interface: single List parameter to minimize Rcpp template instantiation
  // at ABI boundary. All parameter extraction happens inside function body where
  // compiler has full visibility.

  // Extract all parameters from the list - matching cpp_hmc_fit_gp signature
  Rcpp::NumericVector q_init = Rcpp::as<Rcpp::NumericVector>(args["q_init"]);
  Rcpp::IntegerVector y_num = Rcpp::as<Rcpp::IntegerVector>(args["y_num"]);
  Rcpp::IntegerVector y_denom = Rcpp::as<Rcpp::IntegerVector>(args["y_denom"]);
  Rcpp::NumericVector y_denom_cont = Rcpp::as<Rcpp::NumericVector>(args["y_denom_cont"]);
  Rcpp::NumericMatrix X_num = Rcpp::as<Rcpp::NumericMatrix>(args["X_num"]);
  Rcpp::NumericMatrix X_denom = Rcpp::as<Rcpp::NumericMatrix>(args["X_denom"]);
  Rcpp::IntegerVector re_group = Rcpp::as<Rcpp::IntegerVector>(args["re_group"]);
  int n_re_groups = Rcpp::as<int>(args["n_re_groups"]);
  std::string model_type_str = Rcpp::as<std::string>(args["model_type_str"]);
  Rcpp::List gp_params = Rcpp::as<Rcpp::List>(args["gp_params"]);
  Rcpp::List ms_gp_params = Rcpp::as<Rcpp::List>(args["ms_gp_params"]);
  Rcpp::List ms_temporal_params = Rcpp::as<Rcpp::List>(args["ms_temporal_params"]);
  Rcpp::List rsr_params = Rcpp::as<Rcpp::List>(args["rsr_params"]);
  double sigma_beta = Rcpp::as<double>(args["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(args["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(args["phi_prior_shape"]);
  double phi_prior_rate = Rcpp::as<double>(args["phi_prior_rate"]);
  std::string zi_type_str = Rcpp::as<std::string>(args["zi_type_str"]);
  Rcpp::NumericMatrix X_zi = Rcpp::as<Rcpp::NumericMatrix>(args["X_zi"]);
  double zi_prior_sd = Rcpp::as<double>(args["zi_prior_sd"]);
  int n_iter = Rcpp::as<int>(args["n_iter"]);
  int n_warmup = Rcpp::as<int>(args["n_warmup"]);
  int L = Rcpp::as<int>(args["L"]);
  int n_chains = Rcpp::as<int>(args["n_chains"]);
  unsigned int seed = Rcpp::as<unsigned int>(args["seed"]);
  int n_threads = Rcpp::as<int>(args["n_threads"]);
  bool verbose = Rcpp::as<bool>(args["verbose"]);

  // Delegate to the original implementation
  return cpp_hmc_fit_gp(
    q_init, y_num, y_denom, y_denom_cont,
    X_num, X_denom, re_group, n_re_groups,
    model_type_str, gp_params, ms_gp_params, ms_temporal_params, rsr_params,
    sigma_beta, sigma_re_scale, phi_prior_shape, phi_prior_rate,
    zi_type_str, X_zi, zi_prior_sd,
    n_iter, n_warmup, L, n_chains, seed, n_threads, verbose
  );
}
