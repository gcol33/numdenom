// hmc_sampler.cpp
// Full HMC/NUTS backend with spatial, temporal, and ZI support
// Provides Stan-free Bayesian inference for all ratiod models

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_progress.h"
#include "autodiff.h"
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
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
// Dense mass matrix: Cholesky decomposition via Eigen
// =====================================================================

bool DenseMassMatrix::update_from_covariance(const double* cov, int n_samples) {
  // Map the covariance data into an Eigen matrix (column-major)
  Eigen::Map<const Eigen::MatrixXd> C(cov, n, n);

  // Cholesky decomposition: LL^T = C
  Eigen::LLT<Eigen::MatrixXd> llt(C);
  if (llt.info() != Eigen::Success) {
    // Cholesky failed — degrade to diagonal
    type = MassMatrixType::DIAG;
    adapted = true;
    // Extract diagonal from covariance as fallback
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  // Store full covariance as inv_mass_dense
  std::memcpy(inv_mass_dense.data(), cov, static_cast<size_t>(n) * n * sizeof(double));

  // Store Cholesky factor L
  Eigen::MatrixXd L_mat = llt.matrixL();
  std::memcpy(L_inv_mass.data(), L_mat.data(), static_cast<size_t>(n) * n * sizeof(double));

  // Also update diagonal for fallback and find_reasonable_epsilon compatibility
  for (int i = 0; i < n; i++) {
    double var_i = cov[static_cast<size_t>(i) * n + i];
    inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
    sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
  }

  adapted = true;
  return true;
}

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

  // Overdispersion / shape / sigma parameters
  // NEGBIN_NEGBIN: phi_num (overdispersion for num), phi_denom (overdispersion for denom)
  // POISSON_GAMMA: phi_num (shape for gamma denom)
  // GAMMA_GAMMA: phi_num (shape for num), phi_denom (shape for denom)
  // LOGNORMAL: phi_num (sigma for num), phi_denom (sigma for denom)
  // BETA_BINOMIAL: phi_num (precision parameter)
  layout.has_phi_num = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                        data.model_type == ModelType::POISSON_GAMMA ||
                        data.model_type == ModelType::GAMMA_GAMMA ||
                        data.model_type == ModelType::LOGNORMAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);
  layout.has_phi_denom = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.model_type == ModelType::GAMMA_GAMMA ||
                          data.model_type == ModelType::LOGNORMAL);

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
      // BYM2 Riebler: log_sigma_total, logit_rho, phi_scaled, theta
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
  layout.is_temporal_gp = (data.temporal_type == TemporalType::GP);

  if (layout.has_temporal) {
    if (layout.is_temporal_gp) {
      // Temporal GP: log_sigma2 + log_phi + effects
      layout.log_sigma2_temporal_gp_idx = idx++;
      layout.log_phi_temporal_gp_idx = idx++;
      layout.log_tau_temporal_idx = -1;  // Not used for GP
      layout.logit_rho_ar1_idx = -1;
    } else {
      // RW1/RW2/AR1: log_tau + effects (+ rho for AR1)
      layout.log_tau_temporal_idx = idx++;
      layout.log_sigma2_temporal_gp_idx = -1;
      layout.log_phi_temporal_gp_idx = -1;

      // AR1 also has rho parameter
      if (layout.is_ar1) {
        layout.logit_rho_ar1_idx = idx++;
      } else {
        layout.logit_rho_ar1_idx = -1;
      }
    }

    // Temporal effects: n_times * n_groups parameters
    layout.temporal_start = idx;
    idx += data.n_temporal_params;
    layout.temporal_end = idx;
  } else {
    layout.log_tau_temporal_idx = -1;
    layout.logit_rho_ar1_idx = -1;
    layout.log_sigma2_temporal_gp_idx = -1;
    layout.log_phi_temporal_gp_idx = -1;
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

  // One-inflation parameters (for OI-binomial and ZOIB)
  layout.has_oi = (data.zi_type == ZIType::OI_BINOMIAL || data.zi_type == ZIType::ZOIB);

  if (layout.has_oi && data.p_oi > 0) {
    layout.beta_oi_start = idx;
    idx += data.p_oi;
    layout.beta_oi_end = idx;
  } else {
    layout.beta_oi_start = layout.beta_oi_end = -1;
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

  // SVC (Spatially-Varying Coefficients) parameters
  layout.has_svc = data.has_svc;
  if (layout.has_svc && data.svc_data.n_svc > 0) {
    // Log sigma2 per SVC term (spatial variance)
    layout.log_sigma2_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.log_sigma2_svc_end = idx;

    // Log phi per SVC term (spatial range)
    layout.log_phi_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.log_phi_svc_end = idx;

    // SVC coefficients: w_flat[j * n_obs + i] = w_j(s_i)
    // Layout: w_flat[j * n_obs + i] for j in 0..n_svc-1, i in 0..n_obs-1
    layout.svc_w_start = idx;
    idx += data.svc_data.n_svc * data.svc_data.n_obs;
    layout.svc_w_end = idx;
  } else {
    layout.log_sigma2_svc_start = layout.log_sigma2_svc_end = -1;
    layout.log_phi_svc_start = layout.log_phi_svc_end = -1;
    layout.svc_w_start = layout.svc_w_end = -1;
  }

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

  // HSGP (Hilbert Space GP) parameters
  layout.is_hsgp = (data.spatial_type == SpatialType::HSGP);
  if (layout.is_hsgp && data.has_hsgp) {
    layout.log_sigma2_hsgp_idx = idx++;
    layout.log_lengthscale_hsgp_idx = idx++;
    layout.hsgp_beta_start = idx;
    idx += data.hsgp_data.m_total;  // m^2 basis coefficients
    layout.hsgp_beta_end = idx;
  } else {
    layout.log_sigma2_hsgp_idx = -1;
    layout.log_lengthscale_hsgp_idx = -1;
    layout.hsgp_beta_start = layout.hsgp_beta_end = -1;
  }

  // TVC (Temporally-Varying Coefficients) parameters
  layout.has_tvc = data.has_tvc;
  if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
    // Log precision per TVC term
    layout.log_tau_tvc_start = idx;
    idx += data.tvc_data.n_tvc;
    layout.log_tau_tvc_end = idx;

    // AR1 rho parameters (only if structure is AR1)
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
      layout.logit_rho_tvc_start = idx;
      idx += data.tvc_data.n_tvc;
      layout.logit_rho_tvc_end = idx;
    } else {
      layout.logit_rho_tvc_start = layout.logit_rho_tvc_end = -1;
    }

    // TVC values: w[g, j, t] for g in groups, j in tvc terms, t in times
    // Layout: w_flat[g * n_tvc * n_times + j * n_times + t]
    layout.tvc_w_start = idx;
    idx += data.tvc_data.n_groups * data.tvc_data.n_tvc * data.tvc_data.n_times;
    layout.tvc_w_end = idx;
  } else {
    layout.log_tau_tvc_start = layout.log_tau_tvc_end = -1;
    layout.logit_rho_tvc_start = layout.logit_rho_tvc_end = -1;
    layout.tvc_w_start = layout.tvc_w_end = -1;
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
  return std::lgamma(y + phi) - std::lgamma(phi) - std::lgamma(y + 1.0)
       + phi * std::log(phi / (mu + phi))
       + y * std::log(mu / (mu + phi));
}

inline double log_lik_poisson(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - std::lgamma(y + 1.0);
}

inline double log_lik_gamma(double y, double shape, double mu) {
  if (y <= 0 || shape <= 0 || mu <= 0) return -1e10;
  double rate = shape / mu;
  return shape * std::log(rate) + (shape - 1.0) * std::log(y)
       - rate * y - std::lgamma(shape);
}

// Include vectorized gradient header AFTER log_lik_* functions and hmc_sampler.h
// so all types and helpers are defined.
#include "hmc_gradient_vectorized.h"

// Thread-local vectorized gradient workspace (avoids per-call allocation)
static thread_local vectorized::VecGradWorkspace vec_grad_ws;

// =====================================================================
// Observation log-likelihood helper (fused with gradient computation)
// Matches compute_log_post observation loop exactly. Used by specialized
// H gradient functions to avoid a separate O(N) pass.
// =====================================================================

inline double compute_obs_ll(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom
) {
  if (data.model_type == ModelType::BINOMIAL) {
    return log_lik_binomial(data.y_num[i], data.y_denom[i], eta_num);
  } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.y_num[i], mu_num, phi_num)
         + log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_poisson(data.y_num[i], mu_num)
         + log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
  } else if (data.model_type == ModelType::GAMMA_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_gamma(data.y_num_cont[i], phi_num, mu_num)
         + log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.model_type == ModelType::LOGNORMAL) {
    double log_y_num = std::log(data.y_num_cont[i]);
    double log_y_denom = std::log(data.y_denom_cont[i]);
    double z_num = (log_y_num - eta_num) / phi_num;
    double z_denom = (log_y_denom - eta_denom) / phi_denom;
    return -log_y_num - std::log(phi_num) - 0.5 * z_num * z_num
           -log_y_denom - std::log(phi_denom) - 0.5 * z_denom * z_denom;
  } else if (data.model_type == ModelType::BETA_BINOMIAL) {
    double p = 1.0 / (1.0 + std::exp(-eta_num));
    int y = data.y_num[i];
    int n = data.y_denom[i];
    double alpha = p * phi_num;
    double beta_param = (1.0 - p) * phi_num;
    return std::lgamma(y + alpha) + std::lgamma(n - y + beta_param) - std::lgamma(n + phi_num)
         - std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num)
         + ratiod::math::portable_lchoose(n, y);
  }
  return 0.0;
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
    const ParamLayout& layout,
    bool skip_obs_loop
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
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;

  if (layout.has_spatial) {
    phi_spatial = &params[layout.spatial_start];
    if (layout.is_bym2) {
      // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
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
  // Non-centered parameterization: params store z ~ N(0,1), re = sigma * (L * z)
  // Pre-compute actual RE values from z for use in the likelihood loop.
  // Only allocate when the observation loop will run (skip_obs_loop=true avoids
  // this heap allocation on every gradient call — saves ~100ns per call).
  std::vector<double> re_nc_flat;
  if (!skip_obs_loop) {
    re_nc_flat.assign(params.size(), 0.0);
  }
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

        // Build L matrix: tanh parameterization for unconstrained HMC
        // Raw parameters are unconstrained; tanh maps them to (-1, 1)
        // L[i,j] = tanh(raw[idx]) for off-diagonal, L[i,i] = sqrt(1 - sum_j L[i,j]^2)
        // This eliminates hard boundaries that cause gradient explosions
        L_flat.resize(n_coefs * n_coefs, 0.0);

        double log_jac_tanh = 0.0;  // Jacobian for raw -> L transformation
        int chol_idx = 0;
        for (int i = 0; i < n_coefs; i++) {
          double row_sum_sq = 0.0;
          for (int j = 0; j < i; j++) {
            double raw_ij = params[chol_start + chol_idx];
            double l_ij = std::tanh(raw_ij);
            L_flat[i * n_coefs + j] = l_ij;
            row_sum_sq += l_ij * l_ij;
            // Jacobian: log|d(tanh)/d(raw)| = log(1 - tanh^2) = log(sech^2)
            double sech2 = 1.0 - l_ij * l_ij;
            log_jac_tanh += std::log(std::max(1e-300, sech2));
            chol_idx++;
          }
          // Diagonal: guaranteed positive since tanh^2 < 1
          double diag_sq = 1.0 - row_sum_sq;
          if (diag_sq < 1e-10) {
            // Safety guard (shouldn't trigger with tanh)
            return -std::numeric_limits<double>::infinity();
          }
          L_flat[i * n_coefs + i] = std::sqrt(diag_sq);
        }

        // Tanh Jacobian: maps unconstrained raw to bounded L elements
        log_post += log_jac_tanh;

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

      // Non-centered parameterization for correlated slopes:
      // Params store z ~ N(0,1), compute re = diag(sigma) * L * z
      // This eliminates the funnel geometry that plagues centered parameterization.
      // For uncorrelated slopes, params store z ~ N(0,1), re = sigma * z
      if (is_correlated && n_coefs > 1) {
        // Pre-compute re from z for all groups: re[g] = diag(sigma) * L * z[g]
        // Store in re_nc_flat for use in likelihood loop only
        if (!skip_obs_loop) {
          for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
              double Lz_c = 0.0;
              for (int k = 0; k <= c; k++) {
                Lz_c += L_flat[c * n_coefs + k] * params[re_start + g * n_coefs + k];
              }
              re_nc_flat[re_start + g * n_coefs + c] = sigmas[c] * Lz_c;
            }
          }
        }

        // N(0, I) prior on z (trivial in non-centered)
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double z_gc = params[re_start + g * n_coefs + c];
            log_post -= 0.5 * z_gc * z_gc;
          }
        }
        // No log-determinant term: in non-centered parameterization,
        // the |det(diag(sigma)*L)| from the change of variables cancels exactly
        // with the |Sigma|^{-1/2} normalization of the MVN density.

      } else {
        // Uncorrelated case: centered parameterization (unchanged)
        // params store actual re values, prior is N(0, sigma_c^2)
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
      // BYM2 Riebler: Half-Cauchy on sigma_total
      double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);  // recover sigma_total
      double ratio = sigma_total / data.sigma_re_scale;
      log_post -= std::log(1.0 + ratio * ratio);
      log_post += params[layout.log_sigma_bym2_idx];  // Jacobian for log transform

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // log p(logit_rho) = log(rho) + log(1-rho)
      log_post += std::log(rho_bym2) + std::log(1.0 - rho_bym2);

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

  // OI coefficient priors (for OI_BINOMIAL and ZOIB)
  const double* beta_oi = nullptr;
  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    // N(0, oi_prior_sd^2) prior on OI coefficients
    double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    for (int j = 0; j < data.p_oi; j++) {
      log_post -= 0.5 * tau_oi * beta_oi[j] * beta_oi[j];
    }
  }

  // Temporal priors
  double tau_temporal = 1.0, log_tau_temporal = 0.0;
  double rho_ar1 = 0.5;
  const double* phi_temporal = nullptr;
  double sigma2_temporal_gp = 1.0, phi_temporal_gp = 1.0;

  if (layout.has_temporal) {
    phi_temporal = &params[layout.temporal_start];

    if (layout.is_temporal_gp) {
      // Temporal GP: sigma2 and phi (lengthscale) parameters
      double log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
      double log_phi = params[layout.log_phi_temporal_gp_idx];
      sigma2_temporal_gp = std::exp(log_sigma2);
      phi_temporal_gp = std::exp(log_phi);

      // PC prior on sigma2 (favor smaller variance)
      log_post += ratiod_temporal_gp::log_prior_temporal_sigma2_pc(
          sigma2_temporal_gp, data.temporal_gp_sigma2_prior_U,
          data.temporal_gp_sigma2_prior_alpha);
      log_post += log_sigma2;  // Jacobian for log transform

      // Uniform prior on phi within bounds
      log_post += ratiod_temporal_gp::log_prior_temporal_phi_uniform(
          phi_temporal_gp, data.temporal_gp_phi_prior_lower,
          data.temporal_gp_phi_prior_upper);
      log_post += log_phi;  // Jacobian

      // GP log-likelihood for temporal effects
      // For each group, compute GP log-likelihood over the temporal effects
      int T = data.n_times;
      for (int g = 0; g < data.n_temporal_groups; g++) {
        std::vector<double> phi_g_vec(phi_temporal + g * T, phi_temporal + (g + 1) * T);
        log_post += ratiod_temporal_gp::temporal_gp_log_lik(
            phi_g_vec, data.temporal_gp_data, sigma2_temporal_gp, phi_temporal_gp);
      }

    } else {
      // RW1/RW2/AR1: tau-based parameterization
      log_tau_temporal = params[layout.log_tau_temporal_idx];
      tau_temporal = std::exp(log_tau_temporal);

      // tau ~ Gamma(shape, rate) with Jacobian
      log_post += (data.tau_temporal_shape - 1.0) * log_tau_temporal
                - data.tau_temporal_rate * tau_temporal + log_tau_temporal;

      // AR1: also estimate rho
      if (layout.is_ar1) {
        double logit_rho = params[layout.logit_rho_ar1_idx];
        rho_ar1 = 1.0 / (1.0 + std::exp(-logit_rho));

        // rho ~ Uniform(0,1) prior with logit Jacobian
        log_post += std::log(rho_ar1) + std::log(1.0 - rho_ar1);
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

  // HSGP (Hilbert Space GP) priors
  double sigma2_hsgp = 1.0, lengthscale_hsgp = 1.0;
  std::vector<double> hsgp_beta;
  std::vector<double> hsgp_f;

  if (layout.is_hsgp && data.has_hsgp) {
    double log_sigma2 = params[layout.log_sigma2_hsgp_idx];
    double log_lengthscale = params[layout.log_lengthscale_hsgp_idx];
    sigma2_hsgp = std::exp(log_sigma2);
    lengthscale_hsgp = std::exp(log_lengthscale);

    // Extract beta coefficients
    int m_total = data.hsgp_data.m_total;
    hsgp_beta.resize(m_total);
    for (int j = 0; j < m_total; j++) {
      hsgp_beta[j] = params[layout.hsgp_beta_start + j];
    }

    // PC prior on sigma: P(sigma > 1) = 0.01 -> rate = 4.6
    // log p(sigma) = log(rate) - rate*sigma - log(2*sigma)
    // d/d(log_sigma2) includes Jacobian
    double sigma = std::sqrt(sigma2_hsgp);
    double rate_sigma = 4.6;
    log_post += std::log(rate_sigma) - rate_sigma * sigma - std::log(2.0 * sigma);
    log_post += log_sigma2 * 0.5;  // Jacobian: d(sigma)/d(log_sigma2) = 0.5*sigma

    // LogNormal(0, 1) prior on lengthscale
    // log p(ell) = -0.5 * log(ell)^2 - log(ell)
    log_post += -0.5 * log_lengthscale * log_lengthscale - log_lengthscale;
    log_post += log_lengthscale;  // Jacobian for log transform

    // N(0, I) prior on beta
    log_post += ratiod_hsgp::hsgp_log_prior_beta(hsgp_beta);

    // Evaluate HSGP spatial effect: f = Phi * sqrt(S) * beta
    ratiod_hsgp::hsgp_evaluate(hsgp_beta, sigma2_hsgp, lengthscale_hsgp,
                                data.hsgp_data, hsgp_f);
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

  // TVC (Temporally-Varying Coefficients) priors
  std::vector<double> tvc_tau;
  std::vector<double> tvc_rho;
  std::vector<double> tvc_w_flat;
  std::vector<double> tvc_eta;

  if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;

    // Extract tau (precision) parameters
    tvc_tau.resize(n_tvc);
    for (int j = 0; j < n_tvc; j++) {
      double log_tau = params[layout.log_tau_tvc_start + j];
      tvc_tau[j] = std::exp(log_tau);

      // PC prior on tau (exponential prior on sigma = 1/sqrt(tau))
      // P(sigma > U) = alpha  =>  rate = -log(alpha) / U
      double sigma = 1.0 / std::sqrt(tvc_tau[j]);
      double rate = -std::log(0.01) / 1.0;  // P(sigma > 1) = 0.01
      log_post += std::log(rate) - rate * sigma - std::log(2.0 * sigma);
      log_post += log_tau;  // Jacobian for log transform
    }

    // Extract rho parameters for AR1 structure
    tvc_rho.resize(n_tvc, 0.0);
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1 &&
        layout.logit_rho_tvc_start >= 0) {
      for (int j = 0; j < n_tvc; j++) {
        double logit_rho = params[layout.logit_rho_tvc_start + j];
        tvc_rho[j] = 2.0 / (1.0 + std::exp(-logit_rho)) - 1.0;  // Map to (-1, 1)

        // Uniform(-1, 1) prior on rho
        // Jacobian for logit((rho+1)/2) transform
        double x = (tvc_rho[j] + 1.0) / 2.0;
        log_post += std::log(x) + std::log(1.0 - x);
      }
    }

    // Extract TVC values
    int n_tvc_params = n_groups * n_tvc * n_times;
    tvc_w_flat.resize(n_tvc_params);
    for (int k = 0; k < n_tvc_params; k++) {
      tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // TVC temporal prior
    log_post += ratiod_tvc::tvc_log_prior(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho);

    // Soft sum-to-zero constraint for identifiability
    log_post += ratiod_tvc::tvc_sum_to_zero_penalty(tvc_w_flat, data.tvc_data, 0.001);

    // Precompute TVC contribution to linear predictor
    tvc_eta.resize(data.N, 0.0);
    ratiod_tvc::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
  }

  // ============ SVC (Spatially-Varying Coefficients) ============
  std::vector<double> svc_sigma2;
  std::vector<double> svc_phi;
  std::vector<double> svc_w_flat;
  std::vector<double> svc_eta;

  if (layout.has_svc && data.svc_data.n_svc > 0) {
    int n_svc = data.svc_data.n_svc;
    int n_obs = data.svc_data.n_obs;

    // Extract sigma2 (spatial variance) parameters
    svc_sigma2.resize(n_svc);
    for (int j = 0; j < n_svc; j++) {
      double log_sigma2 = params[layout.log_sigma2_svc_start + j];
      svc_sigma2[j] = std::exp(log_sigma2);

      // Half-Cauchy prior on sigma = sqrt(sigma2)
      // p(sigma) ∝ 1 / (1 + (sigma/scale)^2)
      double sigma = std::sqrt(svc_sigma2[j]);
      double scale = data.svc_sigma2_prior_scale;
      log_post += -std::log(1.0 + (sigma * sigma) / (scale * scale));
      log_post += log_sigma2;  // Jacobian for log transform
    }

    // Extract phi (spatial range) parameters
    svc_phi.resize(n_svc);
    for (int j = 0; j < n_svc; j++) {
      double log_phi = params[layout.log_phi_svc_start + j];
      svc_phi[j] = std::exp(log_phi);

      // Uniform prior on phi (transformed to log scale)
      // p(phi) ∝ 1 for phi in [lower, upper]
      // Jacobian: log_phi => phi * d(log_phi) = phi
      if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
        return -INFINITY;
      }
      log_post += log_phi;  // Jacobian for log transform
    }

    // Extract SVC values
    int n_svc_params = n_svc * n_obs;
    svc_w_flat.resize(n_svc_params);
    for (int k = 0; k < n_svc_params; k++) {
      svc_w_flat[k] = params[layout.svc_w_start + k];
    }

    // NNGP prior on each SVC term
    for (int j = 0; j < n_svc; j++) {
      // Extract w_j (the j-th SVC term at all locations)
      std::vector<double> w_j(n_obs);
      for (int i = 0; i < n_obs; i++) {
        w_j[i] = svc_w_flat[j * n_obs + i];
      }

      // NNGP log-likelihood
      log_post += ratiod_svc::nngp_log_lik(w_j, svc_sigma2[j], svc_phi[j], data.svc_data);
    }

    // Soft sum-to-zero constraint for identifiability (mean(w) ~ N(0,1))
    log_post += ratiod_svc::svc_sum_to_zero_penalty(svc_w_flat, data.svc_data, 1.0);

    // Precompute SVC contribution to linear predictor
    svc_eta.resize(data.N, 0.0);
    ratiod_svc::compute_svc_eta(svc_w_flat, data.svc_data, svc_eta);
  }

  // ============ LIKELIHOOD (parallelized) ============
  // When skip_obs_loop is true, we skip this section entirely.
  // This is used by the fused gradient+log_post computation where
  // the observation log-likelihood is accumulated during the gradient pass.

  double log_lik = 0.0;

  if (!skip_obs_loop) {

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
          int group_idx = data.re_group_multi_flat[t * data.N + i];
          if (group_idx > 0) {
            int g = group_idx - 1;
            int n_coefs = layout.re_n_coefs_multi[t];
            int re_base = layout.re_start_multi[t] + g * n_coefs;
            bool is_corr_t = layout.re_correlated_multi[t] && n_coefs > 1;

            // For correlated slopes: use pre-computed non-centered re
            // For uncorrelated slopes: params store re directly (centered)
            double re_contrib = is_corr_t ? re_nc_flat[re_base] : params[re_base];

            int n_slopes = n_coefs - 1;
            if (n_slopes > 0 && !data.re_slope_matrices[t].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                double re_slope = is_corr_t ? re_nc_flat[re_base + 1 + s] : params[re_base + 1 + s];
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            eta_denom += re_contrib;
          }
        }
      } else if (n_terms > 1) {
        // Multiple RE terms (intercept only): params store re directly (centered)
        for (int t = 0; t < n_terms; t++) {
          int group_idx = data.re_group_multi_flat[t * data.N + i];
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
        // BYM2: u = sigma_s * scale * phi + sigma_u * theta
        double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
        spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s];
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

    // Add GP spatial effect (map observation to unique location)
    if (layout.is_gp && data.has_gp && gp_w != nullptr) {
      int loc_i = data.gp_data.obs_to_loc[i];
      double gp_effect = gp_w[loc_i];
      if (data.gp_data.shared) {
        eta_num += gp_effect;
        eta_denom += gp_effect;
      } else {
        eta_num += gp_effect;
      }
    }

    // Add multi-scale GP spatial effects (map observation to unique location)
    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
      int loc_i = data.multiscale_gp_data.obs_to_loc[i];
      double local_effect = gp_local[loc_i];
      double regional_effect = gp_regional[loc_i];
      double ms_spatial_effect = local_effect + regional_effect;

      if (data.multiscale_gp_data.shared) {
        eta_num += ms_spatial_effect;
        eta_denom += ms_spatial_effect;
      } else {
        eta_num += ms_spatial_effect;
      }
    }

    // Add HSGP spatial effect (observation-level)
    if (layout.is_hsgp && data.has_hsgp && !hsgp_f.empty()) {
      double hsgp_effect = hsgp_f[i];
      if (data.hsgp_data.shared) {
        eta_num += hsgp_effect;
        eta_denom += hsgp_effect;
      } else {
        eta_num += hsgp_effect;
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

    // Add TVC (Temporally-Varying Coefficients) effect
    if (layout.has_tvc && !tvc_eta.empty()) {
      double tvc_effect = tvc_eta[i];
      if (data.tvc_data.shared) {
        eta_num += tvc_effect;
        eta_denom += tvc_effect;
      } else {
        eta_num += tvc_effect;
      }
    }

    // Add SVC (Spatially-Varying Coefficients) effect
    if (layout.has_svc && !svc_eta.empty()) {
      double svc_effect = svc_eta[i];
      if (data.svc_data.shared) {
        eta_num += svc_effect;
        eta_denom += svc_effect;
      } else {
        eta_num += svc_effect;
      }
    }

    // Compute ZI linear predictor if applicable (using optimized dot product)
    double logit_zi = 0.0;
    if (layout.has_zi) {
      logit_zi = ratiod_linalg::dot_product(
          &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
    }

    // Compute OI linear predictor if applicable (for OI_BINOMIAL and ZOIB)
    double logit_oi = 0.0;
    if (layout.has_oi && data.p_oi > 0) {
      logit_oi = ratiod_linalg::dot_product(
          &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
    }

    // Likelihood contribution
    double ll_i = 0.0;
    if (data.model_type == ModelType::BINOMIAL) {
      // Handle all binomial ZI/OI variants
      double p = 1.0 / (1.0 + std::exp(-eta_num));
      int n_trials = data.y_denom[i];
      int y = data.y_num[i];

      if (data.zi_type == ratiod_zi::ZIType::ZI_BINOMIAL) {
        ll_i = ratiod_zi::zi_binomial_lpmf_logit(y, n_trials, p, logit_zi);
      } else if (data.zi_type == ratiod_zi::ZIType::HURDLE_BINOMIAL) {
        ll_i = ratiod_zi::hurdle_binomial_lpmf_logit(y, n_trials, p, logit_zi);
      } else if (data.zi_type == ratiod_zi::ZIType::OI_BINOMIAL) {
        ll_i = ratiod_zi::oi_binomial_lpmf_logit(y, n_trials, p, logit_oi);
      } else if (data.zi_type == ratiod_zi::ZIType::ZOIB) {
        ll_i = ratiod_zi::zoib_lpmf_logit(y, n_trials, p, logit_zi, logit_oi);
      } else {
        // Plain binomial (no inflation)
        ll_i = log_lik_binomial(data.y_num[i], data.y_denom[i], eta_num);
      }
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

    } else if (data.model_type == ModelType::POISSON_GAMMA) {
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

    } else if (data.model_type == ModelType::GAMMA_GAMMA) {
      // Gamma-Gamma: both numerator and denominator are Gamma distributed
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);
      // phi_num = shape_num, phi_denom = shape_denom
      ll_i = log_lik_gamma(data.y_num_cont[i], phi_num, mu_num);
      ll_i += log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);

    } else if (data.model_type == ModelType::LOGNORMAL) {
      // Lognormal-Lognormal: both responses are Lognormal
      // eta = mean on log scale, phi = sigma (std dev on log scale)
      double log_y_num = std::log(data.y_num_cont[i]);
      double log_y_denom = std::log(data.y_denom_cont[i]);
      // Log-lik: -log(y) - log(sigma) - 0.5*((log(y)-mu)/sigma)^2
      double z_num = (log_y_num - eta_num) / phi_num;
      double z_denom = (log_y_denom - eta_denom) / phi_denom;
      ll_i = -log_y_num - std::log(phi_num) - 0.5 * z_num * z_num;
      ll_i += -log_y_denom - std::log(phi_denom) - 0.5 * z_denom * z_denom;

    } else if (data.model_type == ModelType::BETA_BINOMIAL) {
      // Beta-binomial: overdispersed binomial
      double p = 1.0 / (1.0 + std::exp(-eta_num));
      int y = data.y_num[i];
      int n = data.y_denom[i];
      // phi_num = precision parameter (alpha + beta)
      double alpha = p * phi_num;
      double beta_param = (1.0 - p) * phi_num;
      // Beta-binomial log-likelihood
      ll_i = std::lgamma(y + alpha) + std::lgamma(n - y + beta_param) - std::lgamma(n + phi_num);
      ll_i += -std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num);
      ll_i += ratiod::math::portable_lchoose(n, y);
    }

    log_lik += ll_i;
  }

  } // end if (!skip_obs_loop)

  log_post += log_lik;
  return log_post;
}

// =====================================================================
// Analytical gradient for simple Poisson-Gamma models
// O(n) instead of O(n*p) - huge speedup for typical models
// =====================================================================

bool can_use_analytical_gradient(const ModelData& data, const ParamLayout& layout) {
  // Hand-coded gradients for basic models without complex structure
  bool is_basic_family = (data.model_type == ModelType::POISSON_GAMMA ||
                          data.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.model_type == ModelType::BINOMIAL ||
                          data.model_type == ModelType::GAMMA_GAMMA ||
                          data.model_type == ModelType::BETA_BINOMIAL ||
                          data.model_type == ModelType::LOGNORMAL);

  // Check if spatial type is one we have hand-coded gradients for
  bool spatial_is_icar_bym2 = (data.spatial_type == SpatialType::ICAR ||
                               data.spatial_type == SpatialType::BYM2);

  // Temporal is OK alone or combined with ICAR/BYM2 spatial (no spatiotemporal interaction)
  // Note: Temporal GP is excluded - use autodiff for that
  bool temporal_ok = !layout.has_temporal ||
                     (layout.has_temporal && !layout.is_temporal_gp &&
                      !layout.has_spatiotemporal &&
                      (!layout.has_spatial || spatial_is_icar_bym2));

  // Spatial is OK for ICAR/BYM2 (alone or combined with temporal)
  bool spatial_ok = !layout.has_spatial ||
                    (layout.has_spatial && spatial_is_icar_bym2 && !layout.has_spatiotemporal);

  // ZI is OK for basic models, including with ICAR/BYM2 spatial (but not temporal)
  // Binomial ZI/Hurdle is also supported
  bool zi_ok = !layout.has_zi ||
               (layout.has_zi && !layout.has_temporal &&
                (!layout.has_spatial || spatial_is_icar_bym2));

  // Random slopes: both correlated (|) and uncorrelated (||) are supported
  // Can combine with ICAR/BYM2 spatial (but not temporal/ZI)
  bool slopes_ok = !layout.has_re_slopes ||
                   (layout.has_re_slopes &&
                    !layout.has_temporal && !layout.has_zi &&
                    (!layout.has_spatial || spatial_is_icar_bym2));

  return (is_basic_family &&
          !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
          temporal_ok && spatial_ok && zi_ok && slopes_ok &&
          !layout.has_latent && !layout.has_spatiotemporal &&
          !layout.has_multiscale_temporal && !layout.has_tvc &&
          !layout.has_svc &&  // SVC has its own gradient function
          (data.n_re_terms <= 1 ||
           (data.n_re_terms > 1 && !layout.has_re_slopes)));  // Crossed RE (intercept-only) is OK
}

void compute_gradient_analytical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Fused log-posterior computation: accumulate observation log-likelihood
  // alongside gradients to avoid a separate O(N) pass.
  const bool compute_lp = (log_post_out != nullptr);
  double obs_log_lik = 0.0;

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
  double phi_denom = 1.0, log_phi_denom = 0.0;
  if (layout.has_phi_num) {
    log_phi_num = params[layout.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.has_phi_denom) {
    log_phi_denom = params[layout.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
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

  // phi priors: Gamma(shape, rate) via log transform
  // d/d(log_phi) = (shape-1) - rate*phi + 1 (Jacobian)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] = (data.phi_prior_shape - 1.0)
                                   - data.phi_prior_rate * phi_num + 1.0;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] = (data.phi_prior_shape - 1.0)
                                     - data.phi_prior_rate * phi_denom + 1.0;
  }

  // RE prior: N(0, sigma_re^2)
  // d/d(re[g]) = -tau_re * re[g]
  // Also accumulates contribution to sigma_re gradient
  double re_prior_grad_sigma = 0.0;
  std::vector<std::vector<double>> grad_re_slopes_lik;  // [term][g*n_coefs+c] likelihood contributions
  int n_re_terms_slopes = 0;

  // Non-centered slopes: pre-computed RE values for observation loop
  std::vector<double> re_nc_flat;
  // Per-term storage for non-centered chain rule in write-back
  std::vector<std::vector<double>> nc_L_flats;   // [term] -> L_flat
  std::vector<std::vector<double>> nc_sigmas_vec; // [term] -> sigmas

  if (layout.has_re && layout.has_re_slopes && !layout.has_re_correlated_slopes) {
    // ============ Uncorrelated random slopes prior gradients ============
    // Each coefficient type c has independent N(0, sigma_c^2) prior
    n_re_terms_slopes = data.n_re_terms;
    grad_re_slopes_lik.resize(n_re_terms_slopes);

    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters and compute priors
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        double log_sigma_c = params[log_sigma_idx];
        double sigma_c = std::exp(log_sigma_c);
        double tau_c = 1.0 / (sigma_c * sigma_c + 1e-10);

        // Half-Cauchy prior on sigma_c
        double ratio_c = sigma_c / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;

        // N(0, sigma_c^2) prior on each re_{g,c}
        double sigma_grad_c = 0.0;
        for (int g = 0; g < n_groups; g++) {
          double re_gc = params[re_start_t + g * n_coefs + c];
          grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
          // Contribution to sigma gradient
          sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
        }
        grad[log_sigma_idx] += sigma_grad_c;
      }
    }
  } else if (layout.has_re && layout.has_re_slopes && layout.has_re_correlated_slopes) {
    // ============ Correlated random slopes prior gradients ============
    // Multivariate normal with Sigma = diag(sigma) * L * L' * diag(sigma)
    // where L is lower-triangular Cholesky factor with L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
    // LKJ(eta=2) prior on correlation matrix
    n_re_terms_slopes = data.n_re_terms;
    grad_re_slopes_lik.resize(n_re_terms_slopes);

    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_correlated = layout.re_correlated_multi[t];
      grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        sigmas[c] = std::exp(params[log_sigma_idx]);

        // Half-Cauchy prior on sigma_c: d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
        double ratio_c = sigmas[c] / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;
      }

      if (is_correlated && n_coefs > 1) {
        // Build Cholesky factor L with tanh parameterization
        // Must match compute_log_post: L[i,j] = tanh(raw[idx])
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> L_flat(n_coefs * n_coefs, 0.0);

        int chol_idx = 0;
        for (int i = 0; i < n_coefs; i++) {
          double row_sum_sq = 0.0;
          for (int j = 0; j < i; j++) {
            double raw_ij = params[chol_start + chol_idx];
            L_flat[i * n_coefs + j] = std::tanh(raw_ij);
            row_sum_sq += L_flat[i * n_coefs + j] * L_flat[i * n_coefs + j];
            chol_idx++;
          }
          double diag_sq = 1.0 - row_sum_sq;
          if (diag_sq < 1e-10) {
            // Safety guard (shouldn't trigger with tanh)
            return;
          }
          L_flat[i * n_coefs + i] = std::sqrt(diag_sq);
        }

        // LKJ(eta=2) prior gradient w.r.t. Cholesky elements
        // LKJ contribution: sum_k (eta - 1 + (n-k-1)/2) * 2 * log(L[k,k])
        // Jacobian contribution: sum_{k>0} (n - k) * log(L[k,k])
        double eta = 2.0;
        int n_chol = n_coefs * (n_coefs - 1) / 2;
        std::vector<double> grad_chol(n_chol, 0.0);

        // Gradient of LKJ + Jacobian w.r.t. L[i,j] (off-diagonal)
        // d(log L[k,k])/d(L[i,j]) = -L[i,j] / L[i,i]^2 if i = k
        for (int k = 1; k < n_coefs; k++) {
          double L_kk = L_flat[k * n_coefs + k];
          double coef_lkj = (eta - 1.0 + (n_coefs - k - 1) / 2.0) * 2.0;
          double coef_jac = (n_coefs - k);
          double coef_total = coef_lkj + coef_jac;

          // d/d(L[k,j]) for j < k
          int chol_base = k * (k - 1) / 2;
          for (int j = 0; j < k; j++) {
            double L_kj = L_flat[k * n_coefs + j];
            // d(log L[k,k])/d(L[k,j]) = -L[k,j] / (L[k,k]^2)
            grad_chol[chol_base + j] += coef_total * (-L_kj / (L_kk * L_kk));
          }
        }

        // ---- Non-centered parameterization ----
        // Params store z ~ N(0,1). Compute re = diag(sigma) * L * z for observation loop.

        // Allocate re_nc_flat if needed
        if (re_nc_flat.empty()) {
          re_nc_flat.assign(params.size(), 0.0);
        }

        // Pre-compute re from z for all groups
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double Lz_c = 0.0;
            for (int k = 0; k <= c; k++) {
              Lz_c += L_flat[c * n_coefs + k] * params[re_start_t + g * n_coefs + k];
            }
            re_nc_flat[re_start_t + g * n_coefs + c] = sigmas[c] * Lz_c;
          }
        }

        // Save term data for write-back chain rule
        nc_L_flats.resize(n_re_terms_slopes);
        nc_sigmas_vec.resize(n_re_terms_slopes);
        nc_L_flats[t] = L_flat;
        nc_sigmas_vec[t] = sigmas;

        // Prior on z: N(0, I) -> grad[z_idx] = -z[g,c]
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
          }
        }

        // Sigma: Half-Cauchy prior already written above. No centered contribution.
        // In non-centered, the log-det of Jacobian (re = diag(sigma)*L*z)
        // cancels with the |Sigma|^{-1/2} normalization, so no -n_groups term.

        // Cholesky: write LKJ prior gradient only (with tanh chain rule)
        // No centered prior contribution in non-centered parameterization
        {
          int cidx = 0;
          for (int i = 1; i < n_coefs; i++) {
            for (int j = 0; j < i; j++) {
              double raw_val = params[chol_start + cidx];
              double l_val = std::tanh(raw_val);
              double sech2 = 1.0 - l_val * l_val;
              grad[chol_start + cidx] = grad_chol[cidx] * sech2 - 2.0 * l_val;
              cidx++;
            }
          }
        }
      } else {
        // Uncorrelated term within a mixed model (fallback)
        for (int c = 0; c < n_coefs; c++) {
          double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
          double sigma_grad_c = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double re_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
          }
          int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
          grad[log_sigma_idx] += sigma_grad_c;
        }
      }
    }
  } else if (layout.has_re && !layout.has_re_slopes) {
    // Intercept-only RE (single or crossed terms)
    int n_terms = (data.n_re_terms > 1) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      // Get term-specific parameters
      int log_sigma_idx = (n_terms > 1) ? layout.log_sigma_re_multi[t] : layout.log_sigma_re_idx;
      int re_start_t = (n_terms > 1) ? layout.re_start_multi[t] : layout.re_start;
      int n_groups_t = (n_terms > 1) ? data.re_n_groups_multi[t] : data.n_re_groups;

      double log_sigma_t = params[log_sigma_idx];
      double sigma_t = std::exp(log_sigma_t);
      double tau_t = 1.0 / (sigma_t * sigma_t + 1e-10);

      // Half-Cauchy prior on sigma_t
      double ratio_t = sigma_t / data.sigma_re_scale;
      double ratio_t_sq = ratio_t * ratio_t;
      grad[log_sigma_idx] = -2.0 * ratio_t_sq / (1.0 + ratio_t_sq) + 1.0;

      // N(0, sigma_t^2) prior on each RE for this term
      double sigma_grad_t = 0.0;
      for (int g = 0; g < n_groups_t; g++) {
        double re_g = params[re_start_t + g];
        grad[re_start_t + g] = -tau_t * re_g;
        sigma_grad_t += tau_t * re_g * re_g - 1.0;
      }
      grad[log_sigma_idx] += sigma_grad_t;
    }
  }

  // ============ Temporal prior gradients ============
  double log_tau_temporal = 0.0, tau_temporal = 1.0;
  double logit_rho_ar1 = 0.0, rho_ar1 = 0.5;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  std::vector<double> grad_temporal_lik;  // Likelihood contribution

  if (layout.has_temporal) {
    log_tau_temporal = params[layout.log_tau_temporal_idx];
    tau_temporal = std::exp(log_tau_temporal);
    T_len = layout.temporal_end - layout.temporal_start;
    phi_temporal = &params[layout.temporal_start];
    grad_temporal_lik.assign(T_len, 0.0);

    // tau prior: Gamma(shape, rate) via log transform
    // d/d(log_tau) = (shape-1) - rate*tau + 1 (Jacobian)
    grad[layout.log_tau_temporal_idx] = (data.tau_temporal_shape - 1.0)
                                        - data.tau_temporal_rate * tau_temporal + 1.0;

    // AR1: extract rho and add prior
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      logit_rho_ar1 = params[layout.logit_rho_ar1_idx];
      rho_ar1 = 1.0 / (1.0 + std::exp(-logit_rho_ar1));
      // Uniform(0,1) prior on rho with logit Jacobian: grad = 1 - 2*rho
      grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }
  }

  // ============ Spatial prior gradients (ICAR and BYM2) ============
  double log_tau_spatial = 0.0, tau_spatial = 1.0;
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
  int n_spatial = 0;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  std::vector<double> grad_spatial_lik;  // Likelihood contribution

  if (layout.has_spatial) {
    n_spatial = data.n_spatial_units;
    phi_spatial = &params[layout.spatial_start];
    grad_spatial_lik.assign(n_spatial, 0.0);

    if (data.spatial_type == SpatialType::BYM2) {
      // BYM2 Riebler: derive sigma_s, sigma_u from sigma_total, rho
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      theta_bym2 = &params[layout.theta_bym2_start];

      // Half-Cauchy prior on sigma_total
      double ratio = sigma_total / data.sigma_re_scale;
      double ratio_sq = ratio * ratio;
      grad[layout.log_sigma_bym2_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // d/d(logit_rho) [log(rho) + log(1-rho)] = (1-rho) - rho = 1 - 2*rho
      grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;

      // Initialize theta gradients (N(0,1) prior: d/d(theta) = -theta)
      for (int s = 0; s < n_spatial; s++) {
        grad[layout.theta_bym2_start + s] = -theta_bym2[s];
      }
    } else {
      // ICAR: extract tau
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);

      // Gamma prior on tau via log transform
      grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0)
                                         - data.tau_spatial_rate * tau_spatial + 1.0;
    }
  }

  // ============ Zero-inflation prior gradients ============
  const double* beta_zi = nullptr;
  std::vector<double> grad_beta_zi;
  double tau_zi = 1.0;

  if (layout.has_zi && data.p_zi > 0) {
    beta_zi = &params[layout.beta_zi_start];
    tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    grad_beta_zi.assign(data.p_zi, 0.0);

    // N(0, zi_prior_sd^2) prior on ZI coefficients
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] = -tau_zi * beta_zi[j];
    }
  }

  // ============ One-inflation (OI) prior gradients ============
  const double* beta_oi = nullptr;
  std::vector<double> grad_beta_oi;
  double tau_oi = 1.0;

  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    grad_beta_oi.assign(data.p_oi, 0.0);

    // N(0, oi_prior_sd^2) prior on OI coefficients
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] = -tau_oi * beta_oi[j];
    }
  }

  // ============ Likelihood gradients (O(n)) ============

  // Try fused single-pass gradient first (best for small p <= 4).
  // Then fall back to 3-pass vectorized (better for larger p with Eigen).
  // Finally fall back to scalar loop for complex models (ZI, slopes, etc.).
  bool used_vectorized = vectorized::dispatch_fused_gradient(
      params, data, layout, grad, obs_log_lik,
      compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws);

  // Fall back to 3-pass vectorized for p > 4 (Eigen matvec is beneficial)
  if (!used_vectorized) {
    used_vectorized = vectorized::dispatch_vectorized_gradient(
        params, data, layout, grad, obs_log_lik,
        compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws);
  }

  if (!used_vectorized) {

  // Accumulators for beta gradients (will be added via X' * residual)
  std::vector<double> grad_beta_num(data.p_num, 0.0);
  std::vector<double> grad_beta_denom(data.p_denom, 0.0);
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel
  {
    std::vector<double> local_grad_beta_num(data.p_num, 0.0);
    std::vector<double> local_grad_beta_denom(data.p_denom, 0.0);
    double local_grad_phi_num = 0.0;
    double local_grad_phi_denom = 0.0;
    std::vector<double> local_grad_re(layout.has_re ? data.n_re_groups : 0, 0.0);
    // Thread-local buffer for crossed RE gradients (all terms combined)
    std::vector<double> local_grad_re_crossed(
        (layout.has_re && data.n_re_terms > 1) ? data.total_re_groups : 0, 0.0);
    double local_obs_ll = 0.0;  // Fused log-likelihood accumulator
    // Pre-allocated per-obs group index buffer for crossed RE (reused across iterations)
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);

    #pragma omp for schedule(static)
    for (int i = 0; i < data.N; i++) {
  #else
    // Pre-allocated per-obs group index buffer for crossed RE
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);
    for (int i = 0; i < data.N; i++) {
  #endif
      // Compute linear predictors
      double eta_num = ratiod_linalg::dot_product(
          &data.X_num_flat[i * data.p_num], beta_num, data.p_num);
      double eta_denom = (data.p_denom > 0) ?
          ratiod_linalg::dot_product(
              &data.X_denom_flat[i * data.p_denom], beta_denom, data.p_denom) :
          0.0;

      // Add RE if present
      int re_idx = -1;
      int re_term_idx = -1;
      int re_group_idx = -1;
      int re_n_coefs_i = 1;
      std::vector<double> re_slope_x_i;  // Slope design values for this obs
      int n_crossed_terms = 0;
      if (layout.has_re) {
        if (layout.has_re_slopes && n_re_terms_slopes > 0) {
          // Random slopes case: handle first term only (single RE term for H gradients)
          re_term_idx = 0;
          re_group_idx = data.re_group_multi_flat[0 * data.N + i];
          if (re_group_idx > 0) {
            int g = re_group_idx - 1;
            re_n_coefs_i = layout.re_n_coefs_multi[0];
            int re_base = layout.re_start_multi[0] + g * re_n_coefs_i;

            // For correlated slopes: use pre-computed non-centered re
            // For uncorrelated slopes: params store re directly (centered)
            bool is_corr_t = !re_nc_flat.empty() &&
                             layout.re_correlated_multi.size() > 0 &&
                             layout.re_correlated_multi[0] && re_n_coefs_i > 1;

            // Intercept contribution
            double re_contrib = is_corr_t ? re_nc_flat[re_base] : params[re_base];

            // Slope contributions
            int n_slopes = re_n_coefs_i - 1;
            re_slope_x_i.resize(n_slopes);
            if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
                re_slope_x_i[s] = x_slope;
                double re_slope = is_corr_t ? re_nc_flat[re_base + 1 + s] : params[re_base + 1 + s];
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            if (data.model_type != ModelType::BINOMIAL) {
              eta_denom += re_contrib;
            }
          }
        } else if (data.n_re_terms > 1) {
          // Crossed RE (multiple intercept-only terms)
          n_crossed_terms = data.n_re_terms;
          for (int t = 0; t < n_crossed_terms; t++) {
            int group_idx = data.re_group_multi_flat[t * data.N + i];
            if (group_idx > 0) {
              int g = group_idx - 1;
              re_idx_multi_buf[t] = g;
              double re_val = params[layout.re_start_multi[t] + g];
              eta_num += re_val;
              if (data.model_type != ModelType::BINOMIAL) {
                eta_denom += re_val;
              }
            } else {
              re_idx_multi_buf[t] = -1;
            }
          }
        } else if (data.re_group[i] > 0) {
          // Simple intercept-only RE (single term)
          re_idx = data.re_group[i] - 1;
          eta_num += re[re_idx];
          if (data.model_type != ModelType::BINOMIAL) {
            eta_denom += re[re_idx];
          }
        }
      }
      // Add temporal effect if present
      int t_idx = -1;
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        t_idx = data.temporal_time_idx[i] - 1;
        double temporal_effect = phi_temporal[t_idx];
        eta_num += temporal_effect;
        if (data.model_type != ModelType::BINOMIAL) {
          eta_denom += temporal_effect;
        }
      }

      // Add spatial effect if present
      int s_idx = -1;
      double d_spatial_d_phi = 0.0;  // Derivative of spatial_effect wrt phi_spatial
      double d_spatial_d_theta = 0.0;  // Derivative of spatial_effect wrt theta_bym2
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        s_idx = data.spatial_group[i] - 1;
        double spatial_effect;
        if (data.spatial_type == SpatialType::BYM2) {
          // BYM2: spatial_effect = sigma_s * scale * phi + sigma_u * theta
          double scaled_phi = phi_spatial[s_idx] * data.bym2_scale_factor;
          spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s_idx];
          d_spatial_d_phi = sigma_s_bym2 * data.bym2_scale_factor;
          d_spatial_d_theta = sigma_u_bym2;
        } else {
          // ICAR: spatial_effect = phi_spatial
          spatial_effect = phi_spatial[s_idx];
          d_spatial_d_phi = 1.0;
        }
        eta_num += spatial_effect;
        if (data.model_type != ModelType::BINOMIAL) {
          eta_denom += spatial_effect;
        }
      }

      double resid_num = 0.0;
      double resid_denom = 0.0;
      double grad_phi_num_i = 0.0;
      double grad_phi_denom_i = 0.0;
      double grad_logit_zi_i = 0.0;  // Gradient w.r.t. logit_zi for this obs
      double grad_logit_oi_i = 0.0;  // Gradient w.r.t. logit_oi for this obs

      // Compute ZI linear predictor if applicable
      double logit_zi = 0.0;
      double zi_prob = 0.0;
      if (layout.has_zi && data.p_zi > 0) {
        logit_zi = ratiod_linalg::dot_product(
            &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
        zi_prob = 1.0 / (1.0 + std::exp(-logit_zi));
      }

      // Compute OI linear predictor if applicable
      double logit_oi = 0.0;
      double oi_prob = 0.0;
      if (layout.has_oi && data.p_oi > 0) {
        logit_oi = ratiod_linalg::dot_product(
            &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
        oi_prob = 1.0 / (1.0 + std::exp(-logit_oi));
      }

      if (data.model_type == ModelType::BINOMIAL) {
        // ---- BINOMIAL ----
        // p = inv_logit(eta_num), LL = y*log(p) + (n-y)*log(1-p)
        // d(LL)/d(eta) = y - n*p
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int n_trials = data.y_denom[i];
        int y_num_i = data.y_num[i];

        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_BINOMIAL) {
          // ZI-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*(1-p)^n
            double p0_binom = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double p0 = zi_prob + (1.0 - zi_prob) * p0_binom;
            // d(LL)/d(eta) = (1-zi) * d((1-p)^n)/d(eta) / p0
            // d((1-p)^n)/d(eta) = n * (1-p)^(n-1) * (-p*(1-p)) = -n*p*(1-p)^(n-1)
            resid_num = -(1.0 - zi_prob) * n_trials * p * std::pow(1.0 - p, n_trials - 1) / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_binom) / p0;
          } else {
            // P(Y=y) = (1-zi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
          }
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_BINOMIAL) {
          // Hurdle-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta
            resid_num = 0.0;  // No p contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncBinomial(y|n,p)
            double p0_binom = std::pow(1.0 - p, n_trials);
            double normalizer = 1.0 - p0_binom;
            if (normalizer < 1e-12) normalizer = 1e-12;
            // Gradient from truncated binomial
            // d(log_normalizer)/d(eta) = n*p*(1-p)^(n-1) / (1-(1-p)^n)
            double grad_normalizer = n_trials * p * std::pow(1.0 - p, n_trials - 1) / normalizer;
            resid_num = (y_num_i - n_trials * p) - grad_normalizer;
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
          }
        } else if (layout.has_oi && data.zi_type == ratiod_zi::ZIType::OI_BINOMIAL) {
          // OI-Binomial (One-inflation only)
          // P(Y=n) = oi + (1-oi) * p^n
          // P(Y=y, y<n) = (1-oi) * Binomial(y|n,p)
          if (y_num_i == n_trials) {
            // y = n (structural one or binomial one)
            double pn = std::pow(p, n_trials);  // p^n
            double P_yn = oi_prob + (1.0 - oi_prob) * pn;
            if (P_yn < 1e-12) P_yn = 1e-12;
            // d(log P)/d(eta) = (1-oi) * n * p^(n-1) * p*(1-p) / P
            //                 = (1-oi) * n * p^n * (1-p) / P
            resid_num = (1.0 - oi_prob) * n_trials * pn * (1.0 - p) / P_yn;
            // d(log P)/d(logit_oi) = oi*(1-oi)*(1 - p^n) / P
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / P_yn;
          } else {
            // y < n: P(Y=y) = (1-oi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else if (layout.has_oi && data.zi_type == ratiod_zi::ZIType::ZOIB) {
          // ZOIB (Zero-One Inflated Binomial) - MIXTURE MODEL
          // P(Y=0) = zi + (1-zi)*(1-oi)*(1-p)^n
          // P(Y=n) = (1-zi)*(oi + (1-oi)*p^n)
          // P(Y=y, 0<y<n) = (1-zi)*(1-oi)*Binomial(y|n,p)
          // Note: zi_prob = zi, oi_prob = oi
          if (y_num_i == 0) {
            // y = 0: P = zi + (1-zi)*(1-oi)*(1-p)^n = A + B
            double binom_zero = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double A = zi_prob;  // structural zero component
            double B = (1.0 - zi_prob) * (1.0 - oi_prob) * binom_zero;  // binomial zero
            double P = A + B;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d((1-p)^n)/d(eta) / P
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*(1-p)^n * p
            double d_binom_d_eta = -n_trials * binom_zero * p;
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_binom_d_eta / P;

            // d(log P)/d(logit_zi) = [dA - B] * zi*(1-zi) / P
            // dA/d(logit_zi) = zi*(1-zi), dB/d(logit_zi) = -(1-oi)*binom_zero * zi*(1-zi)
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - (1.0 - oi_prob) * binom_zero) / P;

            // d(log P)/d(logit_oi) = dB/d(logit_oi) / P
            // dB/d(logit_oi) = -(1-zi)*binom_zero * oi*(1-oi)
            grad_logit_oi_i = -(1.0 - zi_prob) * binom_zero * oi_prob * (1.0 - oi_prob) / P;

          } else if (y_num_i == n_trials) {
            // y = n: P = (1-zi)*(oi + (1-oi)*p^n) = (1-zi)*C
            double pn = std::pow(p, n_trials);  // p^n
            double C = oi_prob + (1.0 - oi_prob) * pn;  // oi + (1-oi)*p^n
            double P = (1.0 - zi_prob) * C;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d(p^n)/d(eta) / P
            // d(p^n)/d(eta) = n*p^(n-1) * p*(1-p) = n*p^n*(1-p)
            double d_pn_d_eta = n_trials * pn * (1.0 - p);
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_pn_d_eta / P;

            // d(log P)/d(logit_zi) = d(log(1-zi))/d(logit_zi) = -zi
            grad_logit_zi_i = -zi_prob;

            // d(log P)/d(logit_oi) = dC/d(logit_oi) / C
            // dC/d(logit_oi) = oi*(1-oi) - p^n * oi*(1-oi) = oi*(1-oi)*(1 - p^n)
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / C;

          } else {
            // 0 < y < n: P = (1-zi)*(1-oi)*Binomial
            // log P = log(1-zi) + log(1-oi) + log_binom
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else {
          // Standard binomial (no ZI)
          resid_num = y_num_i - n_trials * p;
        }
        // No denominator contribution for binomial

      } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
        // ---- NEGBIN_NEGBIN ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.y_num[i];
        int y_denom_i = data.y_denom[i];

        // Denominator NegBin gradient (always standard, not ZI)
        double denom_d = mu_denom + phi_denom;
        resid_denom = y_denom_i - mu_denom * (y_denom_i + phi_denom) / denom_d;
        grad_phi_denom_i = ratiod::math::portable_digamma(y_denom_i + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                           + std::log(phi_denom / denom_d)
                           + (mu_denom - y_denom_i) / denom_d;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*p0_nb
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(LL)/d(mu) = (1-zi) * d(p0_nb)/d(mu) / p0
            // d(p0_nb)/d(mu) = phi * (phi/(phi+mu))^phi * (-1/(phi+mu)) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            // phi gradient for ZI-NegBin at y=0 (complex, using approximation)
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            // P(Y=y) = (1-zi) * NB(y|mu,phi)
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta, where theta = sigmoid(logit_zi) here represents P(Y>0)
            // Note: for hurdle, logit_zi parameterizes theta = P(Y>0), so zi_prob IS theta
            resid_num = 0.0;  // No mu contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
            grad_phi_num_i = 0.0;
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncNB(y|mu,phi)
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double log_normalizer = std::log(1.0 - p0_nb);
            double denom_num = mu_num + phi_num;
            // Gradient from truncated NB: same as regular + correction for normalizer
            // d(log_normalizer)/d(mu) = -d(p0_nb)/d(mu) / (1-p0_nb) = phi*p0_nb / ((phi+mu)*(1-p0_nb))
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        + phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
            grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
        } else {
          // Standard NegBin (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
        }

      } else if (data.model_type == ModelType::POISSON_GAMMA) {
        // ---- POISSON_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.y_num[i];

        // Denominator: Gamma (always standard)
        double y_denom_i = data.y_denom_cont[i];
        resid_denom = phi_num * (y_denom_i / mu_denom - 1.0);
        double rate = phi_num / mu_denom;
        double grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                - ratiod::math::portable_digamma(phi_num) - rate * y_denom_i / phi_num;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_POISSON) {
          // ZI-Poisson numerator
          double exp_neg_mu = std::exp(-mu_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*exp(-mu)
            double p0 = zi_prob + (1.0 - zi_prob) * exp_neg_mu;
            // d(LL)/d(eta) = d(LL)/d(mu) * mu = -(1-zi)*exp(-mu)*mu / p0
            resid_num = -(1.0 - zi_prob) * exp_neg_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - exp_neg_mu) / p0;
            grad_phi_num_i = grad_phi_gamma;  // Only gamma part
          } else {
            // P(Y=y) = (1-zi) * Poisson(y|mu)
            resid_num = y_num_i - mu_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_num_i = grad_phi_gamma;
          }
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_POISSON) {
          // Hurdle-Poisson numerator
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // zi_prob is theta here
            grad_phi_num_i = grad_phi_gamma;
          } else {
            // Truncated Poisson: d(LL)/d(eta) = y - mu + mu*exp(-mu)/(1-exp(-mu))
            double exp_neg_mu = std::exp(-mu_num);
            resid_num = y_num_i - mu_num + mu_num * exp_neg_mu / (1.0 - exp_neg_mu);
            grad_logit_zi_i = 1.0 - zi_prob;
            grad_phi_num_i = grad_phi_gamma;
          }
        } else {
          // Standard Poisson (no ZI)
          resid_num = y_num_i - mu_num;
          grad_phi_num_i = grad_phi_gamma;
        }

      } else if (data.model_type == ModelType::GAMMA_GAMMA) {
        // ---- GAMMA_GAMMA ----
        // Both numerator and denominator are Gamma distributed
        // Gamma GLM: y ~ Gamma(shape, rate=shape/mu), so E[y] = mu
        // LL = shape*log(rate) - lgamma(shape) + (shape-1)*log(y) - rate*y
        //    = shape*log(shape/mu) - lgamma(shape) + (shape-1)*log(y) - (shape/mu)*y
        // d(LL)/d(eta) = d(LL)/d(mu) * d(mu)/d(eta) where mu = exp(eta), so d(mu)/d(eta) = mu
        // d(LL)/d(mu) = -shape/mu + shape*y/mu^2 = (shape/mu)(y/mu - 1)
        // d(LL)/d(eta) = shape*(y/mu - 1)
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double y_num_i = data.y_num_cont[i];
        double y_denom_i = data.y_denom_cont[i];

        // Residuals (gradient w.r.t. eta = log(mu))
        resid_num = phi_num * (y_num_i / mu_num - 1.0);
        resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);

        // Phi (shape) gradients
        // d(LL)/d(shape) = log(shape/mu) + 1 + log(y) - digamma(shape) - y/mu
        //                = log(shape) - log(mu) + 1 + log(y) - digamma(shape) - y/mu
        double rate_num = phi_num / mu_num;
        double rate_denom = phi_denom / mu_denom;
        grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                         - ratiod::math::portable_digamma(phi_num) - y_num_i / mu_num;
        grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                           - ratiod::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;

      } else if (data.model_type == ModelType::LOGNORMAL) {
        // ---- LOGNORMAL ----
        // Both numerator and denominator are Lognormal distributed
        // log(y) ~ Normal(mu, sigma^2), so y ~ Lognormal(mu, sigma^2)
        // LL = -log(y) - log(sigma) - 0.5*((log(y) - mu)/sigma)^2
        // d(LL)/d(eta) = d(LL)/d(mu) = (log(y) - mu) / sigma^2
        // (Note: eta IS mu for lognormal, so no chain rule needed)
        double mu_num = eta_num;  // mu is directly the linear predictor
        double mu_denom = eta_denom;
        double y_num_i = data.y_num_cont[i];
        double y_denom_i = data.y_denom_cont[i];
        double log_y_num = std::log(y_num_i);
        double log_y_denom = std::log(y_denom_i);

        // phi_num, phi_denom are sigma (std dev on log scale)
        double sigma_num = phi_num;
        double sigma_denom = phi_denom;
        double sigma_num_sq = sigma_num * sigma_num;
        double sigma_denom_sq = sigma_denom * sigma_denom;

        // Residuals (gradient w.r.t. mu)
        resid_num = (log_y_num - mu_num) / sigma_num_sq;
        resid_denom = (log_y_denom - mu_denom) / sigma_denom_sq;

        // Sigma gradients: d(LL)/d(sigma), NOT d(LL)/d(log_sigma)
        // Because the accumulation code multiplies by phi to get d(LL)/d(log_phi)
        // d(LL)/d(sigma) = -1/sigma + z^2/sigma = (-1 + z^2) / sigma
        double z_num = (log_y_num - mu_num) / sigma_num;
        double z_denom = (log_y_denom - mu_denom) / sigma_denom;
        grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
        grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

      } else if (data.model_type == ModelType::BETA_BINOMIAL) {
        // ---- BETA_BINOMIAL ----
        // Overdispersed binomial: y ~ BetaBinom(n, alpha, beta)
        // where p = alpha/(alpha+beta), phi = alpha + beta (concentration)
        // We parameterize: logit(p) = eta, phi = overdispersion
        // alpha = p * phi, beta_param = (1-p) * phi
        // LL = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+phi)
        //      - lgamma(alpha) - lgamma(beta) + lgamma(phi) + lchoose(n,y)
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y_i = data.y_num[i];
        int n_i = data.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;

        // d(LL)/d(eta) = d(LL)/d(p) * d(p)/d(eta) where d(p)/d(eta) = p*(1-p)
        // d(LL)/d(p) = phi * (digamma(y+alpha) - digamma(n-y+beta) - digamma(alpha) + digamma(beta))
        double psi_y_alpha = ratiod::math::portable_digamma(y_i + alpha);
        double psi_nmy_beta = ratiod::math::portable_digamma(n_i - y_i + beta_param);
        double psi_alpha = ratiod::math::portable_digamma(alpha);
        double psi_beta = ratiod::math::portable_digamma(beta_param);
        double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        resid_num = dLL_dp * p * (1.0 - p);

        // d(LL)/d(phi) where phi = alpha + beta
        // d(LL)/d(phi) = p*digamma(y+alpha) + (1-p)*digamma(n-y+beta) - digamma(n+phi)
        //               - p*digamma(alpha) - (1-p)*digamma(beta) + digamma(phi)
        double psi_n_phi = ratiod::math::portable_digamma(n_i + phi_num);
        double psi_phi = ratiod::math::portable_digamma(phi_num);
        grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;

        // No denominator contribution for beta-binomial (like binomial)
      }

      // Accumulate ZI coefficient gradients
      if (layout.has_zi && data.p_zi > 0) {
        for (int j = 0; j < data.p_zi; j++) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad_beta_zi[j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
          #else
          grad_beta_zi[j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
          #endif
        }
      }

      // Accumulate OI coefficient gradients
      if (layout.has_oi && data.p_oi > 0) {
        for (int j = 0; j < data.p_oi; j++) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad_beta_oi[j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
          #else
          grad_beta_oi[j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
          #endif
        }
      }

      // Accumulate beta gradients: grad += X[i,:] * resid
      for (int j = 0; j < data.p_num; j++) {
        #ifdef _OPENMP
        local_grad_beta_num[j] += data.X_num_flat[i * data.p_num + j] * resid_num;
        #else
        grad_beta_num[j] += data.X_num_flat[i * data.p_num + j] * resid_num;
        #endif
      }
      // For BINOMIAL, beta_denom doesn't affect likelihood
      if (data.model_type != ModelType::BINOMIAL) {
        for (int j = 0; j < data.p_denom; j++) {
          #ifdef _OPENMP
          local_grad_beta_denom[j] += data.X_denom_flat[i * data.p_denom + j] * resid_denom;
          #else
          grad_beta_denom[j] += data.X_denom_flat[i * data.p_denom + j] * resid_denom;
          #endif
        }
      }

      // Accumulate RE gradient
      if (layout.has_re_slopes && re_group_idx > 0) {
        // Random slopes case: gradient for intercept and each slope
        double re_grad_base = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_base += resid_denom;
        }
        int g = re_group_idx - 1;
        int n_coefs = re_n_coefs_i;

        // Intercept gradient: same as simple RE
        #ifdef _OPENMP
        #pragma omp atomic
        grad_re_slopes_lik[0][g * n_coefs] += re_grad_base;
        #else
        grad_re_slopes_lik[0][g * n_coefs] += re_grad_base;
        #endif

        // Slope gradients: multiply by slope design value
        int n_slopes = n_coefs - 1;
        for (int s = 0; s < n_slopes; s++) {
          double slope_grad = re_grad_base * re_slope_x_i[s];
          #ifdef _OPENMP
          #pragma omp atomic
          grad_re_slopes_lik[0][g * n_coefs + 1 + s] += slope_grad;
          #else
          grad_re_slopes_lik[0][g * n_coefs + 1 + s] += slope_grad;
          #endif
        }
      } else if (n_crossed_terms > 0) {
        // Crossed RE (multiple intercept-only terms)
        double re_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;
        }
        for (int t = 0; t < n_crossed_terms; t++) {
          if (re_idx_multi_buf[t] >= 0) {
            #ifdef _OPENMP
            // Thread-local accumulation (reduced at end of parallel block)
            local_grad_re_crossed[data.re_offsets[t] + re_idx_multi_buf[t]] += re_grad_i;
            #else
            grad[layout.re_start_multi[t] + re_idx_multi_buf[t]] += re_grad_i;
            #endif
          }
        }
      } else if (re_idx >= 0) {
        // Simple intercept-only RE (single term)
        double re_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;  // Shared RE affects both processes
        }
        #ifdef _OPENMP
        local_grad_re[re_idx] += re_grad_i;
        #else
        grad[layout.re_start + re_idx] += re_grad_i;
        #endif
      }

      // Accumulate temporal gradient (from likelihood)
      if (t_idx >= 0) {
        double temp_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          temp_grad_i += resid_denom;
        }
        #ifdef _OPENMP
        #pragma omp atomic
        grad_temporal_lik[t_idx] += temp_grad_i;
        #else
        grad_temporal_lik[t_idx] += temp_grad_i;
        #endif
      }

      // Accumulate spatial gradient (from likelihood)
      if (s_idx >= 0) {
        double lik_grad = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          lik_grad += resid_denom;
        }
        // For ICAR: grad_spatial[s] += lik_grad (d_spatial_d_phi = 1)
        // For BYM2: grad_phi[s] += lik_grad * d_spatial_d_phi, grad_theta[s] += lik_grad * d_spatial_d_theta
        #ifdef _OPENMP
        #pragma omp atomic
        grad_spatial_lik[s_idx] += lik_grad * d_spatial_d_phi;
        #else
        grad_spatial_lik[s_idx] += lik_grad * d_spatial_d_phi;
        #endif

        if (data.spatial_type == SpatialType::BYM2) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad[layout.theta_bym2_start + s_idx] += lik_grad * d_spatial_d_theta;
          #else
          grad[layout.theta_bym2_start + s_idx] += lik_grad * d_spatial_d_theta;
          #endif
        }
      }

      // Accumulate phi gradients
      #ifdef _OPENMP
      local_grad_phi_num += grad_phi_num_i;
      local_grad_phi_denom += grad_phi_denom_i;
      #else
      grad_phi_num_lik += grad_phi_num_i;
      grad_phi_denom_lik += grad_phi_denom_i;
      #endif

      // Fused log-likelihood: compute per-observation log_lik using already-computed
      // intermediates. This avoids a separate O(N) pass through compute_log_post.
      if (compute_lp) {
        double ll_i = 0.0;
        if (data.model_type == ModelType::BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int n_trials = data.y_denom[i];
          int y_i = data.y_num[i];
          if (data.zi_type == ratiod_zi::ZIType::ZI_BINOMIAL) {
            ll_i = ratiod_zi::zi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == ratiod_zi::ZIType::HURDLE_BINOMIAL) {
            ll_i = ratiod_zi::hurdle_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == ratiod_zi::ZIType::OI_BINOMIAL) {
            ll_i = ratiod_zi::oi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_oi);
          } else if (data.zi_type == ratiod_zi::ZIType::ZOIB) {
            ll_i = ratiod_zi::zoib_lpmf_logit(y_i, n_trials, p_i, logit_zi, logit_oi);
          } else {
            ll_i = log_lik_binomial(y_i, n_trials, eta_num);
          }
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_negbin(data.y_denom[i], mu_denom_i, phi_denom);
        } else if (data.model_type == ModelType::POISSON_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_poisson(data.y_num[i], mu_num_i);
          }
          ll_i += log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom_i);
        } else if (data.model_type == ModelType::GAMMA_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          ll_i = log_lik_gamma(data.y_num_cont[i], phi_num, mu_num_i);
          ll_i += log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.model_type == ModelType::LOGNORMAL) {
          double log_y_num_i = std::log(data.y_num_cont[i]);
          double log_y_denom_i = std::log(data.y_denom_cont[i]);
          double z_num_i = (log_y_num_i - eta_num) / phi_num;
          double z_denom_i = (log_y_denom_i - eta_denom) / phi_denom;
          ll_i = -log_y_num_i - std::log(phi_num) - 0.5 * z_num_i * z_num_i;
          ll_i += -log_y_denom_i - std::log(phi_denom) - 0.5 * z_denom_i * z_denom_i;
        } else if (data.model_type == ModelType::BETA_BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int y_i = data.y_num[i];
          int n_i = data.y_denom[i];
          double alpha_i = p_i * phi_num;
          double beta_i = (1.0 - p_i) * phi_num;
          ll_i = std::lgamma(y_i + alpha_i) + std::lgamma(n_i - y_i + beta_i) - std::lgamma(n_i + phi_num);
          ll_i += -std::lgamma(alpha_i) - std::lgamma(beta_i) + std::lgamma(phi_num);
          ll_i += ratiod::math::portable_lchoose(n_i, y_i);
        }
        #ifdef _OPENMP
        local_obs_ll += ll_i;
        #else
        obs_log_lik += ll_i;
        #endif
      }
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
      grad_phi_num_lik += local_grad_phi_num;
      grad_phi_denom_lik += local_grad_phi_denom;
      for (int g = 0; g < (int)local_grad_re.size(); g++) {
        grad[layout.re_start + g] += local_grad_re[g];
      }
      // Reduce crossed RE thread-local buffers to scattered grad positions
      for (int t = 0; t < (int)data.re_n_groups_multi.size(); t++) {
        int re_start_t = layout.re_start_multi[t];
        int offset_t = data.re_offsets[t];
        for (int g = 0; g < data.re_n_groups_multi[t]; g++) {
          grad[re_start_t + g] += local_grad_re_crossed[offset_t + g];
        }
      }
      obs_log_lik += local_obs_ll;
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

  // Phi gradients (with Jacobian for log transform)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // ZI coefficient gradients (likelihood contribution)
  if (layout.has_zi && data.p_zi > 0) {
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] += grad_beta_zi[j];
    }
  }

  // OI coefficient gradients (likelihood contribution)
  if (layout.has_oi && data.p_oi > 0) {
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] += grad_beta_oi[j];
    }
  }

  // Random slopes likelihood gradients
  if (layout.has_re_slopes && n_re_terms_slopes > 0) {
    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_nc = (t < (int)nc_L_flats.size() && !nc_L_flats[t].empty());

      if (is_nc) {
        // Non-centered correlated slopes: chain rule transformation
        // grad_re_slopes_lik[t] contains dLL/d(re), but params store z.
        // Need to transform to dLL/d(z) and add sigma/chol gradients.
        const auto& L_flat = nc_L_flats[t];
        const auto& sigmas = nc_sigmas_vec[t];

        // 1. Transform grad_re_lik to grad_z via chain rule:
        //    dLL/dz[g,k] = sum_{c>=k} dLL/dre[g,c] * sigma[c] * L[c,k]
        for (int g = 0; g < n_groups; g++) {
          for (int k = 0; k < n_coefs; k++) {
            double grad_z_lik = 0.0;
            for (int c = k; c < n_coefs; c++) {
              grad_z_lik += grad_re_slopes_lik[t][g * n_coefs + c] *
                            sigmas[c] * L_flat[c * n_coefs + k];
            }
            grad[re_start_t + g * n_coefs + k] += grad_z_lik;
          }
        }

        // 2. Sigma gradient from likelihood:
        //    dLL/d(log_sigma[c]) = sum_g dLL/dre[g,c] * re_nc[g,c]
        for (int c = 0; c < n_coefs; c++) {
          double sigma_lik_grad = 0.0;
          for (int g = 0; g < n_groups; g++) {
            sigma_lik_grad += grad_re_slopes_lik[t][g * n_coefs + c] *
                              re_nc_flat[re_start_t + g * n_coefs + c];
          }
          int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
          grad[log_sigma_idx] += sigma_lik_grad;
        }

        // 3. Cholesky gradient from likelihood:
        //    dLL/dL[i,j] = sigma[i] * (S_ij - S_ii * L[i,j] / L[i,i])
        //    where S_ik = sum_g dLL/dre[g,i] * z[g,k]
        //    Then apply tanh chain rule: grad_raw += grad_L * sech^2
        int chol_start = layout.chol_re_start_multi[t];
        for (int ii = 1; ii < n_coefs; ii++) {
          double L_ii = L_flat[ii * n_coefs + ii];
          // Compute S_i_k for k = 0..ii
          std::vector<double> S_i(ii + 1, 0.0);
          for (int k = 0; k <= ii; k++) {
            for (int g = 0; g < n_groups; g++) {
              S_i[k] += grad_re_slopes_lik[t][g * n_coefs + ii] *
                        params[re_start_t + g * n_coefs + k];  // z[g,k]
            }
          }

          int chol_base = ii * (ii - 1) / 2;
          for (int j = 0; j < ii; j++) {
            double L_ij = L_flat[ii * n_coefs + j];
            double grad_L_ij = sigmas[ii] * (S_i[j] - S_i[ii] * L_ij / L_ii);

            // Apply tanh chain rule and ADD to existing chol gradient
            double raw_val = params[chol_start + chol_base + j];
            double l_val = std::tanh(raw_val);
            double sech2 = 1.0 - l_val * l_val;
            grad[chol_start + chol_base + j] += grad_L_ij * sech2;
          }
        }

      } else {
        // Uncorrelated or centered: add grad_re_slopes_lik directly
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] += grad_re_slopes_lik[t][g * n_coefs + c];
          }
        }
      }
    }
  }

  } // end if (!used_vectorized)

  // ============ Temporal GMRF prior gradients ============
  if (layout.has_temporal && T_len > 0) {
    // Initialize temporal gradients with likelihood contribution
    for (int t = 0; t < T_len; t++) {
      grad[layout.temporal_start + t] = grad_temporal_lik[t];
    }

    if (data.temporal_type == TemporalType::RW1) {
      // RW1: -0.5 * tau * sum((phi[t] - phi[t-1])^2)
      // d/d(phi[t]) = tau * (phi[t-1] - phi[t]) + tau * (phi[t+1] - phi[t]) (interior)
      double quad_form = 0.0;
      for (int t = 0; t < T_len; t++) {
        double grad_t = 0.0;
        if (t > 0) {
          grad_t += tau_temporal * (phi_temporal[t - 1] - phi_temporal[t]);
          quad_form += (phi_temporal[t] - phi_temporal[t - 1]) * (phi_temporal[t] - phi_temporal[t - 1]);
        }
        if (t < T_len - 1) {
          grad_t += tau_temporal * (phi_temporal[t + 1] - phi_temporal[t]);
        }
        grad[layout.temporal_start + t] += grad_t;
      }
      // tau gradient: 0.5*(T-1)*log(tau) - 0.5*tau*quad => d/d(log_tau) = 0.5*(T-1) - 0.5*tau*quad
      grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 1) - 0.5 * tau_temporal * quad_form;
      // Sum-to-zero penalty gradient: d/d(phi[t]) of [-0.5*lambda*sum(phi)^2] = -lambda*sum(phi)
      { double sum_phi = 0.0;
        for (int t = 0; t < T_len; t++) sum_phi += phi_temporal[t];
        for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sum_phi;
      }

    } else if (data.temporal_type == TemporalType::RW2) {
      // RW2: second-order differences
      double quad_form = 0.0;
      for (int t = 0; t < T_len; t++) {
        double grad_t = 0.0;
        if (t >= 2) {
          double d2 = phi_temporal[t - 2] - 2.0 * phi_temporal[t - 1] + phi_temporal[t];
          grad_t -= tau_temporal * d2;  // coefficient of phi[t] in (phi[t-2] - 2*phi[t-1] + phi[t])^2 is +1
        }
        if (t >= 1 && t < T_len - 1) {
          double d2 = phi_temporal[t - 1] - 2.0 * phi_temporal[t] + phi_temporal[t + 1];
          grad_t += 2.0 * tau_temporal * d2;  // coefficient of phi[t] is -2
        }
        if (t < T_len - 2) {
          double d2 = phi_temporal[t] - 2.0 * phi_temporal[t + 1] + phi_temporal[t + 2];
          grad_t -= tau_temporal * d2;  // coefficient of phi[t] is +1
        }
        grad[layout.temporal_start + t] += grad_t;
      }
      // Compute quadratic form for tau gradient
      for (int t = 2; t < T_len; t++) {
        double d2 = phi_temporal[t - 2] - 2.0 * phi_temporal[t - 1] + phi_temporal[t];
        quad_form += d2 * d2;
      }
      grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 2) - 0.5 * tau_temporal * quad_form;
      // Sum-to-zero penalty gradient
      { double sum_phi = 0.0;
        for (int t = 0; t < T_len; t++) sum_phi += phi_temporal[t];
        for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sum_phi;
      }

    } else if (data.temporal_type == TemporalType::AR1) {
      // Centered AR1: phi[0] ~ N(0, 1/(tau*(1-rho^2))), phi[t] | phi[t-1] ~ N(rho*phi[t-1], 1/tau)
      double one_minus_rho2 = 1.0 - rho_ar1 * rho_ar1;

      // Gradient for phi[0]: d/d(phi[0]) = -tau*(1-rho^2)*phi[0] + tau*rho*(phi[1] - rho*phi[0])
      grad[layout.temporal_start] += -tau_temporal * one_minus_rho2 * phi_temporal[0];
      if (T_len > 1) {
        grad[layout.temporal_start] += tau_temporal * rho_ar1 * (phi_temporal[1] - rho_ar1 * phi_temporal[0]);
      }

      // Gradient for phi[t], t > 0
      double quad_form = one_minus_rho2 * phi_temporal[0] * phi_temporal[0];
      for (int t = 1; t < T_len; t++) {
        double resid = phi_temporal[t] - rho_ar1 * phi_temporal[t - 1];
        quad_form += resid * resid;
        double grad_t = -tau_temporal * resid;
        if (t < T_len - 1) {
          double resid_next = phi_temporal[t + 1] - rho_ar1 * phi_temporal[t];
          grad_t += tau_temporal * rho_ar1 * resid_next;
        }
        grad[layout.temporal_start + t] += grad_t;
      }

      // tau gradient: 0.5*T*log(tau) + 0.5*log(1-rho^2) - 0.5*tau*quad
      grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * quad_form;

      // rho gradient
      if (layout.logit_rho_ar1_idx >= 0) {
        double grad_rho = -rho_ar1 / one_minus_rho2;  // From 0.5*log(1-rho^2)
        grad_rho += tau_temporal * rho_ar1 * phi_temporal[0] * phi_temporal[0];  // From first term
        for (int t = 1; t < T_len; t++) {
          double resid = phi_temporal[t] - rho_ar1 * phi_temporal[t - 1];
          grad_rho += tau_temporal * resid * phi_temporal[t - 1];
        }
        // Chain rule: d/d(logit_rho) = d/d(rho) * d(rho)/d(logit_rho) = grad_rho * rho * (1 - rho)
        grad[layout.logit_rho_ar1_idx] += grad_rho * rho_ar1 * (1.0 - rho_ar1);
      }
    }
  }

  // ============ Spatial GMRF prior gradients (ICAR and BYM2) ============
  if (layout.has_spatial && n_spatial > 0) {
    // Add likelihood contribution to phi_spatial gradients
    for (int s = 0; s < n_spatial; s++) {
      grad[layout.spatial_start + s] = grad_spatial_lik[s];
    }

    // ICAR prior: -0.5 * tau * phi' * Q * phi where Q_ij = n_neighbors[i] if i=j, -1 if i~j
    // d/d(phi[i]) = -tau * (n_neighbors[i]*phi[i] - sum_{j~i} phi[j])
    double icar_quad = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      double Qphi_i = data.n_neighbors[i] * phi_spatial[i];
      int row_start = data.adj_row_ptr[i];
      int row_end = data.adj_row_ptr[i + 1];
      for (int k = row_start; k < row_end; k++) {
        int j = data.adj_col_idx[k];
        Qphi_i -= phi_spatial[j];
        if (j > i) {
          double diff = phi_spatial[i] - phi_spatial[j];
          icar_quad += diff * diff;
        }
      }

      if (data.spatial_type == SpatialType::BYM2) {
        // For BYM2, ICAR prior has no tau scaling (it's absorbed into sigma/rho)
        grad[layout.spatial_start + i] += -Qphi_i;
      } else {
        // For plain ICAR
        grad[layout.spatial_start + i] += -tau_spatial * Qphi_i;
      }
    }

    if (data.spatial_type == SpatialType::BYM2) {
      // BYM2 Riebler: transform (grad_sigma_s, grad_sigma_u) to (grad_log_sigma, grad_logit_rho)
      // grad_sigma_s_lik = d(LL)/d(sigma_s) * sigma_s  (chain rule for log)
      // grad_sigma_u_lik = d(LL)/d(sigma_u) * sigma_u
      double grad_sigma_s_lik = 0.0;
      double grad_sigma_u_lik = 0.0;

      for (int s = 0; s < n_spatial; s++) {
        double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
        double d_LL_d_spatial = grad_spatial_lik[s] / (sigma_s_bym2 * data.bym2_scale_factor);
        grad_sigma_s_lik += d_LL_d_spatial * sigma_s_bym2 * scaled_phi;
        grad_sigma_u_lik += d_LL_d_spatial * sigma_u_bym2 * theta_bym2[s];
      }

      // grad[log_sigma] = grad_sigma_s_lik + grad_sigma_u_lik
      grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
      // grad[logit_rho] = 0.5 * ((1-rho)*grad_sigma_s_lik - rho*grad_sigma_u_lik)
      grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                  - rho_bym2 * grad_sigma_u_lik);

    } else {
      // Plain ICAR: tau gradient
      // log_post = 0.5*(n-1)*log(tau) - 0.5*tau*quad + const
      // d/d(log_tau) = 0.5*(n-1) - 0.5*tau*quad
      grad[layout.log_tau_spatial_idx] += 0.5 * (n_spatial - 1) - 0.5 * tau_spatial * icar_quad;
    }
  }

  // Fused log-posterior output: combine prior/structural terms with observation log-lik.
  // Prior/structural terms are computed via compute_log_post with skip_obs_loop=true (O(p+S+T)).
  // Observation log-lik was accumulated inline during the gradient computation (O(N)).
  // Total: one O(N) pass instead of two.
  if (log_post_out) {
    *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
  }

}

// =====================================================================
// Numerical gradient (fallback for complex models)
// =====================================================================

void compute_gradient_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n = params.size();
  grad.resize(n);

  // Compute log_post at central point if requested (cheap: one extra eval)
  if (log_post_out) {
    *log_post_out = compute_log_post(params, data, layout);
  }

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
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    using namespace ratiod::ad;

    // Thread-safe: each call gets its own tape via RAII
    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    // Create autodiff variables from parameters
    std::vector<Var> params_ad = make_vars(tape, params);

    // Compute log posterior using templated implementation
    Var log_post = ratiod::compute_log_post_impl(params_ad, data, layout);

    // Extract log_post value before backward pass (free: already computed)
    if (log_post_out) *log_post_out = log_post.val();

    // Backward pass to compute gradients
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// Arena-based reverse-mode autodiff gradient (O(N) - fast, all models)
// Uses contiguous SoA memory layout with pre-computed partials.
// ~10-30x faster than tape autodiff, within 50% of hand-coded speed.
// =====================================================================

void compute_gradient_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    using namespace ratiod::arena;

    // Thread-safe: each call gets its own arena via RAII
    ArenaScope scope;
    Arena* arena = scope.arena();

    // Create autodiff variables from parameters
    std::vector<Var> params_ar = make_vars(arena, params);

    // Compute log posterior using templated implementation
    Var log_post = ratiod::compute_log_post_impl(params_ar, data, layout);

    // Extract log_post value before backward pass
    if (log_post_out) *log_post_out = log_post.val();

    // Backward pass to compute gradients
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ar);

    // ArenaScope destructor handles cleanup
}

// =====================================================================
// Forward-mode autodiff gradient (O(n×p) - but ~10x faster than tape)
// Uses dual numbers for efficient gradient computation without heap allocation
// =====================================================================

void compute_gradient_forward(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    int n_params = static_cast<int>(params.size());
    grad.assign(n_params, 0.0);

    // Forward-mode: compute one gradient component per forward pass
    // Seed each parameter in turn and evaluate
    std::vector<fwd::Dual> params_dual(n_params);

    for (int i = 0; i < n_params; i++) {
        // Seed parameter i: value=params[i], gradient=1.0
        // All others: value=params[j], gradient=0.0
        for (int j = 0; j < n_params; j++) {
            params_dual[j].val = params[j];
            params_dual[j].grad = (j == i) ? 1.0 : 0.0;
        }

        // Compute log posterior with dual numbers
        fwd::Dual log_post = ratiod::compute_log_post_impl(params_dual, data, layout);

        // Extract gradient component
        grad[i] = log_post.grad;

        // Extract log_post value on first pass (free: already computed)
        if (i == 0 && log_post_out) *log_post_out = log_post.val;
    }
}

// =====================================================================
// GP gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from gp_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // =========================================================================
    // Extract parameters
    // =========================================================================
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

    // GP parameters
    int N_gp = data.gp_data.n_obs;
    double log_sigma2_gp = params[layout.log_sigma2_gp_idx];
    double log_phi_gp = params[layout.log_phi_gp_idx];
    double sigma2_gp = std::exp(log_sigma2_gp);
    double phi_gp = std::exp(log_phi_gp);

    // Extract GP spatial effects
    std::vector<double> gp_w(N_gp);
    for (int i = 0; i < N_gp; i++) {
        gp_w[i] = params[layout.gp_w_start + i];
    }

    // Bounds check for phi
    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        return;  // Out of bounds - return zero gradient
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    // Fixed effects prior: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] -= tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] -= tau_beta * beta_denom[j];
    }

    // RE prior: Half-Cauchy on sigma, N(0, sigma_re^2) on effects
    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    // Overdispersion prior: Gamma(shape, rate) via log transform
    // d/d(log_phi) = shape - rate*phi (no extra phi factor)
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }

    // PC prior on GP variance: P(sigma > U) = alpha
    // sigma2 ~ transformed Exp, so d/d(log_sigma2) = (-rate*sigma/2 + 0.5)
    double rate_sigma = -std::log(data.gp_sigma2_prior_alpha) / data.gp_sigma2_prior_U;
    double sigma_gp = std::sqrt(sigma2_gp);
    grad[layout.log_sigma2_gp_idx] = -0.5 * rate_sigma * sigma_gp + 0.5;

    // Uniform prior on phi - just Jacobian
    grad[layout.log_phi_gp_idx] = 1.0;

    // =========================================================================
    // Compute NNGP gradients w.r.t. spatial effects (analytical)
    // =========================================================================
    ratiod_gp::NNGPGradients nngp_grads;
    ratiod_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);

    // Add NNGP prior gradient contributions for w
    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
    }

    // Add NNGP gradient contributions for GP hyperparameters
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // =========================================================================
    // Data likelihood loop
    // =========================================================================
    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            eta_num += re[g];
            eta_denom += re[g];
        }

        // GP spatial effect (map observation to unique location)
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        if (data.gp_data.shared) {
            eta_num += gp_effect;
            eta_denom += gp_effect;
        } else {
            eta_num += gp_effect;
        }

        // Fused log-posterior: accumulate obs log-lik
        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        // Likelihood gradients depend on model type
        double dLL_deta_num = 0.0;
        double dLL_deta_denom = 0.0;

        if (data.model_type == ModelType::BINOMIAL) {
            // Binomial: d(log_lik)/d(eta) = y - n*p where p = logit^{-1}(eta)
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
            // denom not used in binomial (y_denom is trials)
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            // NegBin: d(log_lik)/d(eta) = y - mu*(y+phi)/(mu+phi)
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {  // POISSON_GAMMA
            // Poisson: d(log_lik)/d(eta) = y - mu
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num;
            // Gamma: d(log_lik)/d(eta) = phi * (y/mu - 1)
            dLL_deta_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        // Accumulate gradients for fixed effects
        for (int j = 0; j < data.p_num; j++) {
            grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
        }

        // Gradients for RE
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
        }

        // Gradients for GP spatial effects (from likelihood, mapped to unique location)
        double dLL_dspatial = data.gp_data.shared ?
                              (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        grad[layout.gp_w_start + loc_i] += dLL_dspatial;

        // Gradient w.r.t. phi_num (for NegBin)
        if (layout.has_phi_num && data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double y = data.y_num[i];
            // d(log_lik)/d(phi) = digamma(y+phi) - digamma(phi) + log(phi/(mu+phi)) + 1 - (y+phi)/(mu+phi)
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
        }

        // Gradient w.r.t. phi_denom
        if (layout.has_phi_denom) {
            if (data.model_type == ModelType::NEGBIN_NEGBIN) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom[i];
                double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                                 + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                                 - (y + phi_denom) / (mu_denom + phi_denom);
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            } else if (data.model_type == ModelType::POISSON_GAMMA) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom_cont[i];
                double digamma_phi = ratiod::math::portable_digamma(phi_denom);
                double dLL_dphi = std::log(phi_denom) + 1.0 - digamma_phi
                                 + std::log(y) - std::log(mu_denom)
                                 - y / mu_denom;
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// GP + Temporal gradient (hand-coded)
// Combines GP spatial with temporal RW1/RW2/AR1
// =====================================================================

void compute_gradient_gp_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    const double* beta_num = &params[layout.beta_num_start];
    const double* beta_denom = &params[layout.beta_denom_start];
    double sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    const double* re = layout.has_re ? &params[layout.re_start] : nullptr;
    double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);
    std::vector<double> gp_w(N_gp);
    for (int i = 0; i < N_gp; i++) gp_w[i] = params[layout.gp_w_start + i];

    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    const double* phi_temporal = &params[layout.temporal_start];
    double rho_ar1 = (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        ? 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx])) : 0.5;

    // Prior gradients
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];

    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    if (layout.has_phi_num) grad[layout.log_phi_num_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_num;
    if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_denom;

    double rate_sigma = -std::log(data.gp_sigma2_prior_alpha) / data.gp_sigma2_prior_U;
    grad[layout.log_sigma2_gp_idx] = -0.5 * rate_sigma * std::sqrt(sigma2_gp) + 0.5;
    grad[layout.log_phi_gp_idx] = 1.0;
    grad[layout.log_tau_temporal_idx] = (data.tau_temporal_shape - 1.0) - data.tau_temporal_rate * tau_temporal + 1.0;
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;

    // NNGP prior gradients
    ratiod_gp::NNGPGradients nngp_grads;
    ratiod_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
    for (int i = 0; i < N_gp; i++) grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // Likelihood
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    for (int i = 0; i < data.N; i++) {
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        for (int j = 0; j < data.p_denom; j++) eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        if (layout.has_re && data.re_group[i] > 0) { eta_num += re[data.re_group[i] - 1]; eta_denom += re[data.re_group[i] - 1]; }
        int loc_i = data.gp_data.obs_to_loc[i];
        if (data.gp_data.shared) { eta_num += gp_w[loc_i]; eta_denom += gp_w[loc_i]; } else { eta_num += gp_w[loc_i]; }

        int t_idx = -1;
        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            t_idx = data.temporal_time_idx[i] - 1;
            if (t_idx >= 0 && t_idx < T_len) {
                if (data.temporal_shared) { eta_num += phi_temporal[t_idx]; eta_denom += phi_temporal[t_idx]; }
                else { eta_num += phi_temporal[t_idx]; }
            }
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        double dLL_num = 0.0, dLL_denom = 0.0;
        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num;
            dLL_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] += dLL_num * data.X_num_flat[i * data.p_num + j];
        for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] += dLL_denom * data.X_denom_flat[i * data.p_denom + j];
        if (layout.has_re && data.re_group[i] > 0) grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        grad[layout.gp_w_start + loc_i] += data.gp_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        if (t_idx >= 0 && t_idx < T_len) grad_temporal_lik[t_idx] += data.temporal_shared ? (dLL_num + dLL_denom) : dLL_num;
    }

    // Temporal GMRF
    for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] = grad_temporal_lik[t];
    if (data.temporal_type == TemporalType::RW1) {
        double qf = 0.0;
        for (int t = 0; t < T_len; t++) {
            double g = 0.0;
            if (t > 0) { g += tau_temporal * (phi_temporal[t-1] - phi_temporal[t]); qf += std::pow(phi_temporal[t] - phi_temporal[t-1], 2); }
            if (t < T_len - 1) g += tau_temporal * (phi_temporal[t+1] - phi_temporal[t]);
            grad[layout.temporal_start + t] += g;
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 1) - 0.5 * tau_temporal * qf;
        { double sp = 0.0; for (int t = 0; t < T_len; t++) sp += phi_temporal[t];
          for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
    } else if (data.temporal_type == TemporalType::RW2) {
        double qf = 0.0;
        for (int t = 0; t < T_len; t++) {
            double g = 0.0;
            if (t >= 2) g -= tau_temporal * (phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t]);
            if (t >= 1 && t < T_len - 1) g += 2.0 * tau_temporal * (phi_temporal[t-1] - 2.0*phi_temporal[t] + phi_temporal[t+1]);
            if (t < T_len - 2) g -= tau_temporal * (phi_temporal[t] - 2.0*phi_temporal[t+1] + phi_temporal[t+2]);
            grad[layout.temporal_start + t] += g;
        }
        for (int t = 2; t < T_len; t++) qf += std::pow(phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t], 2);
        grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 2) - 0.5 * tau_temporal * qf;
        { double sp = 0.0; for (int t = 0; t < T_len; t++) sp += phi_temporal[t];
          for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
    } else if (data.temporal_type == TemporalType::AR1) {
        double omr2 = 1.0 - rho_ar1 * rho_ar1;
        grad[layout.temporal_start] += -tau_temporal * omr2 * phi_temporal[0];
        if (T_len > 1) grad[layout.temporal_start] += tau_temporal * rho_ar1 * (phi_temporal[1] - rho_ar1 * phi_temporal[0]);
        double qf = omr2 * phi_temporal[0] * phi_temporal[0];
        for (int t = 1; t < T_len; t++) {
            double r = phi_temporal[t] - rho_ar1 * phi_temporal[t-1]; qf += r * r;
            double g = -tau_temporal * r;
            if (t < T_len - 1) g += tau_temporal * rho_ar1 * (phi_temporal[t+1] - rho_ar1 * phi_temporal[t]);
            grad[layout.temporal_start + t] += g;
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * qf;
        if (layout.logit_rho_ar1_idx >= 0) {
            double gr = -rho_ar1 / omr2 + tau_temporal * rho_ar1 * phi_temporal[0] * phi_temporal[0];
            for (int t = 1; t < T_len; t++) gr += tau_temporal * (phi_temporal[t] - rho_ar1 * phi_temporal[t-1]) * phi_temporal[t-1];
            grad[layout.logit_rho_ar1_idx] += gr * rho_ar1 * (1.0 - rho_ar1);
        }
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Temporal GP (standalone) hand-coded gradients
// Temporal GP with exponential covariance uses state-space AR(1) form
// =====================================================================

void compute_gradient_temporal_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract parameters
    const double* beta_num = &params[layout.beta_num_start];
    const double* beta_denom = &params[layout.beta_denom_start];
    double sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    const double* re = layout.has_re ? &params[layout.re_start] : nullptr;
    double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

    // Temporal GP hyperparameters (on log scale in params)
    double sigma2_tgp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
    double phi_tgp = std::exp(params[layout.log_phi_temporal_gp_idx]);

    // Temporal effects: n_temporal_groups * n_times parameters
    int T_times = data.n_times;
    int n_groups = data.n_temporal_groups;
    const double* phi_temporal = &params[layout.temporal_start];

    // ---- Prior gradients ----

    // Beta priors: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];

    // RE priors: Half-Cauchy on sigma_re + N(0, sigma_re^2) on each RE
    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    // Dispersion priors: Gamma on phi via log transform, d/d(log_phi) = shape - rate*phi
    if (layout.has_phi_num) grad[layout.log_phi_num_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_num;
    if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_denom;

    // Temporal GP hyperparameter priors
    // sigma2: PC prior => d/d(log_sigma2) [ log(rate) - rate*sqrt(sigma2) - log(2*sqrt(sigma2)) + log_sigma2 ]
    //       = -0.5 * rate * sqrt(sigma2) + 0.5
    double rate_tgp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
    grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_tgp * std::sqrt(sigma2_tgp) + 0.5;

    // phi: Uniform prior on (lower, upper), parameterized on log scale
    // d/d(log_phi) [ -log(upper - lower) + log_phi ] = 1.0 (Jacobian only)
    // Bounds check: if out of bounds, return -inf gradient (will be rejected)
    if (phi_tgp < data.temporal_gp_phi_prior_lower || phi_tgp > data.temporal_gp_phi_prior_upper) {
        // Out of bounds - set large negative gradient to push back
        grad[layout.log_phi_temporal_gp_idx] = (phi_tgp < data.temporal_gp_phi_prior_lower) ? 1e6 : -1e6;
        return;
    }
    grad[layout.log_phi_temporal_gp_idx] = 1.0;

    // ---- Temporal GP prior gradients (state-space exponential form) ----
    // For each group g, with T time points and effects f[0..T-1]:
    //   f[0] ~ N(0, sigma2)
    //   f[t] | f[t-1] ~ N(rho_t * f[t-1], sigma2 * (1 - rho_t^2))
    // where rho_t = exp(-dt / phi)

    int T_len = layout.temporal_end - layout.temporal_start;
    for (int g = 0; g < n_groups; g++) {
        int offset = g * T_times;

        // First time point: N(0, sigma2)
        double f0 = phi_temporal[offset];
        grad[layout.temporal_start + offset] += -f0 / sigma2_tgp;

        // Accumulate hyperparameter gradients
        // d/d(log_sigma2) from first time point:
        //   -0.5 * log(sigma2) - 0.5 * f0^2 / sigma2
        //   => d/d(log_sigma2) = -0.5 + 0.5 * f0^2 / sigma2
        double grad_log_sigma2_prior = -0.5 + 0.5 * f0 * f0 / sigma2_tgp;
        double grad_log_phi_prior = 0.0;

        // Subsequent time points
        for (int t = 1; t < T_times; t++) {
            double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
            double rho = std::exp(-dt / phi_tgp);
            double rho2 = rho * rho;
            double cv = sigma2_tgp * (1.0 - rho2);
            if (cv < 1e-10) cv = 1e-10;

            double f_prev = phi_temporal[offset + t - 1];
            double f_curr = phi_temporal[offset + t];
            double r = f_curr - rho * f_prev;  // residual

            // Gradients w.r.t. temporal effects
            grad[layout.temporal_start + offset + t] += -r / cv;
            grad[layout.temporal_start + offset + t - 1] += rho * r / cv;

            // Accumulate hyperparameter gradients
            // d/d(log_sigma2) from conditional:
            //   -0.5 * log(cv) - 0.5 * r^2 / cv
            //   cv = sigma2 * (1 - rho^2), so d(log cv)/d(log sigma2) = 1
            //   => d/d(log_sigma2) = -0.5 + 0.5 * r^2 / cv
            grad_log_sigma2_prior += -0.5 + 0.5 * r * r / cv;

            // d/d(log_phi) from conditional:
            //   d(rho)/d(log_phi) = (dt / phi) * rho  [since d(log_phi) = d(phi)/phi]
            //   d(cv)/d(log_phi) = sigma2 * (-2 * rho * d(rho)/d(log_phi))
            //                    = -2 * sigma2 * rho^2 * (dt / phi)
            //   d(cond_mean)/d(log_phi) = f_prev * d(rho)/d(log_phi) = f_prev * rho * (dt / phi)
            //   d(r)/d(log_phi) = -f_prev * rho * (dt / phi)
            //
            //   d(log_lik)/d(log_phi) = -0.5 * (1/cv) * d(cv)/d(log_phi)
            //                          + (r / cv) * (-d(r)/d(log_phi))
            //                          - 0.5 * r^2 * (-1/cv^2) * d(cv)/d(log_phi)
            //                         = (dt/phi) * [ sigma2*rho^2/cv
            //                                        + rho*r*f_prev/cv
            //                                        + sigma2*rho^2*r^2/cv^2 ]
            double dt_over_phi = dt / phi_tgp;
            grad_log_phi_prior += dt_over_phi * (
                sigma2_tgp * rho2 / cv
                + rho * r * f_prev / cv
                + sigma2_tgp * rho2 * r * r / (cv * cv)
            );
        }

        grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_prior;
        grad[layout.log_phi_temporal_gp_idx] += grad_log_phi_prior;
    }

    // ---- Likelihood gradients ----
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    double grad_phi_num_lik = 0.0;
    double grad_phi_denom_lik = 0.0;

    for (int i = 0; i < data.N; i++) {
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        for (int j = 0; j < data.p_denom; j++) eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];

        if (layout.has_re && data.re_group[i] > 0) {
            eta_num += re[data.re_group[i] - 1];
            eta_denom += re[data.re_group[i] - 1];
        }

        // Temporal effect
        int t_idx = -1;
        int g_idx = 0;
        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            t_idx = data.temporal_time_idx[i] - 1;
            g_idx = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g_idx * T_times + t_idx;
            if (flat_idx >= 0 && flat_idx < T_len) {
                if (data.temporal_shared) { eta_num += phi_temporal[flat_idx]; eta_denom += phi_temporal[flat_idx]; }
                else { eta_num += phi_temporal[flat_idx]; }
            }
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        // Family-specific residuals
        double resid_num = 0.0, resid_denom = 0.0;
        double grad_phi_num_i = 0.0, grad_phi_denom_i = 0.0;

        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            resid_num = data.y_num[i] - data.y_denom[i] * p;

        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            double denom_num = mu_num + phi_num, denom_d = mu_denom + phi_denom;
            resid_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / denom_num;
            resid_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / denom_d;
            grad_phi_num_i = ratiod::math::portable_digamma(data.y_num[i] + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - data.y_num[i]) / denom_num;
            grad_phi_denom_i = ratiod::math::portable_digamma(data.y_denom[i] + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                               + std::log(phi_denom / denom_d)
                               + (mu_denom - data.y_denom[i]) / denom_d;

        } else if (data.model_type == ModelType::POISSON_GAMMA) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            resid_num = data.y_num[i] - mu_num;
            double y_denom_i = data.y_denom_cont[i];
            resid_denom = phi_num * (y_denom_i / mu_denom - 1.0);
            double rate = phi_num / mu_denom;
            grad_phi_num_i = std::log(rate) + 1.0 + std::log(y_denom_i)
                             - ratiod::math::portable_digamma(phi_num) - rate * y_denom_i / phi_num;

        } else if (data.model_type == ModelType::GAMMA_GAMMA) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            double y_num_i = data.y_num_cont[i], y_denom_i = data.y_denom_cont[i];
            resid_num = phi_num * (y_num_i / mu_num - 1.0);
            resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate_num = phi_num / mu_num, rate_denom = phi_denom / mu_denom;
            grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                             - ratiod::math::portable_digamma(phi_num) - y_num_i / mu_num;
            grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                               - ratiod::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;

        } else if (data.model_type == ModelType::LOGNORMAL) {
            double mu_num = eta_num, mu_denom = eta_denom;
            double y_num_i = data.y_num_cont[i], y_denom_i = data.y_denom_cont[i];
            double log_y_num = std::log(y_num_i), log_y_denom = std::log(y_denom_i);
            double sigma_num = phi_num, sigma_denom = phi_denom;
            double sigma_num_sq = sigma_num * sigma_num, sigma_denom_sq = sigma_denom * sigma_denom;
            resid_num = (log_y_num - mu_num) / sigma_num_sq;
            resid_denom = (log_y_denom - mu_denom) / sigma_denom_sq;
            double z_num = (log_y_num - mu_num) / sigma_num;
            double z_denom = (log_y_denom - mu_denom) / sigma_denom;
            grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
            grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

        } else if (data.model_type == ModelType::BETA_BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            int y_i = data.y_num[i], n_i = data.y_denom[i];
            double alpha = p * phi_num, beta_param = (1.0 - p) * phi_num;
            double psi_y_alpha = ratiod::math::portable_digamma(y_i + alpha);
            double psi_nmy_beta = ratiod::math::portable_digamma(n_i - y_i + beta_param);
            double psi_alpha = ratiod::math::portable_digamma(alpha);
            double psi_beta = ratiod::math::portable_digamma(beta_param);
            double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
            resid_num = dLL_dp * p * (1.0 - p);
            double psi_n_phi = ratiod::math::portable_digamma(n_i + phi_num);
            double psi_phi = ratiod::math::portable_digamma(phi_num);
            grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                             - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;
        }

        // Accumulate gradients
        for (int j = 0; j < data.p_num; j++)
            grad[layout.beta_num_start + j] += resid_num * data.X_num_flat[i * data.p_num + j];
        for (int j = 0; j < data.p_denom; j++)
            grad[layout.beta_denom_start + j] += resid_denom * data.X_denom_flat[i * data.p_denom + j];

        if (layout.has_re && data.re_group[i] > 0)
            grad[layout.re_start + data.re_group[i] - 1] += resid_num + resid_denom;

        // Temporal effect gradient
        if (t_idx >= 0) {
            int flat_idx = g_idx * T_times + t_idx;
            if (flat_idx >= 0 && flat_idx < T_len)
                grad_temporal_lik[flat_idx] += data.temporal_shared ? (resid_num + resid_denom) : resid_num;
        }

        // Dispersion gradients
        grad_phi_num_lik += grad_phi_num_i;
        grad_phi_denom_lik += grad_phi_denom_i;
    }

    // Add likelihood contribution to temporal effects
    for (int t = 0; t < T_len; t++) {
        grad[layout.temporal_start + t] += grad_temporal_lik[t];
    }

    // Dispersion gradient accumulation (on log scale: multiply by phi)
    if (layout.has_phi_num) grad[layout.log_phi_num_idx] += grad_phi_num_lik * phi_num;
    if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] += grad_phi_denom_lik * phi_denom;

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Multi-scale GP + Temporal hand-coded gradients
// Combines MSGP spatial gradients with temporal GMRF gradients
// =====================================================================

void compute_gradient_msgp_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract parameters
    const double* beta_num = &params[layout.beta_num_start];
    const double* beta_denom = &params[layout.beta_denom_start];
    double sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    const double* re = layout.has_re ? &params[layout.re_start] : nullptr;
    double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

    // Multi-scale GP parameters
    int N_gp = data.multiscale_gp_data.n_obs;
    double sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
    double phi_local = std::exp(params[layout.log_phi_gp_local_idx]);
    double sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
    double phi_regional = std::exp(params[layout.log_phi_gp_regional_idx]);

    std::vector<double> w_local(N_gp), w_regional(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local[i] = params[layout.gp_local_start + i];
        w_regional[i] = params[layout.gp_regional_start + i];
    }

    // Temporal parameters
    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    const double* phi_temporal = &params[layout.temporal_start];
    double rho_ar1 = (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        ? 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx])) : 0.5;

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return;
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];

    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    if (layout.has_phi_num) grad[layout.log_phi_num_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_num;
    if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_denom;

    // PC priors on MSGP variances
    double rate_sigma_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
    grad[layout.log_sigma2_gp_local_idx] = -0.5 * rate_sigma_local * std::sqrt(sigma2_local) + 0.5;
    double rate_sigma_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
    grad[layout.log_sigma2_gp_regional_idx] = -0.5 * rate_sigma_regional * std::sqrt(sigma2_regional) + 0.5;
    grad[layout.log_phi_gp_local_idx] = 1.0;
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // Temporal prior
    grad[layout.log_tau_temporal_idx] = (data.tau_temporal_shape - 1.0) - data.tau_temporal_rate * tau_temporal + 1.0;
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;

    // =========================================================================
    // NNGP prior gradients for multi-scale GP
    // =========================================================================
    GPData gp_local;
    gp_local.n_obs = data.multiscale_gp_data.n_obs;
    gp_local.nn = data.multiscale_gp_data.nn_local;
    gp_local.coords = data.multiscale_gp_data.coords;
    gp_local.nn_idx = data.multiscale_gp_data.nn_idx_local;
    gp_local.nn_dist = data.multiscale_gp_data.nn_dist_local;
    gp_local.nn_order = data.multiscale_gp_data.nn_order_local;
    gp_local.nn_order_inv = data.multiscale_gp_data.nn_order_inv_local;
    gp_local.cov_type = data.multiscale_gp_data.cov_type;

    GPData gp_regional;
    gp_regional.n_obs = data.multiscale_gp_data.n_obs;
    gp_regional.nn = data.multiscale_gp_data.nn_regional;
    gp_regional.coords = data.multiscale_gp_data.coords;
    gp_regional.nn_idx = data.multiscale_gp_data.nn_idx_regional;
    gp_regional.nn_dist = data.multiscale_gp_data.nn_dist_regional;
    gp_regional.nn_order = data.multiscale_gp_data.nn_order_regional;
    gp_regional.nn_order_inv = data.multiscale_gp_data.nn_order_inv_regional;
    gp_regional.cov_type = data.multiscale_gp_data.cov_type;

    ratiod_gp::NNGPGradients nngp_grads_local, nngp_grads_regional;
    ratiod_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local, gp_local, nngp_grads_local);
    ratiod_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional, gp_regional, nngp_grads_regional);

    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_local_start + i] += nngp_grads_local.grad_w[i];
        grad[layout.gp_regional_start + i] += nngp_grads_regional.grad_w[i];
    }
    grad[layout.log_sigma2_gp_local_idx] += nngp_grads_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += nngp_grads_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += nngp_grads_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += nngp_grads_regional.grad_log_phi;

    // =========================================================================
    // Likelihood loop
    // =========================================================================
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    for (int i = 0; i < data.N; i++) {
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        for (int j = 0; j < data.p_denom; j++) eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        if (layout.has_re && data.re_group[i] > 0) { eta_num += re[data.re_group[i] - 1]; eta_denom += re[data.re_group[i] - 1]; }

        // Multi-scale GP spatial effect (map observation to unique location)
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        if (data.multiscale_gp_data.shared) { eta_num += ms_spatial; eta_denom += ms_spatial; }
        else { eta_num += ms_spatial; }

        int t_idx = -1;
        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            t_idx = data.temporal_time_idx[i] - 1;
            if (t_idx >= 0 && t_idx < T_len) {
                if (data.temporal_shared) { eta_num += phi_temporal[t_idx]; eta_denom += phi_temporal[t_idx]; }
                else { eta_num += phi_temporal[t_idx]; }
            }
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        double dLL_num = 0.0, dLL_denom = 0.0;
        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num;
            dLL_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] += dLL_num * data.X_num_flat[i * data.p_num + j];
        for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] += dLL_denom * data.X_denom_flat[i * data.p_denom + j];
        if (layout.has_re && data.re_group[i] > 0) grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;

        double dLL_dspatial = data.multiscale_gp_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        grad[layout.gp_local_start + loc_i] += dLL_dspatial;
        grad[layout.gp_regional_start + loc_i] += dLL_dspatial;

        if (t_idx >= 0 && t_idx < T_len) grad_temporal_lik[t_idx] += data.temporal_shared ? (dLL_num + dLL_denom) : dLL_num;
    }

    // =========================================================================
    // Temporal GMRF gradients
    // =========================================================================
    for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] = grad_temporal_lik[t];
    if (data.temporal_type == TemporalType::RW1) {
        double qf = 0.0;
        for (int t = 0; t < T_len; t++) {
            double g = 0.0;
            if (t > 0) { g += tau_temporal * (phi_temporal[t-1] - phi_temporal[t]); qf += std::pow(phi_temporal[t] - phi_temporal[t-1], 2); }
            if (t < T_len - 1) g += tau_temporal * (phi_temporal[t+1] - phi_temporal[t]);
            grad[layout.temporal_start + t] += g;
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 1) - 0.5 * tau_temporal * qf;
        { double sp = 0.0; for (int t = 0; t < T_len; t++) sp += phi_temporal[t];
          for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
    } else if (data.temporal_type == TemporalType::RW2) {
        double qf = 0.0;
        for (int t = 0; t < T_len; t++) {
            double g = 0.0;
            if (t >= 2) g -= tau_temporal * (phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t]);
            if (t >= 1 && t < T_len - 1) g += 2.0 * tau_temporal * (phi_temporal[t-1] - 2.0*phi_temporal[t] + phi_temporal[t+1]);
            if (t < T_len - 2) g -= tau_temporal * (phi_temporal[t] - 2.0*phi_temporal[t+1] + phi_temporal[t+2]);
            grad[layout.temporal_start + t] += g;
        }
        for (int t = 2; t < T_len; t++) qf += std::pow(phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t], 2);
        grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 2) - 0.5 * tau_temporal * qf;
        { double sp = 0.0; for (int t = 0; t < T_len; t++) sp += phi_temporal[t];
          for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
    } else if (data.temporal_type == TemporalType::AR1) {
        double omr2 = 1.0 - rho_ar1 * rho_ar1;
        grad[layout.temporal_start] += -tau_temporal * omr2 * phi_temporal[0];
        if (T_len > 1) grad[layout.temporal_start] += tau_temporal * rho_ar1 * (phi_temporal[1] - rho_ar1 * phi_temporal[0]);
        double qf = omr2 * phi_temporal[0] * phi_temporal[0];
        for (int t = 1; t < T_len; t++) {
            double r = phi_temporal[t] - rho_ar1 * phi_temporal[t-1]; qf += r * r;
            double g = -tau_temporal * r;
            if (t < T_len - 1) g += tau_temporal * rho_ar1 * (phi_temporal[t+1] - rho_ar1 * phi_temporal[t]);
            grad[layout.temporal_start + t] += g;
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * qf;
        if (layout.logit_rho_ar1_idx >= 0) {
            double gr = -rho_ar1 / omr2 + tau_temporal * rho_ar1 * phi_temporal[0] * phi_temporal[0];
            for (int t = 1; t < T_len; t++) gr += tau_temporal * (phi_temporal[t] - rho_ar1 * phi_temporal[t-1]) * phi_temporal[t-1];
            grad[layout.logit_rho_ar1_idx] += gr * rho_ar1 * (1.0 - rho_ar1);
        }
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// SVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from svc_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_svc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

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

    // SVC parameters
    int n_svc = data.svc_data.n_svc;
    int N_obs = data.svc_data.n_obs;

    std::vector<double> svc_sigma2(n_svc), svc_phi(n_svc);
    for (int j = 0; j < n_svc; j++) {
        svc_sigma2[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
        svc_phi[j] = std::exp(params[layout.log_phi_svc_start + j]);

        // Bounds check for phi
        if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
            return;  // Out of bounds - return zero gradient
        }
    }

    // Extract SVC spatial effects
    std::vector<double> svc_w_flat(N_obs * n_svc);
    for (int k = 0; k < N_obs * n_svc; k++) {
        svc_w_flat[k] = params[layout.svc_w_start + k];
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    // Fixed effects prior: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] -= tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] -= tau_beta * beta_denom[j];
    }

    // RE prior: Half-Cauchy on sigma, N(0, sigma_re^2) on effects
    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    // Overdispersion prior: Gamma via log transform, d/d(log_phi) = shape - rate*phi
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }

    // SVC hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        // Half-Cauchy on sigma, d/d(log_sigma2) = -ratio^2/(1+ratio^2) + 1
        double sigma = std::sqrt(svc_sigma2[j]);
        double scale = data.svc_sigma2_prior_scale;
        double ratio = sigma / scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma2_svc_start + j] = -ratio_sq / (1.0 + ratio_sq) + 1.0;

        // Uniform prior on phi - just Jacobian
        grad[layout.log_phi_svc_start + j] = 1.0;
    }

    // =========================================================================
    // Compute NNGP gradients w.r.t. SVC effects (analytical)
    // =========================================================================
    for (int j = 0; j < n_svc; j++) {
        // Extract w for this SVC term
        std::vector<double> w_j(N_obs);
        for (int i = 0; i < N_obs; i++) {
            w_j[i] = svc_w_flat[j * N_obs + i];
        }

        ratiod_svc::SVCGradients svc_grads;
        ratiod_svc::svc_nngp_gradients(w_j, svc_sigma2[j], svc_phi[j], data.svc_data, svc_grads);

        // Add NNGP prior gradient contributions for w
        for (int i = 0; i < N_obs; i++) {
            grad[layout.svc_w_start + j * N_obs + i] += svc_grads.grad_w[i];
        }

        // Add NNGP gradient contributions for SVC hyperparameters
        grad[layout.log_sigma2_svc_start + j] += svc_grads.grad_log_sigma2;
        grad[layout.log_phi_svc_start + j] += svc_grads.grad_log_phi;

        // Sum-to-zero gradient: d/dw_k [-0.5 * lambda_mean * N * mean(w)^2] = -lambda_mean * mean(w)
        double sum_w = 0.0;
        for (int i = 0; i < N_obs; i++) {
            sum_w += w_j[i];
        }
        double mean_w = sum_w / N_obs;
        double stz_grad = -1.0 * mean_w;  // lambda_mean = 1.0
        for (int i = 0; i < N_obs; i++) {
            grad[layout.svc_w_start + j * N_obs + i] += stz_grad;
        }
    }

    // =========================================================================
    // Data likelihood loop
    // =========================================================================
    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            eta_num += re[g];
            eta_denom += re[g];
        }

        // SVC effect: sum_j X_svc[i,j] * w_j[i]
        double svc_effect = 0.0;
        for (int j = 0; j < n_svc; j++) {
            double x_ij = data.svc_data.X_svc[i * n_svc + j];
            double w_ij = svc_w_flat[j * N_obs + i];
            svc_effect += x_ij * w_ij;
        }
        if (data.svc_data.shared) {
            eta_num += svc_effect;
            eta_denom += svc_effect;
        } else {
            eta_num += svc_effect;
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        // Likelihood gradients depend on model type
        double dLL_deta_num = 0.0;
        double dLL_deta_denom = 0.0;

        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {  // POISSON_GAMMA
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num;
            dLL_deta_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        // Accumulate gradients for fixed effects
        for (int j = 0; j < data.p_num; j++) {
            grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
        }

        // Gradients for RE
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
        }

        // Gradients for SVC spatial effects (from likelihood)
        double dLL_dsvc = data.svc_data.shared ?
                          (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        for (int j = 0; j < n_svc; j++) {
            double x_ij = data.svc_data.X_svc[i * n_svc + j];
            grad[layout.svc_w_start + j * N_obs + i] += dLL_dsvc * x_ij;
        }

        // Gradient w.r.t. phi_num (for NegBin)
        if (layout.has_phi_num && data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double y = data.y_num[i];
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
        }

        // Gradient w.r.t. phi_denom
        if (layout.has_phi_denom) {
            if (data.model_type == ModelType::NEGBIN_NEGBIN) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom[i];
                double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                                 + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                                 - (y + phi_denom) / (mu_denom + phi_denom);
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            } else if (data.model_type == ModelType::POISSON_GAMMA) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom_cont[i];
                double digamma_phi = ratiod::math::portable_digamma(phi_denom);
                double dLL_dphi = std::log(phi_denom) + 1.0 - digamma_phi
                                 + std::log(y) - std::log(mu_denom)
                                 - y / mu_denom;
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }

    // Debug output (first call only)
    static int svc_grad_debug_counter = 0;
    if (svc_grad_debug_counter == 0) {
        double sum_abs_grad_svc = 0.0;
        int n_svc_params = n_svc * N_obs;
        for (int k = 0; k < n_svc_params; k++) {
            sum_abs_grad_svc += std::abs(grad[layout.svc_w_start + k]);
        }
        Rcpp::Rcout << "[SVC TOTAL] sum|grad_svc_w|=" << sum_abs_grad_svc
                    << " (n_svc=" << n_svc << ", N_obs=" << N_obs << ")\n";
        svc_grad_debug_counter++;
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// TVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from hmc_tvc_grad.h for RW1/RW2/AR1 priors
// =====================================================================

void compute_gradient_tvc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

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

    // TVC parameters (use pre-allocated workspace buffers)
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;
    int n_w = n_groups * n_tvc * n_times;

    double* tvc_tau = data.tvc_data.tau_ws.data();
    double* tvc_rho = data.tvc_data.rho_ws.data();
    for (int j = 0; j < n_tvc; j++) {
        tvc_tau[j] = std::exp(params[layout.log_tau_tvc_start + j]);
        if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
            double logit_rho = params[layout.logit_rho_tvc_start + j];
            double u = 1.0 / (1.0 + std::exp(-logit_rho));
            tvc_rho[j] = 2.0 * u - 1.0;
        } else {
            tvc_rho[j] = 0.0;
        }
    }

    // Extract TVC w values into pre-allocated buffer
    double* tvc_w_flat = data.tvc_data.w_flat_ws.data();
    for (int k = 0; k < n_w; k++) {
        tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    // Fixed effects prior: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] -= tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] -= tau_beta * beta_denom[j];
    }

    // RE prior: Half-Cauchy on sigma, N(0, sigma_re^2) on effects
    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    // Overdispersion prior: Gamma via log transform, d/d(log_phi) = shape - rate*phi
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }

    // TVC hyperparameter priors: PC prior on tau (must match compute_log_post)
    // log_post = log(rate) - rate/sqrt(tau) - log(2*sigma) + log(tau)
    //          = const - rate * tau^{-1/2} + 1.5 * log(tau)
    // d/d(log_tau) = 0.5 * rate / sqrt(tau) + 1.5
    double tvc_pc_rate = -std::log(0.01) / 1.0;  // P(sigma > 1) = 0.01
    for (int j = 0; j < n_tvc; j++) {
        double sigma_j = 1.0 / std::sqrt(tvc_tau[j]);
        grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
    }

    // AR1: Uniform(-1,1) prior on rho (must match compute_log_post)
    // log_post Jacobian: log(u) + log(1-u) where u = (rho+1)/2
    // d/d(logit_u) [log(u) + log(1-u)] = (1-u) + (-u) = 1 - 2u
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            double u = (tvc_rho[j] + 1.0) / 2.0;  // u in (0,1)
            grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
        }
    }

    // =========================================================================
    // Compute TVC prior gradients using zero-allocation workspace version
    // =========================================================================
    ratiod_tvc::TVCGradientWS tvc_ws;
    tvc_ws.grad_w = data.tvc_data.grad_w_ws.data();
    tvc_ws.grad_log_tau = data.tvc_data.grad_log_tau_ws.data();
    tvc_ws.grad_logit_rho = data.tvc_data.grad_logit_rho_ws.data();
    tvc_ws.grad_w_jg = data.tvc_data.grad_w_jg_ws.data();
    tvc_ws.d_buf = data.tvc_data.d_ws.data();
    tvc_ws.n_w = n_w;
    tvc_ws.n_tvc = n_tvc;
    ratiod_tvc::tvc_prior_gradients_ws(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho, tvc_ws);

    // Add TVC prior gradient contributions
    for (int k = 0; k < n_w; k++) {
        grad[layout.tvc_w_start + k] += tvc_ws.grad_w[k];
    }
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] += tvc_ws.grad_log_tau[j];
    }
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.logit_rho_tvc_start + j] += tvc_ws.grad_logit_rho[j];
        }
    }

    // =========================================================================
    // Precompute TVC contribution to linear predictor (pre-allocated buffer)
    // =========================================================================
    double* tvc_eta = data.tvc_data.eta_ws.data();
    std::fill(tvc_eta, tvc_eta + data.N, 0.0);
    for (int i = 0; i < data.N; i++) {
        int t = data.tvc_data.time_index[i] - 1;  // 0-based
        int g = data.tvc_data.group_index[i] - 1;  // 0-based

        for (int j = 0; j < n_tvc; j++) {
            double x_ij = data.tvc_data.X_tvc[i * n_tvc + j];
            double w_jgt = tvc_w_flat[(g * n_tvc + j) * n_times + t];
            tvc_eta[i] += x_ij * w_jgt;
        }
    }

    // =========================================================================
    // Data likelihood loop
    // =========================================================================
    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            eta_num += re[g];
            eta_denom += re[g];
        }

        // TVC effect
        double tvc_effect = tvc_eta[i];
        if (data.tvc_data.shared) {
            eta_num += tvc_effect;
            eta_denom += tvc_effect;
        } else {
            eta_num += tvc_effect;
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        // Likelihood gradients depend on model type
        double dLL_deta_num = 0.0;
        double dLL_deta_denom = 0.0;

        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {  // POISSON_GAMMA
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num;
            dLL_deta_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        // Accumulate gradients for fixed effects
        for (int j = 0; j < data.p_num; j++) {
            grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
        }

        // Gradients for RE
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
        }

        // Gradients for TVC w values (from likelihood)
        double dLL_dtvc = data.tvc_data.shared ?
                          (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            double x_ij = data.tvc_data.X_tvc[i * n_tvc + j];
            int w_idx = (g * n_tvc + j) * n_times + t;
            grad[layout.tvc_w_start + w_idx] += dLL_dtvc * x_ij;
        }

        // Gradient w.r.t. phi_num (for NegBin)
        if (layout.has_phi_num && data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double y = data.y_num[i];
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
        }

        // Gradient w.r.t. phi_denom
        if (layout.has_phi_denom) {
            if (data.model_type == ModelType::NEGBIN_NEGBIN) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom[i];
                double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                                 + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                                 - (y + phi_denom) / (mu_denom + phi_denom);
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            } else if (data.model_type == ModelType::POISSON_GAMMA) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom_cont[i];
                double digamma_phi = ratiod::math::portable_digamma(phi_denom);
                double dLL_dphi = std::log(phi_denom) + 1.0 - digamma_phi
                                 + std::log(y) - std::log(mu_denom)
                                 - y / mu_denom;
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }

    // Debug output (first call only)
    static int tvc_grad_debug_counter = 0;
    if (tvc_grad_debug_counter == 0) {
        double sum_abs_grad_tvc = 0.0;
        for (int k = 0; k < n_w; k++) {
            sum_abs_grad_tvc += std::abs(grad[layout.tvc_w_start + k]);
        }
        Rcpp::Rcout << "[TVC TOTAL] sum|grad_tvc_w|=" << sum_abs_grad_tvc
                    << " (n_tvc=" << n_tvc << ", n_times=" << n_times
                    << ", n_groups=" << n_groups << ")\n";
        tvc_grad_debug_counter++;
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Latent factor gradient (hand-coded, O(N*K))
// Uses analytical gradients for latent factor models
// =====================================================================

void compute_gradient_latent_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

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

    // Latent factor parameters
    int K = data.latent_n_factors;
    int N = data.N;

    // Extract log_sigma for latent factors
    std::vector<double> log_sigma_latent(K);
    std::vector<double> sigma_latent(K);
    for (int k = 0; k < K; k++) {
        log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        sigma_latent[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factors (unconstrained)
    int n_factor_params = N * K;
    std::vector<double> factors_raw(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
        factors_raw[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint to get constrained factors
    std::vector<double> factors_constrained = factors_raw;
    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += factors_constrained[i * K + k];
            }
            double mean = sum / N;
            for (int i = 0; i < N; i++) {
                factors_constrained[i * K + k] -= mean;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            double first_val = factors_constrained[k];  // factors[0, k]
            for (int i = 0; i < N; i++) {
                factors_constrained[i * K + k] -= first_val;
            }
        }
    }

    // Precompute latent contribution to eta
    std::vector<double> latent_eta(N, 0.0);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            latent_eta[i] += factors_constrained[i * K + k] * sigma_latent[k];
        }
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    // Fixed effects prior: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] -= tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] -= tau_beta * beta_denom[j];
    }

    // RE prior: Half-Cauchy on sigma, N(0, sigma_re^2) on effects
    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    // Overdispersion prior: Gamma(shape, rate) on phi, with log transform
    // log p(log_phi) = (shape-1)*log_phi - rate*phi + log_phi  [includes Jacobian]
    // d/d(log_phi) = (shape-1) + 1 - rate*phi = shape - rate*phi
    // Note: NO additional phi factor since the prior is written w.r.t. log_phi directly
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_denom;
    }

    // Latent sigma prior: Exponential(rate) on sigma, with Jacobian for log transform
    // log p(log_sigma) = log(rate) + log_sigma - rate * sigma
    // d/d(log_sigma) = 1 - rate * sigma
    double latent_rate = data.latent_sigma_prior_rate;
    for (int k = 0; k < K; k++) {
        grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent[k];
    }

    // Latent factor prior: N(0, 1) on constrained factors
    // The autodiff applies the prior to constrained factors, then chain-rules to raw factors.
    // Direct computation: d(-0.5*f_constrained^2)/d(f_raw) = -f_constrained * d(f_constrained)/d(f_raw)
    // For SUM_TO_ZERO: d(fc[i])/d(fr[j]) = delta_ij - 1/N, so gradient = -fc[i] + mean(fc) = -fc[i]
    // For FIRST_ZERO: d(fc[i])/d(fr[j]) = delta_ij - delta_j0, similar result
    // In both cases, the prior gradient w.r.t. raw factors equals -constrained_factor
    // But we need to apply the chain rule properly below, so here we just store the
    // gradient w.r.t. constrained factors.

    // Note: The prior is conceptually on the constrained factors (which sum to zero),
    // and the gradient flows back through the constraint transformation.
    // We handle this by computing prior gradient on constrained factors,
    // then adding it to grad_factors_constrained, which gets chain-ruled below.

    // =========================================================================
    // Likelihood loop - compute dLL/deta and chain-rule to all parameters
    // =========================================================================

    // Gradients to accumulate for latent factors (on constrained factors first)
    std::vector<double> grad_factors_constrained(n_factor_params, 0.0);

    for (int i = 0; i < N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            eta_num += re[g];
            eta_denom += re[g];
        }

        // Latent effect
        double latent_effect = latent_eta[i];
        if (data.latent_shared) {
            eta_num += latent_effect;
            eta_denom += latent_effect;
        } else {
            eta_num += latent_effect;
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        // Likelihood gradients depend on model type
        double dLL_deta_num = 0.0;
        double dLL_deta_denom = 0.0;

        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {  // POISSON_GAMMA
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num;
            // For POISSON_GAMMA, phi_num is the shape parameter for gamma denom
            dLL_deta_denom = phi_num * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        // Total gradient through latent effect
        double dLL_dlatent = data.latent_shared ?
                             (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;

        // Accumulate gradients for fixed effects
        for (int j = 0; j < data.p_num; j++) {
            grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
        }

        // Gradients for RE
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
        }

        // Gradients for latent factors (on constrained space)
        // eta_latent[i] = sum_k factor[i,k] * sigma[k]
        // d(LL)/d(factor[i,k]) = dLL_dlatent * sigma[k]
        // d(LL)/d(log_sigma[k]) += dLL_dlatent * factor[i,k] * sigma[k]
        for (int k = 0; k < K; k++) {
            grad_factors_constrained[i * K + k] = dLL_dlatent * sigma_latent[k];
            grad[layout.log_sigma_latent_start + k] += dLL_dlatent * factors_constrained[i * K + k] * sigma_latent[k];
        }

        // Gradient w.r.t. phi_num
        if (layout.has_phi_num) {
            if (data.model_type == ModelType::NEGBIN_NEGBIN) {
                double mu_num = std::exp(eta_num);
                double y = data.y_num[i];
                double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                                 + std::log(phi_num / (mu_num + phi_num)) + 1.0
                                 - (y + phi_num) / (mu_num + phi_num);
                grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
            } else if (data.model_type == ModelType::POISSON_GAMMA) {
                // For POISSON_GAMMA, phi_num is the gamma shape parameter for denominator
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom_cont[i];
                double digamma_phi = ratiod::math::portable_digamma(phi_num);
                double dLL_dphi = std::log(phi_num) + 1.0 - digamma_phi
                                 + std::log(y) - std::log(mu_denom)
                                 - y / mu_denom;
                grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
            }
        }

        // Gradient w.r.t. phi_denom (NEGBIN_NEGBIN only)
        if (layout.has_phi_denom && data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_denom = std::exp(eta_denom);
            double y = data.y_denom[i];
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                             + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                             - (y + phi_denom) / (mu_denom + phi_denom);
            grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
        }
    }

    // =========================================================================
    // Add prior gradient to grad_factors_constrained
    // =========================================================================
    // Prior: N(0, 1) on constrained factors, log p = -0.5 * f_constrained^2
    // Gradient: d(-0.5 * f^2)/d(f) = -f
    // For FIRST_ZERO: constrained factor 0 is always 0, so skip it
    int prior_start = (data.latent_constraint == 0) ? 0 : 1;
    for (int k = 0; k < K; k++) {
        for (int i = prior_start; i < N; i++) {
            grad_factors_constrained[i * K + k] += -factors_constrained[i * K + k];
        }
    }

    // =========================================================================
    // Apply constraint chain-rule to get gradients on raw (unconstrained) factors
    // =========================================================================

    // For sum-to-zero: d(LL)/d(factor_raw[j,k]) = d(LL)/d(factor_constrained[j,k])
    //                                           - (1/N) * sum_i d(LL)/d(factor_constrained[i,k])
    // For first-zero: d(LL)/d(factor_raw[0,k]) = -sum_{i>0} d(LL)/d(factor_constrained[i,k])
    //                 d(LL)/d(factor_raw[j,k]) = d(LL)/d(factor_constrained[j,k]) for j > 0

    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            // Compute mean gradient for this factor
            double sum_grad = 0.0;
            for (int i = 0; i < N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
            }
            double mean_grad = sum_grad / N;

            // Adjust each gradient
            for (int i = 0; i < N; i++) {
                grad[layout.latent_factor_start + i * K + k] +=
                    grad_factors_constrained[i * K + k] - mean_grad;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            // Gradient for factor_raw[0,k] = -sum of gradients for i > 0
            double sum_grad = 0.0;
            for (int i = 1; i < N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
                // Gradient for factor_raw[i,k] = gradient of constrained[i,k] for i > 0
                grad[layout.latent_factor_start + i * K + k] += grad_factors_constrained[i * K + k];
            }
            grad[layout.latent_factor_start + k] += -sum_grad;  // factor_raw[0,k]
        }
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// GP gradient via autodiff (O(N*nn^3) - much faster than numerical O(N^2))
// Uses templated NNGP likelihood from hmc_gp_autodiff.h
// =====================================================================

void compute_gradient_gp_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace ratiod::ad;
    using namespace ratiod::math;

    // Thread-safe: each call gets its own tape via RAII
    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Create autodiff variables from all parameters
    std::vector<Var> params_ad = make_vars(tape, params);

    Var log_post(tape, 0.0);

    // =========================================================================
    // Fixed effects priors: N(0, sigma_beta^2)
    // =========================================================================
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        Var beta = params_ad[layout.beta_num_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }
    for (int j = 0; j < data.p_denom; j++) {
        Var beta = params_ad[layout.beta_denom_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }

    // =========================================================================
    // Random effects priors (if present)
    // =========================================================================
    Var sigma_re(tape, 1.0);
    if (layout.has_re && data.n_re_groups > 0) {
        Var log_sigma_re = params_ad[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        // Half-Cauchy prior on sigma_re
        Var ratio = sigma_re / data.sigma_re_scale;
        log_post = log_post - safe_log(1.0 + ratio * ratio);
        log_post = log_post + log_sigma_re;  // Jacobian

        // N(0, sigma_re^2) prior on RE
        Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            Var re_g = params_ad[layout.re_start + g];
            log_post = log_post - 0.5 * tau_re * re_g * re_g;
            log_post = log_post + 0.5 * safe_log(tau_re);
        }
    }

    // =========================================================================
    // Overdispersion priors (Gamma)
    // =========================================================================
    Var phi_num(tape, 1.0);
    Var phi_denom(tape, 1.0);

    if (layout.has_phi_num) {
        Var log_phi = params_ad[layout.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
        // Gamma(shape, rate) prior on phi
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_num
                            + log_phi;  // Jacobian
    }
    if (layout.has_phi_denom) {
        Var log_phi = params_ad[layout.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_denom
                            + log_phi;
    }

    // =========================================================================
    // GP priors and NNGP likelihood
    // =========================================================================
    std::vector<Var> gp_w_ad;
    Var sigma2_gp(tape, 1.0);
    Var phi_gp(tape, 0.1);

    if (layout.is_gp && data.has_gp) {
        Var log_sigma2_gp = params_ad[layout.log_sigma2_gp_idx];
        Var log_phi_gp = params_ad[layout.log_phi_gp_idx];
        sigma2_gp = safe_exp(log_sigma2_gp);
        phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 (penalizes large variance)
        log_post = log_post + ratiod_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds
        log_post = log_post + ratiod_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects
        int N_gp = data.gp_data.n_obs;
        gp_w_ad.resize(N_gp);
        for (int i = 0; i < N_gp; i++) {
            gp_w_ad[i] = params_ad[layout.gp_w_start + i];
        }

        // NNGP log-likelihood using templated function
        Var gp_ll = ratiod_gp::gp_nngp_log_lik_t(gp_w_ad, sigma2_gp, phi_gp, data.gp_data);
        log_post = log_post + gp_ll;
    }

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.p_num);
    std::vector<Var> beta_denom_ad(data.p_denom);
    for (int j = 0; j < data.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.beta_num_start + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.beta_denom_start + j];
    }

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.p_num; j++) {
            eta_num = eta_num + data.X_num_flat[i * data.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom = eta_denom + data.X_denom_flat[i * data.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            eta_num = eta_num + re_g;
            eta_denom = eta_denom + re_g;
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && data.has_gp && !gp_w_ad.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            Var gp_effect = gp_w_ad[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = ratiod::math::log_lik_binomial(data.y_num[i], data.y_denom[i], p);
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = ratiod::math::log_lik_negbin(data.y_num[i], mu_num, phi_num) +
                   ratiod::math::log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = ratiod::math::log_lik_poisson(data.y_num[i], mu_num) +
                   ratiod::math::log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// Multi-scale GP gradient (hand-coded, ~2-3x faster than autodiff)
// Uses analytical gradients for w and numerical for sigma2/phi
// =====================================================================

void compute_gradient_msgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // =========================================================================
    // Extract parameters
    // =========================================================================
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

    // Multi-scale GP parameters
    int N_gp = data.multiscale_gp_data.n_obs;

    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_phi_local = params[layout.log_phi_gp_local_idx];
    double sigma2_local = std::exp(log_sigma2_local);
    double phi_local = std::exp(log_phi_local);

    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_phi_regional = params[layout.log_phi_gp_regional_idx];
    double sigma2_regional = std::exp(log_sigma2_regional);
    double phi_regional = std::exp(log_phi_regional);

    // Extract spatial effects
    std::vector<double> w_local(N_gp), w_regional(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local[i] = params[layout.gp_local_start + i];
        w_regional[i] = params[layout.gp_regional_start + i];
    }

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return; // Out of bounds - return zero gradient
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    // Fixed effects prior: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] -= tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] -= tau_beta * beta_denom[j];
    }

    // RE prior: Half-Cauchy on sigma, N(0, sigma_re^2) on effects
    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    // Overdispersion prior: Gamma via log transform, d/d(log_phi) = shape - rate*phi
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }

    // PC priors on GP variances
    // sigma2 ~ Exp(rate) where rate = -log(alpha)/U (via sigma = sqrt(sigma2))
    // log p(sigma2) = -rate * sqrt(sigma2) + const - log(sigma2)/2
    // d/d(log_sigma2) = d/d(sigma2) * sigma2 = -rate * sqrt(sigma2)/2 + 0.5
    double rate_sigma_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
    double sigma_local = std::sqrt(sigma2_local);
    grad[layout.log_sigma2_gp_local_idx] = -0.5 * rate_sigma_local * sigma_local + 0.5;

    double rate_sigma_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
    double sigma_regional = std::sqrt(sigma2_regional);
    grad[layout.log_sigma2_gp_regional_idx] = -0.5 * rate_sigma_regional * sigma_regional + 0.5;

    // Jacobians for log-transforms
    grad[layout.log_phi_gp_local_idx] = 1.0;    // Uniform prior, just Jacobian
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // =========================================================================
    // Compute NNGP gradients w.r.t. spatial effects (analytical)
    // =========================================================================
    GPData gp_local;
    gp_local.n_obs = data.multiscale_gp_data.n_obs;
    gp_local.nn = data.multiscale_gp_data.nn_local;
    gp_local.coords = data.multiscale_gp_data.coords;
    gp_local.nn_idx = data.multiscale_gp_data.nn_idx_local;
    gp_local.nn_dist = data.multiscale_gp_data.nn_dist_local;
    gp_local.nn_order = data.multiscale_gp_data.nn_order_local;
    gp_local.nn_order_inv = data.multiscale_gp_data.nn_order_inv_local;
    gp_local.cov_type = data.multiscale_gp_data.cov_type;

    GPData gp_regional;
    gp_regional.n_obs = data.multiscale_gp_data.n_obs;
    gp_regional.nn = data.multiscale_gp_data.nn_regional;
    gp_regional.coords = data.multiscale_gp_data.coords;
    gp_regional.nn_idx = data.multiscale_gp_data.nn_idx_regional;
    gp_regional.nn_dist = data.multiscale_gp_data.nn_dist_regional;
    gp_regional.nn_order = data.multiscale_gp_data.nn_order_regional;
    gp_regional.nn_order_inv = data.multiscale_gp_data.nn_order_inv_regional;
    gp_regional.cov_type = data.multiscale_gp_data.cov_type;

    // Get NNGP gradients (analytical for w, numerical for sigma2/phi)
    ratiod_gp::NNGPGradients nngp_grads_local, nngp_grads_regional;
    ratiod_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local, gp_local, nngp_grads_local);
    ratiod_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional, gp_regional, nngp_grads_regional);

    // Add NNGP prior gradient contributions for w
    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_local_start + i] += nngp_grads_local.grad_w[i];
        grad[layout.gp_regional_start + i] += nngp_grads_regional.grad_w[i];
    }

    // Add NNGP gradient contributions for GP hyperparameters
    grad[layout.log_sigma2_gp_local_idx] += nngp_grads_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += nngp_grads_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += nngp_grads_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += nngp_grads_regional.grad_log_phi;

    // =========================================================================
    // Data likelihood loop
    // =========================================================================
    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            eta_num += re[g];
            eta_denom += re[g];
        }

        // Multi-scale GP spatial effect (map observation to unique location)
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num += ms_spatial;
            eta_denom += ms_spatial;
        } else {
            eta_num += ms_spatial;
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        // Likelihood gradients depend on model type
        double dLL_deta_num = 0.0;
        double dLL_deta_denom = 0.0;

        if (data.model_type == ModelType::BINOMIAL) {
            // Binomial: d(log_lik)/d(eta) = y - n*p where p = logit^{-1}(eta)
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
            // denom not used in binomial (y_denom is trials)
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            // NegBin: d(log_lik)/d(eta) = y - mu*(y+phi)/(mu+phi)
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {  // POISSON_GAMMA
            // Poisson: d(log_lik)/d(eta) = y - mu
            double mu_num = std::exp(eta_num);
            double mu_denom = std::exp(eta_denom);
            dLL_deta_num = data.y_num[i] - mu_num;
            // Gamma: d(log_lik)/d(eta) = phi * (y/mu - 1)
            dLL_deta_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        // Accumulate gradients for fixed effects
        for (int j = 0; j < data.p_num; j++) {
            grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
        }

        // Gradients for RE
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
        }

        // Gradients for GP spatial effects (from likelihood, mapped to unique location)
        double dLL_dspatial = data.multiscale_gp_data.shared ?
                              (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        grad[layout.gp_local_start + loc_i] += dLL_dspatial;
        grad[layout.gp_regional_start + loc_i] += dLL_dspatial;

        // Gradient w.r.t. phi_num (for NegBin)
        if (layout.has_phi_num && data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num);
            double y = data.y_num[i];
            // d(log_lik)/d(phi) = digamma(y+phi) - digamma(phi) + log(phi/(mu+phi)) + 1 - (y+phi)/(mu+phi)
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
        }

        // Gradient w.r.t. phi_denom
        if (layout.has_phi_denom) {
            if (data.model_type == ModelType::NEGBIN_NEGBIN) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom[i];
                double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                                 + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                                 - (y + phi_denom) / (mu_denom + phi_denom);
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            } else if (data.model_type == ModelType::POISSON_GAMMA) {
                double mu_denom = std::exp(eta_denom);
                double y = data.y_denom_cont[i];
                double digamma_phi = ratiod::math::portable_digamma(phi_denom);
                double dLL_dphi = std::log(phi_denom) + 1.0 - digamma_phi
                                 + std::log(y) - std::log(mu_denom)
                                 - y / mu_denom;
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Multi-scale GP gradient (autodiff, ~3x faster than numerical)
// =====================================================================

void compute_gradient_msgp_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace ratiod::ad;
    using namespace ratiod::math;

    // Thread-safe: each call gets its own tape via RAII
    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Create autodiff variables from all parameters
    std::vector<Var> params_ad = make_vars(tape, params);

    Var log_post(tape, 0.0);

    // =========================================================================
    // Fixed effects priors: N(0, sigma_beta^2)
    // =========================================================================
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        Var beta = params_ad[layout.beta_num_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }
    for (int j = 0; j < data.p_denom; j++) {
        Var beta = params_ad[layout.beta_denom_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }

    // =========================================================================
    // Random effects priors (if present)
    // =========================================================================
    Var sigma_re(tape, 1.0);
    if (layout.has_re && data.n_re_groups > 0) {
        Var log_sigma_re = params_ad[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        // Half-Cauchy prior on sigma_re
        Var ratio = sigma_re / data.sigma_re_scale;
        log_post = log_post - safe_log(1.0 + ratio * ratio);
        log_post = log_post + log_sigma_re;  // Jacobian

        // N(0, sigma_re^2) prior on RE
        Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            Var re_g = params_ad[layout.re_start + g];
            log_post = log_post - 0.5 * tau_re * re_g * re_g;
            log_post = log_post + 0.5 * safe_log(tau_re);
        }
    }

    // =========================================================================
    // Overdispersion priors (Gamma)
    // =========================================================================
    Var phi_num(tape, 1.0);
    Var phi_denom(tape, 1.0);

    if (layout.has_phi_num) {
        Var log_phi = params_ad[layout.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_num
                            + log_phi;  // Jacobian
    }
    if (layout.has_phi_denom) {
        Var log_phi = params_ad[layout.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_denom
                            + log_phi;
    }

    // =========================================================================
    // Multi-scale GP priors and NNGP likelihoods
    // =========================================================================
    int N_gp = data.multiscale_gp_data.n_obs;

    // Local scale parameters
    Var log_sigma2_local = params_ad[layout.log_sigma2_gp_local_idx];
    Var log_phi_local = params_ad[layout.log_phi_gp_local_idx];
    Var sigma2_local = safe_exp(log_sigma2_local);
    Var phi_local = safe_exp(log_phi_local);

    // Regional scale parameters
    Var log_sigma2_regional = params_ad[layout.log_sigma2_gp_regional_idx];
    Var log_phi_regional = params_ad[layout.log_phi_gp_regional_idx];
    Var sigma2_regional = safe_exp(log_sigma2_regional);
    Var phi_regional = safe_exp(log_phi_regional);

    // PC priors on variances
    log_post = log_post + ratiod_gp::log_prior_sigma2_pc_t(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    log_post = log_post + log_sigma2_local;  // Jacobian

    log_post = log_post + ratiod_gp::log_prior_sigma2_pc_t(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
    log_post = log_post + log_sigma2_regional;  // Jacobian

    // Range priors (uniform within bounds) - check bounds, return -inf if violated
    double phi_local_val = get_value(phi_local);
    double phi_regional_val = get_value(phi_regional);
    if (phi_local_val < data.multiscale_gp_data.range_local_lower ||
        phi_local_val > data.multiscale_gp_data.range_local_upper ||
        phi_regional_val < data.multiscale_gp_data.range_regional_lower ||
        phi_regional_val > data.multiscale_gp_data.range_regional_upper) {
        // Out of bounds - return zero gradients (log_post = -inf)
        // TapeScope destructor handles cleanup
        return;
    }
    log_post = log_post + log_phi_local;    // Jacobian
    log_post = log_post + log_phi_regional; // Jacobian

    // Extract GP spatial effects
    std::vector<Var> w_local_ad(N_gp);
    std::vector<Var> w_regional_ad(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local_ad[i] = params_ad[layout.gp_local_start + i];
        w_regional_ad[i] = params_ad[layout.gp_regional_start + i];
    }

    // NNGP log-likelihood for each scale using templated function
    Var msgp_ll = ratiod_gp::multiscale_gp_log_lik_t(
        w_local_ad, w_regional_ad,
        sigma2_local, phi_local,
        sigma2_regional, phi_regional,
        data.multiscale_gp_data);
    log_post = log_post + msgp_ll;

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.p_num);
    std::vector<Var> beta_denom_ad(data.p_denom);
    for (int j = 0; j < data.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.beta_num_start + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.beta_denom_start + j];
    }

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.p_num; j++) {
            eta_num = eta_num + data.X_num_flat[i * data.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom = eta_denom + data.X_denom_flat[i * data.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            eta_num = eta_num + re_g;
            eta_denom = eta_denom + re_g;
        }

        // Add multi-scale GP spatial effect (map observation to unique location)
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        Var ms_spatial = w_local_ad[loc_i] + w_regional_ad[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num = eta_num + ms_spatial;
            eta_denom = eta_denom + ms_spatial;
        } else {
            eta_num = eta_num + ms_spatial;
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = ratiod::math::log_lik_binomial(data.y_num[i], data.y_denom[i], p);
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = ratiod::math::log_lik_negbin(data.y_num[i], mu_num, phi_num) +
                   ratiod::math::log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = ratiod::math::log_lik_poisson(data.y_num[i], mu_num) +
                   ratiod::math::log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// GP + Temporal gradient (autodiff, combines GP and temporal effects)
// =====================================================================

void compute_gradient_gp_temporal_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace ratiod::ad;
    using namespace ratiod::math;

    // Thread-safe: each call gets its own tape via RAII
    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Create autodiff variables from all parameters
    std::vector<Var> params_ad = make_vars(tape, params);

    Var log_post(tape, 0.0);

    // =========================================================================
    // Fixed effects priors: N(0, sigma_beta^2)
    // =========================================================================
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        Var beta = params_ad[layout.beta_num_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }
    for (int j = 0; j < data.p_denom; j++) {
        Var beta = params_ad[layout.beta_denom_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }

    // =========================================================================
    // Random effects priors (if present)
    // =========================================================================
    Var sigma_re(tape, 1.0);
    if (layout.has_re && data.n_re_groups > 0) {
        Var log_sigma_re = params_ad[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        // Half-Cauchy prior on sigma_re
        Var ratio = sigma_re / data.sigma_re_scale;
        log_post = log_post - safe_log(1.0 + ratio * ratio);
        log_post = log_post + log_sigma_re;  // Jacobian

        // N(0, sigma_re^2) prior on RE
        Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            Var re_g = params_ad[layout.re_start + g];
            log_post = log_post - 0.5 * tau_re * re_g * re_g;
            log_post = log_post + 0.5 * safe_log(tau_re);
        }
    }

    // =========================================================================
    // Overdispersion priors (Gamma)
    // =========================================================================
    Var phi_num(tape, 1.0);
    Var phi_denom(tape, 1.0);

    if (layout.has_phi_num) {
        Var log_phi = params_ad[layout.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_num
                            + log_phi;  // Jacobian
    }
    if (layout.has_phi_denom) {
        Var log_phi = params_ad[layout.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_denom
                            + log_phi;
    }

    // =========================================================================
    // Temporal priors
    // =========================================================================
    Var tau_temporal(tape, 1.0);
    Var rho_ar1(tape, 0.0);
    std::vector<std::vector<Var>> phi_temporal_ad;  // [group][time]

    if (layout.has_temporal) {
        Var log_tau_temporal = params_ad[layout.log_tau_temporal_idx];
        tau_temporal = safe_exp(log_tau_temporal);

        // tau ~ Gamma(shape, rate) with Jacobian
        log_post = log_post + (data.tau_temporal_shape - 1.0) * log_tau_temporal
                            - data.tau_temporal_rate * tau_temporal
                            + log_tau_temporal;

        // AR1: also estimate rho
        if (layout.is_ar1) {
            Var logit_rho = params_ad[layout.logit_rho_ar1_idx];
            rho_ar1 = 1.0 / (1.0 + safe_exp(-logit_rho));

            // rho ~ Uniform(0,1) prior with logit Jacobian
            log_post = log_post + safe_log(rho_ar1) + safe_log(1.0 - rho_ar1);
        }

        // Extract temporal effects for each group
        int T = data.n_times;
        phi_temporal_ad.resize(data.n_temporal_groups);
        for (int g = 0; g < data.n_temporal_groups; g++) {
            phi_temporal_ad[g].resize(T);
            for (int t = 0; t < T; t++) {
                phi_temporal_ad[g][t] = params_ad[layout.temporal_start + g * T + t];
            }

            // Temporal prior
            Var temporal_prior = ratiod_temporal::temporal_log_prior_t(
                phi_temporal_ad[g], data.temporal_type, tau_temporal, rho_ar1, data.temporal_cyclic);
            log_post = log_post + temporal_prior;
        }
    }

    // =========================================================================
    // GP priors and NNGP likelihood
    // =========================================================================
    std::vector<Var> gp_w_ad;
    Var sigma2_gp(tape, 1.0);
    Var phi_gp(tape, 0.1);

    if (layout.is_gp && data.has_gp) {
        Var log_sigma2_gp = params_ad[layout.log_sigma2_gp_idx];
        Var log_phi_gp = params_ad[layout.log_phi_gp_idx];
        sigma2_gp = safe_exp(log_sigma2_gp);
        phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 (penalizes large variance)
        log_post = log_post + ratiod_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds
        log_post = log_post + ratiod_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects
        int N_gp = data.gp_data.n_obs;
        gp_w_ad.resize(N_gp);
        for (int i = 0; i < N_gp; i++) {
            gp_w_ad[i] = params_ad[layout.gp_w_start + i];
        }

        // NNGP log-likelihood using templated function
        Var gp_ll = ratiod_gp::gp_nngp_log_lik_t(gp_w_ad, sigma2_gp, phi_gp, data.gp_data);
        log_post = log_post + gp_ll;
    }

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.p_num);
    std::vector<Var> beta_denom_ad(data.p_denom);
    for (int j = 0; j < data.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.beta_num_start + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.beta_denom_start + j];
    }

    int T = data.n_times;

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.p_num; j++) {
            eta_num = eta_num + data.X_num_flat[i * data.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom = eta_denom + data.X_denom_flat[i * data.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            eta_num = eta_num + re_g;
            eta_denom = eta_denom + re_g;
        }

        // Add temporal effect
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;  // 0-based
            int g = data.temporal_group_idx[i] - 1; // 0-based
            if (g >= 0 && g < (int)phi_temporal_ad.size() &&
                t >= 0 && t < (int)phi_temporal_ad[g].size()) {
                Var temporal_effect = phi_temporal_ad[g][t];
                if (data.temporal_shared) {
                    eta_num = eta_num + temporal_effect;
                    eta_denom = eta_denom + temporal_effect;
                } else {
                    eta_num = eta_num + temporal_effect;
                }
            }
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && data.has_gp && !gp_w_ad.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            Var gp_effect = gp_w_ad[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = ratiod::math::log_lik_binomial(data.y_num[i], data.y_denom[i], p);
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = ratiod::math::log_lik_negbin(data.y_num[i], mu_num, phi_num) +
                   ratiod::math::log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = ratiod::math::log_lik_poisson(data.y_num[i], mu_num) +
                   ratiod::math::log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// HSGP gradient (O(N*M^2) - analytical, ~50x faster than numerical)
// =====================================================================

void compute_gradient_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Extract base parameters
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

  // HSGP parameters
  double log_sigma2 = params[layout.log_sigma2_hsgp_idx];
  double log_lengthscale = params[layout.log_lengthscale_hsgp_idx];
  double sigma2_hsgp = std::exp(log_sigma2);
  double lengthscale_hsgp = std::exp(log_lengthscale);

  int m_total = data.hsgp_data.m_total;
  std::vector<double> hsgp_beta(m_total);
  for (int j = 0; j < m_total; j++) {
    hsgp_beta[j] = params[layout.hsgp_beta_start + j];
  }

  // Evaluate HSGP spatial effect
  std::vector<double> hsgp_f;
  ratiod_hsgp::hsgp_evaluate(hsgp_beta, sigma2_hsgp, lengthscale_hsgp,
                              data.hsgp_data, hsgp_f);

  // Temporal parameters (for HSGP + temporal combinations)
  double tau_temporal = 0.0;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  double rho_ar1 = 0.5;
  if (layout.has_temporal) {
    tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    T_len = layout.temporal_end - layout.temporal_start;
    phi_temporal = &params[layout.temporal_start];
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
    }
  }

  // Compute likelihood and accumulate grad_f (gradient of log-lik w.r.t. f_i)
  std::vector<double> grad_f(data.N, 0.0);
  std::vector<double> grad_temporal_lik(T_len, 0.0);

  // --- Prior gradients ---

  // Fixed effects prior: N(0, sigma_beta^2)
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.p_num; j++) {
    grad[layout.beta_num_start + j] -= tau_beta * beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    grad[layout.beta_denom_start + j] -= tau_beta * beta_denom[j];
  }

  // RE prior: Half-Cauchy on sigma, N(0, sigma_re^2) on effects
  if (layout.has_re && data.n_re_groups > 0) {
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    for (int g = 0; g < data.n_re_groups; g++) {
      grad[layout.re_start + g] = -tau_re * re[g];
      grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
    }
  }

  // Overdispersion prior: Gamma via log transform, d/d(log_phi) = shape - rate*phi
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] = data.phi_prior_shape
                                  - data.phi_prior_rate * phi_num;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] = data.phi_prior_shape
                                    - data.phi_prior_rate * phi_denom;
  }

  // HSGP prior gradients (will be added to by hsgp_compute_gradients)
  // Initialize with prior contributions for sigma2 and lengthscale
  double sigma = std::sqrt(sigma2_hsgp);
  double rate_sigma = 4.6;
  // d/d(log_sigma2) of [log(rate) - rate*sigma - log(2*sigma) + 0.5*log_sigma2]
  // = -rate * sigma * 0.5 - 1/(2*sigma) * sigma * 0.5 + 0.5
  // = -0.5*rate*sigma - 0.25 + 0.5 = 0.25 - 0.5*rate*sigma
  // Simpler: just use the chain rule more carefully
  // log p = -rate*sigma + const in sigma
  // d/d(log_sigma2) = d/d(sigma) * d(sigma)/d(log_sigma2) = -rate * 0.5*sigma = -0.5*rate*sigma
  // Plus Jacobian contribution: 0.5
  grad[layout.log_sigma2_hsgp_idx] = -0.5 * rate_sigma * sigma + 0.5 - 0.5;  // -0.5 from log(sigma) Jacobian

  // LogNormal(0,1) on lengthscale: log p = -0.5*log_ell^2 - log_ell
  // d/d(log_ell) = -log_ell - 1 + 1 (Jacobian) = -log_ell
  grad[layout.log_lengthscale_hsgp_idx] = -log_lengthscale;

  // N(0, I) prior on beta: d/d(beta_j) = -beta_j
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] = -hsgp_beta[j];
  }

  // Temporal prior on tau (Gamma) and rho (Beta)
  if (layout.has_temporal) {
    grad[layout.log_tau_temporal_idx] = (data.tau_temporal_shape - 1.0) - data.tau_temporal_rate * tau_temporal + 1.0;
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }
  }

  // --- Likelihood loop ---
  for (int i = 0; i < data.N; i++) {
    // Linear predictors
    double eta_num = 0.0, eta_denom = 0.0;
    for (int j = 0; j < data.p_num; j++) {
      eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
      eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
    }

    // Random effects
    if (layout.has_re && data.re_group[i] > 0) {
      int g = data.re_group[i] - 1;
      eta_num += re[g];
      eta_denom += re[g];
    }

    // HSGP spatial effect
    if (data.hsgp_data.shared) {
      eta_num += hsgp_f[i];
      eta_denom += hsgp_f[i];
    } else {
      eta_num += hsgp_f[i];
    }

    // Temporal effect
    int t_idx = -1;
    if (layout.has_temporal && !data.temporal_time_idx.empty() &&
        i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
      t_idx = data.temporal_time_idx[i] - 1;
      if (t_idx >= 0 && t_idx < T_len) {
        if (data.temporal_shared) { eta_num += phi_temporal[t_idx]; eta_denom += phi_temporal[t_idx]; }
        else { eta_num += phi_temporal[t_idx]; }
      }
    }

    if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

    // Likelihood gradients depend on model type
    double dLL_deta_num = 0.0;
    double dLL_deta_denom = 0.0;

    if (data.model_type == ModelType::BINOMIAL) {
      // Binomial: d(log_lik)/d(eta) = y - n*p where p = logit^{-1}(eta)
      double p = 1.0 / (1.0 + std::exp(-eta_num));
      dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
      // denom not used in binomial (y_denom is trials)
    } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
      // NegBin: d(log_lik)/d(eta) = y - mu*(y+phi)/(mu+phi)
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);
      dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
      dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
    } else {  // POISSON_GAMMA
      // Poisson: d(log_lik)/d(eta) = y - mu
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);
      dLL_deta_num = data.y_num[i] - mu_num;
      // Gamma: d(log_lik)/d(eta) = phi * (y/mu - 1)
      dLL_deta_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
    }

    // Accumulate gradients for fixed effects
    for (int j = 0; j < data.p_num; j++) {
      grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
      grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
    }

    // Gradients for RE
    if (layout.has_re && data.re_group[i] > 0) {
      int g = data.re_group[i] - 1;
      grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
    }

    // Gradient w.r.t. phi_num (for NegBin)
    if (layout.has_phi_num && data.model_type == ModelType::NEGBIN_NEGBIN) {
      double mu_num = std::exp(eta_num);
      double y = data.y_num[i];
      // d(log_lik)/d(phi) = digamma(y+phi) - digamma(phi) + log(phi/(mu+phi)) + 1 - (y+phi)/(mu+phi)
      double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                       + std::log(phi_num / (mu_num + phi_num)) + 1.0
                       - (y + phi_num) / (mu_num + phi_num);
      grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
    }

    // Gradient w.r.t. phi_denom
    if (layout.has_phi_denom) {
      if (data.model_type == ModelType::NEGBIN_NEGBIN) {
        double mu_denom = std::exp(eta_denom);
        double y = data.y_denom[i];
        double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                         + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                         - (y + phi_denom) / (mu_denom + phi_denom);
        grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
      } else if (data.model_type == ModelType::POISSON_GAMMA) {
        double mu_denom = std::exp(eta_denom);
        double y = data.y_denom_cont[i];
        double digamma_phi = ratiod::math::portable_digamma(phi_denom);
        double dLL_dphi = std::log(phi_denom) + 1.0 - digamma_phi
                         + std::log(y) - std::log(mu_denom)
                         - y / mu_denom;
        grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
      }
    }

    // Accumulate temporal likelihood gradient
    if (layout.has_temporal && t_idx >= 0 && t_idx < T_len) {
      grad_temporal_lik[t_idx] += data.temporal_shared ? (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
    }

    // Accumulate grad_f for HSGP
    if (data.hsgp_data.shared) {
      grad_f[i] = dLL_deta_num + dLL_deta_denom;
    } else {
      grad_f[i] = dLL_deta_num;
    }
  }

  // Compute HSGP parameter gradients using analytical formulas
  ratiod_hsgp::HSGPGradients hsgp_grads;
  ratiod_hsgp::hsgp_compute_gradients(hsgp_beta, sigma2_hsgp, lengthscale_hsgp,
                                       data.hsgp_data, grad_f, hsgp_grads);

  // Add likelihood contribution to HSGP gradients
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] += hsgp_grads.grad_beta[j];
  }
  grad[layout.log_sigma2_hsgp_idx] += hsgp_grads.grad_log_sigma2;
  grad[layout.log_lengthscale_hsgp_idx] += hsgp_grads.grad_log_lengthscale;

  // =========================================================================
  // Temporal GMRF gradients (same pattern as compute_gradient_analytical)
  // =========================================================================
  if (layout.has_temporal && T_len > 0) {
    for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] = grad_temporal_lik[t];
    if (data.temporal_type == TemporalType::RW1) {
      double qf = 0.0;
      for (int t = 0; t < T_len; t++) {
        double g = 0.0;
        if (t > 0) { g += tau_temporal * (phi_temporal[t-1] - phi_temporal[t]); qf += std::pow(phi_temporal[t] - phi_temporal[t-1], 2); }
        if (t < T_len - 1) g += tau_temporal * (phi_temporal[t+1] - phi_temporal[t]);
        grad[layout.temporal_start + t] += g;
      }
      grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 1) - 0.5 * tau_temporal * qf;
      { double sp = 0.0; for (int t = 0; t < T_len; t++) sp += phi_temporal[t];
        for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
    } else if (data.temporal_type == TemporalType::RW2) {
      double qf = 0.0;
      for (int t = 0; t < T_len; t++) {
        double g = 0.0;
        if (t >= 2) g -= tau_temporal * (phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t]);
        if (t >= 1 && t < T_len - 1) g += 2.0 * tau_temporal * (phi_temporal[t-1] - 2.0*phi_temporal[t] + phi_temporal[t+1]);
        if (t < T_len - 2) g -= tau_temporal * (phi_temporal[t] - 2.0*phi_temporal[t+1] + phi_temporal[t+2]);
        grad[layout.temporal_start + t] += g;
      }
      for (int t = 2; t < T_len; t++) qf += std::pow(phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t], 2);
      grad[layout.log_tau_temporal_idx] += 0.5 * (T_len - 2) - 0.5 * tau_temporal * qf;
      { double sp = 0.0; for (int t = 0; t < T_len; t++) sp += phi_temporal[t];
        for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
    } else if (data.temporal_type == TemporalType::AR1) {
      double omr2 = 1.0 - rho_ar1 * rho_ar1;
      grad[layout.temporal_start] += -tau_temporal * omr2 * phi_temporal[0];
      if (T_len > 1) grad[layout.temporal_start] += tau_temporal * rho_ar1 * (phi_temporal[1] - rho_ar1 * phi_temporal[0]);
      double qf = omr2 * phi_temporal[0] * phi_temporal[0];
      for (int t = 1; t < T_len; t++) {
        double r = phi_temporal[t] - rho_ar1 * phi_temporal[t-1]; qf += r * r;
        double g = -tau_temporal * r;
        if (t < T_len - 1) g += tau_temporal * rho_ar1 * (phi_temporal[t+1] - rho_ar1 * phi_temporal[t]);
        grad[layout.temporal_start + t] += g;
      }
      grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * qf;
      if (layout.logit_rho_ar1_idx >= 0) {
        double gr = -rho_ar1 / omr2 + tau_temporal * rho_ar1 * phi_temporal[0] * phi_temporal[0];
        for (int t = 1; t < T_len; t++) gr += tau_temporal * (phi_temporal[t] - rho_ar1 * phi_temporal[t-1]) * phi_temporal[t-1];
        grad[layout.logit_rho_ar1_idx] += gr * rho_ar1 * (1.0 - rho_ar1);
      }
    }
  }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Multiscale temporal gradient (hand-coded, rows 15, 45, 75)
// Uses analytical gradients from hmc_multiscale_temporal_grad.h
// Supports optional ICAR/BYM2 spatial
// =====================================================================

void compute_gradient_ms_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    const bool fuse_lp = (log_post_out != nullptr);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract base parameters
    const double* beta_num = &params[layout.beta_num_start];
    const double* beta_denom = &params[layout.beta_denom_start];
    double sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    const double* re = layout.has_re ? &params[layout.re_start] : nullptr;
    double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

    // Spatial parameters (ICAR/BYM2 if present)
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // Multiscale temporal parameters
    const auto& mst = data.multiscale_temporal_data;
    int n_trend = layout.trend_end - layout.trend_start;
    int n_seasonal = layout.seasonal_end - layout.seasonal_start;
    int n_short = layout.short_term_end - layout.short_term_start;

    const double* trend = (n_trend > 0) ? &params[layout.trend_start] : nullptr;
    const double* seasonal = (n_seasonal > 0) ? &params[layout.seasonal_start] : nullptr;
    const double* short_term = (n_short > 0) ? &params[layout.short_term_start] : nullptr;

    double sigma2_trend = (n_trend > 0) ? std::exp(params[layout.log_sigma2_trend_idx]) : 1.0;
    double sigma2_seasonal = (n_seasonal > 0) ? std::exp(params[layout.log_sigma2_seasonal_idx]) : 1.0;
    double sigma2_short = (n_short > 0) ? std::exp(params[layout.log_sigma2_short_idx]) : 1.0;
    double rho_short = 0.5;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double logit_rho = params[layout.logit_rho_short_idx];
        double u = 1.0 / (1.0 + std::exp(-logit_rho));
        rho_short = 2.0 * u - 1.0;
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];

    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    if (layout.has_phi_num) grad[layout.log_phi_num_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_num;
    if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_denom;

    // Spatial prior gradients (ICAR/BYM2)
    if (layout.has_spatial && !layout.is_bym2) {
        // ICAR: Gamma prior on tau_spatial
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        // BYM2 Riebler prior gradients
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // PC priors on multiscale temporal variances + Jacobian (+1 for log_sigma2 = log(exp(·)))
    if (n_trend > 0) {
        grad[layout.log_sigma2_trend_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
    }
    if (n_seasonal > 0) {
        grad[layout.log_sigma2_seasonal_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
    }
    if (n_short > 0) {
        grad[layout.log_sigma2_short_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
    }
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        // Beta(2,2) prior on u=(rho+1)/2: log p(u) = log(u) + log(1-u)
        // Jacobian for logit transform: log|drho/d(logit_rho)| = log(u) + log(1-u) + const
        // Total: 2*log(u) + 2*log(1-u), gradient = 2*(1 - 2*u)
        double u = (rho_short + 1.0) / 2.0;
        grad[layout.logit_rho_short_idx] = 2.0 * (1.0 - 2.0 * u);
    }

    // =========================================================================
    // Likelihood loop
    // =========================================================================
    std::vector<double> grad_trend_lik(n_trend, 0.0);
    std::vector<double> grad_seasonal_lik(n_seasonal, 0.0);
    std::vector<double> grad_short_lik(n_short, 0.0);
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);

    for (int i = 0; i < data.N; i++) {
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        for (int j = 0; j < data.p_denom; j++) eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        if (layout.has_re && data.re_group[i] > 0) { eta_num += re[data.re_group[i] - 1]; eta_denom += re[data.re_group[i] - 1]; }

        // Spatial effect
        int s_unit = -1;
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            s_unit = data.spatial_group[i] - 1;
            if (layout.is_bym2) {
                double spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
                eta_num += spatial_eff; eta_denom += spatial_eff;
            } else {
                eta_num += spatial_phi[s_unit]; eta_denom += spatial_phi[s_unit];
            }
        }

        // Multiscale temporal effect
        int t_idx = -1;
        if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            t_idx = mst.time_index[i] - 1;
            double ms_effect = 0.0;
            if (trend != nullptr && t_idx < n_trend) ms_effect += trend[t_idx];
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) ms_effect += seasonal[s_idx];
            }
            if (short_term != nullptr && t_idx < n_short) ms_effect += short_term[t_idx];
            if (mst.shared) { eta_num += ms_effect; eta_denom += ms_effect; }
            else { eta_num += ms_effect; }
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        double dLL_num = 0.0, dLL_denom = 0.0;
        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num;
            dLL_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] += dLL_num * data.X_num_flat[i * data.p_num + j];
        for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] += dLL_denom * data.X_denom_flat[i * data.p_denom + j];
        if (layout.has_re && data.re_group[i] > 0) grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;

        double dLL_shared = dLL_num + dLL_denom;

        // Spatial likelihood gradient
        if (layout.has_spatial && s_unit >= 0) {
            if (layout.is_bym2) {
                grad_spatial_lik[s_unit] += dLL_shared;
                grad_theta_lik[s_unit] += dLL_shared;
            } else {
                grad_spatial_lik[s_unit] += dLL_shared;
            }
        }

        // Multiscale temporal likelihood gradients
        if (t_idx >= 0) {
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik[t_idx] += dLL_temporal;
        }
    }

    // =========================================================================
    // Spatial GMRF prior gradients (ICAR/BYM2)
    // =========================================================================
    if (layout.has_spatial) {
        int S = data.n_spatial_units;
        if (layout.is_bym2) {
            // BYM2 (separate variance): phi and theta gradients
            // ICAR prior on spatial_phi (no tau_spatial for BYM2 — using unit precision)
            for (int s = 0; s < S; s++) {
                double icar_grad = 0.0;
                for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                    int j = data.adj_col_idx[idx];  // already 0-based from R
                    icar_grad += (spatial_phi[j] - spatial_phi[s]);
                }
                // d(LL)/d(phi[s]) = d(LL)/d(spatial) * d(spatial)/d(phi) = dLL_shared * sigma_s * scale
                grad[layout.spatial_start + s] = grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor + icar_grad;
                // d(LL)/d(theta[s]) = d(LL)/d(spatial) * d(spatial)/d(theta) = dLL_shared * sigma_u
                grad[layout.theta_bym2_start + s] = grad_theta_lik[s] * sigma_u_bym2 - theta_bym2[s];
            }
            // Riebler: transform (grad_sigma_s, grad_sigma_u) -> (grad_log_sigma, grad_logit_rho)
            double grad_sigma_s_lik = 0.0, grad_sigma_u_lik = 0.0;
            for (int s = 0; s < S; s++) {
                grad_sigma_s_lik += grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s];
                grad_sigma_u_lik += grad_theta_lik[s] * sigma_u_bym2 * theta_bym2[s];
            }
            grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
            grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                        - rho_bym2 * grad_sigma_u_lik);
        } else {
            // ICAR
            for (int s = 0; s < S; s++) {
                double icar_grad = 0.0;
                for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                    int j = data.adj_col_idx[idx];  // already 0-based from R
                    icar_grad += tau_spatial * (spatial_phi[j] - spatial_phi[s]);
                }
                grad[layout.spatial_start + s] = grad_spatial_lik[s] + icar_grad;
            }
            double icar_qf = icar_quadratic_form(std::vector<double>(spatial_phi, spatial_phi + S), data);
            grad[layout.log_tau_spatial_idx] += 0.5 * (S - 1) - 0.5 * tau_spatial * icar_qf;
        }
    }

    // =========================================================================
    // Multiscale temporal GMRF prior gradients
    // =========================================================================
    ratiod_temporal_grad::MultiscaleTemporalGradients ms_grads;
    ratiod_temporal_grad::multiscale_temporal_prior_gradients(
        trend, n_trend,
        seasonal, n_seasonal,
        short_term, n_short,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        mst, ms_grads);

    for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik[t] + ms_grads.grad_trend[t];
    for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik[t] + ms_grads.grad_seasonal[t];
    for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik[t] + ms_grads.grad_short_term[t];
    if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
    if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
    if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Spatiotemporal interaction gradient (hand-coded, rows 28-29, 58-59, 90-91)
// Supports Knorr-Held Type I-IV with ICAR spatial + RW1/RW2 temporal
// =====================================================================

void compute_gradient_spatiotemporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    const bool fuse_lp = (log_post_out != nullptr);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract base parameters
    const double* beta_num = &params[layout.beta_num_start];
    const double* beta_denom = &params[layout.beta_denom_start];
    double sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    const double* re = layout.has_re ? &params[layout.re_start] : nullptr;
    double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

    // Spatial parameters (ICAR/BYM2)
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // Temporal parameters
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    double rho_ar1 = 0.5;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }

    // Spatiotemporal interaction parameters
    const auto& st = data.spatiotemporal_data;
    int S = st.n_spatial;
    int T = st.n_times;
    int ST = st.n_params;
    double tau_st = std::exp(params[layout.log_tau_st_idx]);
    double tau_st2 = 1.0;
    if (layout.log_tau_st2_idx >= 0) tau_st2 = std::exp(params[layout.log_tau_st2_idx]);

    const double* delta = &params[layout.st_delta_start];

    // =========================================================================
    // Prior gradients for base parameters
    // =========================================================================
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];

    if (layout.has_re && data.n_re_groups > 0) {
        double ratio = sigma_re / data.sigma_re_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }

    if (layout.has_phi_num) grad[layout.log_phi_num_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_num;
    if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] = data.phi_prior_shape - data.phi_prior_rate * phi_denom;

    // Spatial prior (ICAR: Gamma on tau)
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        // BYM2 Riebler prior gradients
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // Temporal prior (Gamma on tau, Beta on rho)
    if (layout.has_temporal) {
        grad[layout.log_tau_temporal_idx] = (data.tau_temporal_shape - 1.0) - data.tau_temporal_rate * tau_temporal + 1.0;
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
        }
    }

    // ST interaction prior on tau_st (PC prior: exponential on sigma_st)
    {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        // d log_prior / d log_tau = d/d(tau) * tau
        // log_prior = log(lambda) - lambda*sigma - log(2*sigma) + log_tau (Jacobian)
        // sigma = 1/sqrt(tau), d sigma/d tau = -0.5 * tau^{-3/2}
        // d log_prior / d tau = (lambda/(2*sigma*tau) + 1/(2*sigma^2*tau))* (-sigma/tau ... )
        // Simpler: work through chain rule
        // d/d(log_tau) = [lambda * 0.5 * sigma - 0.5] * (-1) + 1
        // = 0.5 * (1 - lambda * sigma) + 1 - 0.5
        // = 1.0 - 0.5 * lambda * sigma
        // Actually, from log_post code:
        // log p = log(lambda) - lambda*sigma - log(2*sigma) + log_tau
        // d/d(log_tau) = lambda * (-d sigma/d log_tau) - 1/sigma * (-d sigma/d log_tau) + 1
        // d sigma/d log_tau = d sigma/d tau * tau = -0.5 * tau^{-3/2} * tau = -0.5/sqrt(tau) = -0.5*sigma
        // So: d/d(log_tau) = lambda * 0.5 * sigma + (1/sigma) * 0.5 * sigma + 1
        //                  = 0.5 * lambda * sigma + 0.5 + 1
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
    }
    if (layout.log_tau_st2_idx >= 0) {
        double sigma_st2 = 1.0 / std::sqrt(tau_st2);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        grad[layout.log_tau_st2_idx] = 0.5 * lambda * sigma_st2 + 0.5 + 1.0;
    }

    // =========================================================================
    // Likelihood loop
    // =========================================================================
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_temporal_lik(T_temporal, 0.0);
    std::vector<double> grad_delta_lik(ST, 0.0);

    for (int i = 0; i < data.N; i++) {
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        for (int j = 0; j < data.p_denom; j++) eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        if (layout.has_re && data.re_group[i] > 0) { eta_num += re[data.re_group[i] - 1]; eta_denom += re[data.re_group[i] - 1]; }

        // Spatial effect
        int s_unit = -1;
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            s_unit = data.spatial_group[i] - 1;
            if (layout.is_bym2) {
                double spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
                eta_num += spatial_eff; eta_denom += spatial_eff;
            } else {
                eta_num += spatial_phi[s_unit]; eta_denom += spatial_phi[s_unit];
            }
        }

        // Temporal effect
        int t_base = -1;
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            t_base = data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < T_temporal) {
                if (data.temporal_shared) { eta_num += phi_temporal[t_base]; eta_denom += phi_temporal[t_base]; }
                else { eta_num += phi_temporal[t_base]; }
            }
        }

        // Spatiotemporal interaction effect
        int st_idx = -1;
        if (st.st_flat[i] > 0) {
            st_idx = st.st_flat[i] - 1;
            double st_effect = delta[st_idx];
            if (st.shared) { eta_num += st_effect; eta_denom += st_effect; }
            else { eta_num += st_effect; }
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        double dLL_num = 0.0, dLL_denom = 0.0;
        if (data.model_type == ModelType::BINOMIAL) {
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            dLL_num = data.y_num[i] - data.y_denom[i] * p;
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
        } else {
            double mu_num = std::exp(eta_num), mu_denom = std::exp(eta_denom);
            dLL_num = data.y_num[i] - mu_num;
            dLL_denom = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
        }

        for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] += dLL_num * data.X_num_flat[i * data.p_num + j];
        for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] += dLL_denom * data.X_denom_flat[i * data.p_denom + j];
        if (layout.has_re && data.re_group[i] > 0) grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;

        double dLL_shared = dLL_num + dLL_denom;

        if (layout.has_spatial && s_unit >= 0) {
            if (layout.is_bym2) { grad_spatial_lik[s_unit] += dLL_shared; grad_theta_lik[s_unit] += dLL_shared; }
            else { grad_spatial_lik[s_unit] += dLL_shared; }
        }
        if (layout.has_temporal && t_base >= 0 && t_base < T_temporal) {
            grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }
        if (st_idx >= 0) {
            grad_delta_lik[st_idx] += st.shared ? dLL_shared : dLL_num;
        }
    }

    // =========================================================================
    // Spatial GMRF prior gradients (ICAR/BYM2)
    // =========================================================================
    if (layout.has_spatial) {
        int S_sp = data.n_spatial_units;
        if (layout.is_bym2) {
            // BYM2 (separate variance): phi and theta gradients
            for (int s = 0; s < S_sp; s++) {
                double icar_grad = 0.0;
                for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                    int j = data.adj_col_idx[idx];  // already 0-based from R
                    icar_grad += (spatial_phi[j] - spatial_phi[s]);
                }
                grad[layout.spatial_start + s] = grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor + icar_grad;
                grad[layout.theta_bym2_start + s] = grad_theta_lik[s] * sigma_u_bym2 - theta_bym2[s];
            }
            // Riebler: transform (grad_sigma_s, grad_sigma_u) -> (grad_log_sigma, grad_logit_rho)
            double grad_sigma_s_lik = 0.0, grad_sigma_u_lik = 0.0;
            for (int s = 0; s < S_sp; s++) {
                grad_sigma_s_lik += grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s];
                grad_sigma_u_lik += grad_theta_lik[s] * sigma_u_bym2 * theta_bym2[s];
            }
            grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
            grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                        - rho_bym2 * grad_sigma_u_lik);
        } else {
            for (int s = 0; s < S_sp; s++) {
                double icar_grad = 0.0;
                for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                    int j = data.adj_col_idx[idx];  // already 0-based from R
                    icar_grad += tau_spatial * (spatial_phi[j] - spatial_phi[s]);
                }
                grad[layout.spatial_start + s] = grad_spatial_lik[s] + icar_grad;
            }
            double icar_qf = icar_quadratic_form(std::vector<double>(spatial_phi, spatial_phi + S_sp), data);
            grad[layout.log_tau_spatial_idx] += 0.5 * (S_sp - 1) - 0.5 * tau_spatial * icar_qf;
        }
    }

    // =========================================================================
    // Temporal GMRF prior gradients
    // =========================================================================
    if (layout.has_temporal && T_temporal > 0) {
        for (int t = 0; t < T_temporal; t++) grad[layout.temporal_start + t] = grad_temporal_lik[t];
        if (data.temporal_type == TemporalType::RW1) {
            double qf = 0.0;
            for (int t = 0; t < T_temporal; t++) {
                double g = 0.0;
                if (t > 0) { g += tau_temporal * (phi_temporal[t-1] - phi_temporal[t]); qf += std::pow(phi_temporal[t] - phi_temporal[t-1], 2); }
                if (t < T_temporal - 1) g += tau_temporal * (phi_temporal[t+1] - phi_temporal[t]);
                grad[layout.temporal_start + t] += g;
            }
            grad[layout.log_tau_temporal_idx] += 0.5 * (T_temporal - 1) - 0.5 * tau_temporal * qf;
            { double sp = 0.0; for (int t = 0; t < T_temporal; t++) sp += phi_temporal[t];
              for (int t = 0; t < T_temporal; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
        } else if (data.temporal_type == TemporalType::RW2) {
            double qf = 0.0;
            for (int t = 0; t < T_temporal; t++) {
                double g = 0.0;
                if (t >= 2) g -= tau_temporal * (phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t]);
                if (t >= 1 && t < T_temporal - 1) g += 2.0 * tau_temporal * (phi_temporal[t-1] - 2.0*phi_temporal[t] + phi_temporal[t+1]);
                if (t < T_temporal - 2) g -= tau_temporal * (phi_temporal[t] - 2.0*phi_temporal[t+1] + phi_temporal[t+2]);
                grad[layout.temporal_start + t] += g;
            }
            for (int t = 2; t < T_temporal; t++) qf += std::pow(phi_temporal[t-2] - 2.0*phi_temporal[t-1] + phi_temporal[t], 2);
            grad[layout.log_tau_temporal_idx] += 0.5 * (T_temporal - 2) - 0.5 * tau_temporal * qf;
            { double sp = 0.0; for (int t = 0; t < T_temporal; t++) sp += phi_temporal[t];
              for (int t = 0; t < T_temporal; t++) grad[layout.temporal_start + t] -= 0.001 * sp; }
        } else if (data.temporal_type == TemporalType::AR1) {
            double omr2 = 1.0 - rho_ar1 * rho_ar1;
            grad[layout.temporal_start] += -tau_temporal * omr2 * phi_temporal[0];
            if (T_temporal > 1) grad[layout.temporal_start] += tau_temporal * rho_ar1 * (phi_temporal[1] - rho_ar1 * phi_temporal[0]);
            double qf = omr2 * phi_temporal[0] * phi_temporal[0];
            for (int t = 1; t < T_temporal; t++) {
                double r = phi_temporal[t] - rho_ar1 * phi_temporal[t-1]; qf += r * r;
                double g = -tau_temporal * r;
                if (t < T_temporal - 1) g += tau_temporal * rho_ar1 * (phi_temporal[t+1] - rho_ar1 * phi_temporal[t]);
                grad[layout.temporal_start + t] += g;
            }
            grad[layout.log_tau_temporal_idx] += 0.5 * T_temporal - 0.5 * tau_temporal * qf;
            if (layout.logit_rho_ar1_idx >= 0) {
                double gr = -rho_ar1 / omr2 + tau_temporal * rho_ar1 * phi_temporal[0] * phi_temporal[0];
                for (int t = 1; t < T_temporal; t++) gr += tau_temporal * (phi_temporal[t] - rho_ar1 * phi_temporal[t-1]) * phi_temporal[t-1];
                grad[layout.logit_rho_ar1_idx] += gr * rho_ar1 * (1.0 - rho_ar1);
            }
        }
    }

    // =========================================================================
    // Spatiotemporal interaction prior gradients (Type I-IV)
    // =========================================================================
    // delta is stored column-major: delta[s*T + t]
    if (st.type == STType::TYPE_I) {
        // IID: log p = 0.5*n*log(tau) - 0.5*tau*sum(delta^2)
        double qf = 0.0;
        for (int k = 0; k < ST; k++) {
            grad[layout.st_delta_start + k] = grad_delta_lik[k] - tau_st * delta[k];
            qf += delta[k] * delta[k];
        }
        grad[layout.log_tau_st_idx] += 0.5 * ST - 0.5 * tau_st * qf;

    } else if (st.type == STType::TYPE_II) {
        // Structured time per spatial unit: temporal GMRF applied to delta[s,:]
        double total_qf = 0.0;
        for (int s = 0; s < S; s++) {
            // Apply temporal stencil to delta[s*T .. s*T+T-1]
            const double* delta_s = &delta[s * T];
            if (st.temporal_type == TemporalType::RW1) {
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    double g = 0.0;
                    if (t > 0) { g += tau_st * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
                    if (t < T - 1) g += tau_st * (delta_s[t+1] - delta_s[t]);
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + g;
                }
                total_qf += qf;
            } else if (st.temporal_type == TemporalType::RW2) {
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    double g = 0.0;
                    if (t >= 2) g -= tau_st * (delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t]);
                    if (t >= 1 && t < T - 1) g += 2.0 * tau_st * (delta_s[t-1] - 2.0*delta_s[t] + delta_s[t+1]);
                    if (t < T - 2) g -= tau_st * (delta_s[t] - 2.0*delta_s[t+1] + delta_s[t+2]);
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + g;
                }
                for (int t = 2; t < T; t++) qf += std::pow(delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t], 2);
                total_qf += qf;
            }
        }
        int rank_per_unit = (st.temporal_type == TemporalType::RW1) ? (T - 1) :
                            (st.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        grad[layout.log_tau_st_idx] += 0.5 * S * rank_per_unit - 0.5 * tau_st * total_qf;

    } else if (st.type == STType::TYPE_III) {
        // Structured space per time point: ICAR applied to delta[:,t]
        double total_qf = 0.0;
        for (int t = 0; t < T; t++) {
            // Apply ICAR stencil to delta[0*T+t, 1*T+t, ..., (S-1)*T+t]
            for (int s = 0; s < S; s++) {
                double icar_grad = 0.0;
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    icar_grad += tau_st * (delta[j * T + t] - delta[s * T + t]);
                }
                grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + icar_grad;
            }
            // Compute ICAR quadratic form for this time slice
            for (int s = 0; s < S; s++) {
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    if (j > s) {
                        double diff = delta[s * T + t] - delta[j * T + t];
                        total_qf += diff * diff;
                    }
                }
            }
        }
        int rank_spatial = S - 1;
        grad[layout.log_tau_st_idx] += 0.5 * T * rank_spatial - 0.5 * tau_st * total_qf;

    } else if (st.type == STType::TYPE_IV) {
        // Kronecker: Q_delta = Q_s ⊗ Q_t
        // Step 1: Apply temporal stencil to each spatial unit's series: v[s,t] = (Q_t * delta[s,:])_t
        std::vector<double> v(S * T, 0.0);
        if (st.temporal_type == TemporalType::RW1) {
            for (int s = 0; s < S; s++) {
                for (int t = 0; t < T; t++) {
                    double qt_delta = 0.0;
                    int n_t_neigh = 0;
                    if (t > 0) { qt_delta -= delta[s * T + t - 1]; n_t_neigh++; }
                    if (t < T - 1) { qt_delta -= delta[s * T + t + 1]; n_t_neigh++; }
                    qt_delta += n_t_neigh * delta[s * T + t];
                    v[s * T + t] = qt_delta;
                }
            }
        } else if (st.temporal_type == TemporalType::RW2) {
            // RW2 precision matrix action: more complex stencil
            for (int s = 0; s < S; s++) {
                const double* d_s = &delta[s * T];
                for (int t = 0; t < T; t++) {
                    double qt_delta = 0.0;
                    // RW2: Q_t is the second-difference precision
                    // Q_t[t,t] = sum of contributions from all second-differences involving t
                    for (int k = std::max(0, t - 2); k <= std::min(T - 3, t); k++) {
                        // Second difference at k: d_s[k] - 2*d_s[k+1] + d_s[k+2]
                        // Coefficient of d_s[t] in this second difference:
                        int pos = t - k;  // 0, 1, or 2
                        double coef = (pos == 1) ? -2.0 : 1.0;
                        // The quadratic form contributes coef * d2[k] to the gradient
                        double d2 = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
                        qt_delta += coef * d2;
                    }
                    v[s * T + t] = qt_delta;
                }
            }
        }

        // Step 2: Apply spatial ICAR stencil to v: result[s,t] = (Q_s * v[:,t])_s
        // This gives (Q_s ⊗ Q_t) delta
        double total_qf = 0.0;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                double qs_v = 0.0;
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    qs_v -= v[j * T + t];
                }
                int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
                qs_v += n_neigh * v[s * T + t];

                grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] - tau_st * tau_st2 * qs_v;
                total_qf += delta[s * T + t] * qs_v;
            }
        }

        int rank_space = S - 1;
        int rank_time = (st.temporal_type == TemporalType::RW1) ? (T - 1) :
                        (st.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        if (st.temporal_cyclic) rank_time = T;
        int total_rank = rank_space * rank_time;

        grad[layout.log_tau_st_idx] += 0.5 * total_rank - 0.5 * tau_st * tau_st2 * total_qf;
        if (layout.log_tau_st2_idx >= 0) {
            grad[layout.log_tau_st2_idx] += 0.5 * total_rank - 0.5 * tau_st * tau_st2 * total_qf;
        }
    }

    // Sum-to-zero penalty gradients
    {
        double lambda_stz = 0.001;
        // Marginal over space: for each t, sum_s delta[s*T+t] -> 0
        for (int t = 0; t < T; t++) {
            double row_sum = 0.0;
            for (int s = 0; s < S; s++) row_sum += delta[s * T + t];
            for (int s = 0; s < S; s++) {
                grad[layout.st_delta_start + s * T + t] -= lambda_stz * row_sum;
            }
        }
        // Marginal over time: for each s, sum_t delta[s*T+t] -> 0
        for (int s = 0; s < S; s++) {
            double col_sum = 0.0;
            for (int t = 0; t < T; t++) col_sum += delta[s * T + t];
            for (int t = 0; t < T; t++) {
                grad[layout.st_delta_start + s * T + t] -= lambda_stz * col_sum;
            }
        }
    }

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// Global gradient mode (set by cpp_hmc_fit, used by compute_gradient)
// Thread-safe since each chain runs in its own process or the mode is set once before sampling
static GradientMode g_gradient_mode = GradientMode::AUTO;

// Set global gradient mode (called at start of sampling)
void set_gradient_mode(GradientMode mode) {
    g_gradient_mode = mode;
}

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Check explicit mode first
    if (g_gradient_mode == GradientMode::NUMERICAL) {
        compute_gradient_numerical(params, data, layout, grad, log_post_out);
        return;
    }

    if (g_gradient_mode == GradientMode::AUTODIFF_TAPE) {
        compute_gradient_autodiff(params, data, layout, grad, log_post_out);
        return;
    }

    if (g_gradient_mode == GradientMode::AUTODIFF_ARENA) {
        compute_gradient_arena(params, data, layout, grad, log_post_out);
        return;
    }

    if (g_gradient_mode == GradientMode::AUTODIFF_FORWARD) {
        compute_gradient_forward(params, data, layout, grad, log_post_out);
        return;
    }

    // AUTO or HANDCODED mode: use fastest available
    // Priority: H (hand-coded) > A_r (arena autodiff) > A (forward) > N (numerical)
    // Note: A_r is O(N) with SoA layout, A is O(p×N), A_t is O(N) but slow due to heap alloc

    // Use hand-coded analytical gradients for simple models (fastest, 9x)
    if (can_use_analytical_gradient(data, layout)) {
        compute_gradient_analytical(params, data, layout, grad, log_post_out);
    } else if (layout.is_hsgp && data.has_hsgp) {
        // HSGP has hand-coded analytical gradients (~50x faster than autodiff)
        // Supports all families: poisson_gamma, negbin_negbin, binomial
        compute_gradient_hsgp(params, data, layout, grad, log_post_out);
    } else if (layout.is_gp && data.has_gp && !layout.has_temporal) {
        // Single-scale GP uses hand-coded gradients (~3x faster than autodiff)
        compute_gradient_gp_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.is_multiscale_gp && data.has_multiscale_gp && layout.has_temporal) {
        // Multi-scale GP + temporal uses hand-coded gradients
        compute_gradient_msgp_temporal_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        // Multi-scale GP uses hand-coded gradients (~2-3x faster than autodiff)
        compute_gradient_msgp_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.is_gp && layout.has_temporal) {
        // GP+temporal uses hand-coded gradients
        compute_gradient_gp_temporal_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.has_svc && data.has_svc) {
        // SVC uses hand-coded gradients (~3x faster than autodiff)
        compute_gradient_svc_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.has_tvc && data.has_tvc) {
        // TVC uses hand-coded gradients (~3x faster than autodiff)
        compute_gradient_tvc_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.has_spatiotemporal && !layout.is_st_gp &&
               !layout.is_gp && !layout.is_multiscale_gp &&
               data.spatiotemporal_data.type != STType::NONE &&
               layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0) {
        // Spatiotemporal interaction (Knorr-Held Type I-IV) with hand-coded gradients
        // Handles base spatial (ICAR/BYM2) + temporal (RW1/RW2/AR1) + ST interaction
        compute_gradient_spatiotemporal_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.is_temporal_gp && layout.has_temporal &&
               !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
               data.temporal_gp_data.cov_type == ratiod_temporal_gp::TemporalCovType::EXPONENTIAL) {
        // Temporal GP (standalone, exponential covariance) uses hand-coded gradients
        compute_gradient_temporal_gp_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.has_multiscale_temporal && !layout.is_gp && !layout.is_multiscale_gp &&
               !layout.is_hsgp && !layout.has_svc && !layout.has_tvc &&
               !layout.has_latent && !layout.has_spatiotemporal) {
        // Multiscale temporal (trend + seasonal + short-term) with hand-coded gradients
        // Supports optional ICAR/BYM2 spatial
        compute_gradient_ms_temporal_handcoded(params, data, layout, grad, log_post_out);
    } else if (layout.has_latent && data.latent_n_factors > 0) {
        // Latent factor models have hand-coded gradients (O(N*K))
        compute_gradient_latent_handcoded(params, data, layout, grad, log_post_out);
    } else {
        // No hand-coded gradients available — use arena reverse-mode autodiff
        // Arena is O(N) with SoA layout, ~10-30x faster than tape, ~2x faster than forward for p>5
        compute_gradient_arena(params, data, layout, grad, log_post_out);
    }
}

// =====================================================================
// Resolve gradient function pointer (called once at sampling start)
// Same dispatch logic as compute_gradient(), but returns a function pointer
// to eliminate per-call branching during leapfrog steps.
// =====================================================================

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    // Explicit mode overrides
    if (mode == GradientMode::NUMERICAL)
        return &compute_gradient_numerical;
    if (mode == GradientMode::AUTODIFF_TAPE)
        return &compute_gradient_autodiff;
    if (mode == GradientMode::AUTODIFF_ARENA)
        return &compute_gradient_arena;
    if (mode == GradientMode::AUTODIFF_FORWARD)
        return &compute_gradient_forward;

    // AUTO or HANDCODED: use fastest available (H > A_r > A > N)
    if (can_use_analytical_gradient(data, layout)) {
        return &compute_gradient_analytical;
    }
    if (layout.is_hsgp && data.has_hsgp)
        return &compute_gradient_hsgp;
    if (layout.is_gp && data.has_gp && !layout.has_temporal)
        return &compute_gradient_gp_handcoded;
    if (layout.is_multiscale_gp && data.has_multiscale_gp && layout.has_temporal)
        return &compute_gradient_msgp_temporal_handcoded;
    if (layout.is_multiscale_gp && data.has_multiscale_gp)
        return &compute_gradient_msgp_handcoded;
    if (layout.is_gp && layout.has_temporal)
        return &compute_gradient_gp_temporal_handcoded;
    if (layout.has_svc && data.has_svc)
        return &compute_gradient_svc_handcoded;
    if (layout.has_tvc && data.has_tvc)
        return &compute_gradient_tvc_handcoded;
    if (layout.has_spatiotemporal && !layout.is_st_gp &&
        !layout.is_gp && !layout.is_multiscale_gp &&
        data.spatiotemporal_data.type != STType::NONE &&
        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0)
        return &compute_gradient_spatiotemporal_handcoded;
    if (layout.is_temporal_gp && layout.has_temporal &&
        !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
        data.temporal_gp_data.cov_type == ratiod_temporal_gp::TemporalCovType::EXPONENTIAL)
        return &compute_gradient_temporal_gp_handcoded;
    if (layout.has_multiscale_temporal && !layout.is_gp && !layout.is_multiscale_gp &&
        !layout.is_hsgp && !layout.has_svc && !layout.has_tvc &&
        !layout.has_latent && !layout.has_spatiotemporal)
        return &compute_gradient_ms_temporal_handcoded;
    if (layout.has_latent && data.latent_n_factors > 0)
        return &compute_gradient_latent_handcoded;

    // Fallback: arena autodiff
    return &compute_gradient_arena;
}

// =====================================================================
// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, int n_params, double target_boost)
  : mu(std::log(10.0 * epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(compute_target(n_params, target_boost)), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: exp(-14) ≈ 8e-7, Upper bound: exp(2) ≈ 7.4
  log_epsilon = std::max(-14.0, std::min(log_epsilon, 2.0));
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

  // Get inverse mass matrix (= variance, regularized for stability)
  // For HMC: M = diag(1/var), so M^{-1} = diag(var)
  // High variance parameters should move faster in position space
  // Uses Stan-style Bayesian shrinkage toward unit variance to prevent
  // extreme mass matrix entries from small sample sizes
  std::vector<double> inv_mass() const {
    auto var = variance();
    double shrink = (n < 2) ? 0.0 : (double)n / (n + 5.0);
    for (size_t i = 0; i < var.size(); i++) {
      var[i] = shrink * var[i] + 1e-3 * (5.0 / (n + 5.0));
      // Safety clamp for extreme values
      var[i] = std::max(1e-3, std::min(var[i], 1e3));
    }
    return var;
  }

  // Get sqrt of mass matrix for momentum sampling
  // Since M = diag(1/var), sqrt(M) = diag(1/sqrt(var))
  // p ~ N(0, M), so p_i = z_i / sqrt(var_i)
  // Uses the same regularized variance as inv_mass()
  std::vector<double> sqrt_mass() const {
    auto inv_m = inv_mass();
    std::vector<double> sqrt_m(inv_m.size());
    for (size_t i = 0; i < inv_m.size(); i++) {
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
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

  // Half step for momentum (fused gradient + log_prob)
  compute_gradient(result.q, data, layout, grad, &result.log_prob);
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

  // Half step for momentum (fused gradient + log_prob)
  compute_gradient(result.q, data, layout, grad, &result.log_prob);
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
  // Stan-style algorithm: start at epsilon=1, double or halve until
  // acceptance probability crosses 0.5
  int n = q.size();

  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> p(n);
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng);
  }

  // Fused: compute gradient + log_post in a single O(N) pass
  // (eliminates redundant compute_log_post call)
  double log_prob_init;
  std::vector<double> grad_init(n);
  compute_gradient(q, data, layout, grad_init, &log_prob_init);
  double kinetic_init = 0.5 * ratiod_linalg::norm_squared(p.data(), n);
  double H_init = -log_prob_init + kinetic_init;

  double epsilon = 1.0;

  LeapfrogResult lf = leapfrog_step(q, p, epsilon, data, layout);
  double kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
  double H_new = -lf.log_prob + kinetic_new;
  double delta_H = H_new - H_init;

  // Determine direction: if accept prob > 0.5, increase; else decrease
  // accept_prob = exp(-delta_H), so > 0.5 iff delta_H < log(2)
  int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;

  for (int iter = 0; iter < 50; iter++) {
    if (direction == 1) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon < 1e-10 || epsilon > 1e5) break;

    lf = leapfrog_step(q, p, epsilon, data, layout);
    if (!std::isfinite(lf.log_prob)) {
      if (direction == 1) break;  // Was increasing, hit instability
      continue;  // Was decreasing, keep going
    }

    kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    // Stop when we cross the 0.5 acceptance threshold
    if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
  }

  return std::max(1e-10, std::min(epsilon, 1e3));
}

// Mass-aware version: uses diagonal mass matrix for leapfrog and kinetic energy
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
) {
  int n = q.size();

  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> p(n);
  // Sample p ~ N(0, M) where M = diag(1/inv_mass)
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng) / std::sqrt(inv_mass[i]);
  }

  // Fused: compute gradient + log_post in a single O(N) pass
  double log_prob_init;
  std::vector<double> grad_init(n);
  compute_gradient(q, data, layout, grad_init, &log_prob_init);
  double kinetic_init = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_init += p[i] * p[i] * inv_mass[i];
  }
  kinetic_init *= 0.5;
  double H_init = -log_prob_init + kinetic_init;

  double epsilon = 1.0;

  LeapfrogResult lf = leapfrog_step_mass(q, p, epsilon, inv_mass, data, layout);
  double kinetic_new = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_new += lf.p[i] * lf.p[i] * inv_mass[i];
  }
  kinetic_new *= 0.5;
  double H_new = -lf.log_prob + kinetic_new;
  double delta_H = H_new - H_init;

  int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;

  for (int iter = 0; iter < 50; iter++) {
    if (direction == 1) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon < 1e-10 || epsilon > 1e5) break;

    lf = leapfrog_step_mass(q, p, epsilon, inv_mass, data, layout);
    if (!std::isfinite(lf.log_prob)) {
      if (direction == 1) break;
      continue;
    }

    kinetic_new = 0.0;
    for (int i = 0; i < n; i++) {
      kinetic_new += lf.p[i] * lf.p[i] * inv_mass[i];
    }
    kinetic_new *= 0.5;
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
  }

  return std::max(1e-10, std::min(epsilon, 1e3));
}

// Dense-mass-aware version: uses full DenseMassMatrix for momentum sampling,
// leapfrog integration, and kinetic energy computation.
// This ensures the step size is calibrated for the rotated phase space
// dynamics of the dense mass matrix, not just the diagonal approximation.
double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
) {
  int n = q.size();

  // Sample momentum using the full dense mass matrix: p ~ N(0, M)
  std::vector<double> p(n);
  // const_cast is safe here - sample_momentum only reads rng state
  const_cast<DenseMassMatrix&>(mass).sample_momentum(p.data(), rng);

  // Compute gradient + log_post
  double log_prob_init;
  std::vector<double> grad_init(n);
  compute_gradient(q, data, layout, grad_init, &log_prob_init);

  // Kinetic energy using full dense mass matrix
  double kinetic_init = mass.kinetic_energy(p.data());
  double H_init = -log_prob_init + kinetic_init;

  double epsilon = 1.0;

  // Single leapfrog step using dense mass matrix
  auto do_leapfrog = [&](const std::vector<double>& q_in,
                         const std::vector<double>& p_in,
                         double eps) -> std::pair<std::vector<double>, std::vector<double>> {
    std::vector<double> q_out = q_in;
    std::vector<double> p_out = p_in;
    std::vector<double> grad(n);
    compute_gradient(q_out, data, layout, grad);

    // Half step for momentum
    for (int i = 0; i < n; i++) p_out[i] += 0.5 * eps * grad[i];

    // Full step for position: q += eps * M^{-1} * p
    std::vector<double> Mp(n);
    mass.inv_mass_times_p(p_out.data(), Mp.data());
    for (int i = 0; i < n; i++) q_out[i] += eps * Mp[i];

    // Compute gradient at new position
    compute_gradient(q_out, data, layout, grad);

    // Half step for momentum
    for (int i = 0; i < n; i++) p_out[i] += 0.5 * eps * grad[i];

    return {q_out, p_out};
  };

  auto [q_new, p_new] = do_leapfrog(q, p, epsilon);

  // Check if log_prob is finite at new position
  double log_prob_new;
  std::vector<double> grad_new(n);
  compute_gradient(q_new, data, layout, grad_new, &log_prob_new);

  double kinetic_new = mass.kinetic_energy(p_new.data());
  double H_new = -log_prob_new + kinetic_new;
  double delta_H = H_new - H_init;

  int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;

  for (int iter = 0; iter < 50; iter++) {
    if (direction == 1) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon < 1e-10 || epsilon > 1e5) break;

    auto [q_try, p_try] = do_leapfrog(q, p, epsilon);

    double lp_try;
    std::vector<double> grad_try(n);
    compute_gradient(q_try, data, layout, grad_try, &lp_try);

    if (!std::isfinite(lp_try)) {
      if (direction == 1) break;
      continue;
    }

    kinetic_new = mass.kinetic_energy(p_try.data());
    H_new = -lp_try + kinetic_new;
    delta_H = H_new - H_init;

    if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
  }

  return std::max(1e-10, std::min(epsilon, 1e3));
}

// =====================================================================
// NUTS (No-U-Turn Sampler) helper functions
// =====================================================================

double nuts_log_sum_exp(double a, double b) {
  double m = std::max(a, b);
  if (!std::isfinite(m)) return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n) {
  double kinetic = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic += p[i] * p[i] * inv_mass[i];
  }
  return -log_prob + 0.5 * kinetic;
}

bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n) {
  // Generalized U-turn criterion (Betancourt 2017, Section 3.2)
  // Check both directions: (q+ - q-) . (M^-1 p-) and (q+ - q-) . (M^-1 p+)
  double dot_fwd = 0.0, dot_bwd = 0.0;
  for (int i = 0; i < n; i++) {
    double dq = q_plus[i] - q_minus[i];
    dot_fwd += dq * (inv_mass[i] * p_plus[i]);
    dot_bwd += dq * (inv_mass[i] * p_minus[i]);
  }
  return (dot_fwd < 0.0) || (dot_bwd < 0.0);
}

LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout) {

  int n = q.size();
  LeapfrogResultWithGrad result;
  result.q = q;
  result.p = p;
  result.grad.resize(n);
  result.divergent = false;

  // Half step for momentum using provided gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  if (use_mass) {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * inv_mass[i] * result.p[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * result.p[i];
    }
  }

  // Compute gradient and log_prob at new position (fused: single O(N) pass)
  compute_gradient(result.q, data, layout, result.grad, &result.log_prob);

  // Half step for momentum using new gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * result.grad[i];
  }

  // Check for divergence
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

NUTSTreeResult build_tree(const NUTSNode& node, int direction, int depth,
                          double epsilon, const std::vector<double>& inv_mass,
                          bool use_mass, double H0, double delta_max,
                          const ModelData& data, const ParamLayout& layout,
                          std::mt19937& rng) {
  int n = node.q.size();
  NUTSTreeResult result;

  if (depth == 0) {
    // Base case: single leapfrog step
    LeapfrogResultWithGrad lf = leapfrog_step_with_grad(
      node.q, node.p, node.grad,
      direction * epsilon, inv_mass, use_mass, data, layout
    );

    double H_new = nuts_compute_hamiltonian(lf.log_prob, lf.p, inv_mass, n);
    double delta_H = H_new - H0;

    result.left.q = lf.q;
    result.left.p = lf.p;
    result.left.grad = lf.grad;
    result.left.log_prob = lf.log_prob;
    result.right = result.left;

    result.q_proposal = std::move(lf.q);
    result.grad_proposal = std::move(lf.grad);
    result.log_prob_proposal = lf.log_prob;

    // Multinomial weight: log(weight) = -H_new (proportional to exp(-H))
    result.sum_log_weight = -H_new;

    // Divergence check
    result.divergent = lf.divergent || (delta_H > delta_max);
    result.stop = result.divergent;

    // Valid if not divergent
    result.n_valid = result.divergent ? 0 : 1;

    // Acceptance statistic: min(1, exp(-delta_H))
    double accept_stat = std::min(1.0, std::exp(-delta_H));
    if (!std::isfinite(accept_stat)) accept_stat = 0.0;
    result.sum_accept_prob = accept_stat;
    result.n_leapfrog = 1;

    return result;
  }

  // Recursive case: build inner subtree
  NUTSTreeResult inner = build_tree(node, direction, depth - 1,
                                     epsilon, inv_mass, use_mass, H0, delta_max,
                                     data, layout, rng);
  result = std::move(inner);

  if (result.stop) return result;

  // Build outer subtree from the appropriate endpoint
  // (read from result, since inner is moved-from)
  NUTSNode outer_start;
  if (direction == 1) {
    outer_start = result.right;
  } else {
    outer_start = result.left;
  }

  NUTSTreeResult outer = build_tree(outer_start, direction, depth - 1,
                                     epsilon, inv_mass, use_mass, H0, delta_max,
                                     data, layout, rng);

  // Combine results (use result.xxx since inner was moved into result)
  result.n_leapfrog += outer.n_leapfrog;
  result.sum_accept_prob += outer.sum_accept_prob;
  result.divergent = result.divergent || outer.divergent;

  // Multinomial sampling: accept outer proposal with probability
  // exp(outer.sum_log_weight) / exp(log_sum_exp(result.sum_log_weight, outer.sum_log_weight))
  double new_sum_log_weight = nuts_log_sum_exp(result.sum_log_weight, outer.sum_log_weight);
  double accept_prob_outer = std::exp(outer.sum_log_weight - new_sum_log_weight);
  if (!std::isfinite(accept_prob_outer)) accept_prob_outer = 0.0;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  if (unif(rng) < accept_prob_outer) {
    result.q_proposal = std::move(outer.q_proposal);
    result.grad_proposal = std::move(outer.grad_proposal);
    result.log_prob_proposal = outer.log_prob_proposal;
  }
  // else keep inner proposal (already in result from move)

  result.sum_log_weight = new_sum_log_weight;
  result.n_valid = result.n_valid + outer.n_valid;

  // Update tree endpoints
  // result already holds inner's data from the move above
  if (direction == 1) {
    // result.left is already inner.left (correct from move)
    result.right = std::move(outer.right);
  } else {
    result.left = std::move(outer.left);
    // result.right is already inner.right (correct from move)
  }

  // Check U-turn on the combined trajectory
  result.stop = outer.stop ||
    nuts_check_uturn(result.left.q, result.right.q,
                     result.left.p, result.right.p,
                     inv_mass, n);

  return result;
}

// =====================================================================
// Optimized NUTS: zero-allocation infrastructure
// =====================================================================

// Pointer-based Hamiltonian (avoids std::vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n) {
  return -log_prob + mass.kinetic_energy(p);
}

// Pointer-based U-turn check
// scratch: temporary buffer of size n (for dense matvec result)
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n) {
  if (mass.type == MassMatrixType::DIAG || !mass.adapted) {
    // Diagonal path (fast, unrolled)
    double dot_fwd = 0.0, dot_bwd = 0.0;
    const double* inv_mass = mass.inv_mass_diag.data();
    int i = 0;
    for (; i + 3 < n; i += 4) {
      double dq0 = q_plus[i]   - q_minus[i];
      double dq1 = q_plus[i+1] - q_minus[i+1];
      double dq2 = q_plus[i+2] - q_minus[i+2];
      double dq3 = q_plus[i+3] - q_minus[i+3];
      dot_fwd += dq0 * (inv_mass[i]   * p_plus[i])
               + dq1 * (inv_mass[i+1] * p_plus[i+1])
               + dq2 * (inv_mass[i+2] * p_plus[i+2])
               + dq3 * (inv_mass[i+3] * p_plus[i+3]);
      dot_bwd += dq0 * (inv_mass[i]   * p_minus[i])
               + dq1 * (inv_mass[i+1] * p_minus[i+1])
               + dq2 * (inv_mass[i+2] * p_minus[i+2])
               + dq3 * (inv_mass[i+3] * p_minus[i+3]);
    }
    for (; i < n; i++) {
      double dq = q_plus[i] - q_minus[i];
      dot_fwd += dq * (inv_mass[i] * p_plus[i]);
      dot_bwd += dq * (inv_mass[i] * p_minus[i]);
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else {
    // Dense path: compute (q+ - q-) . (C * p+) and (q+ - q-) . (C * p-)
    // Use scratch for C * p
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  }
}

// In-place leapfrog step operating on a workspace slot
// Mutates q, p, grad in the slot directly — zero heap allocation
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout) {

  double* q = ws.q_at(slot);
  double* p = ws.p_at(slot);
  double* grad = ws.grad_at(slot);
  int n = ws.n;

  LeapfrogInPlaceResult result;
  result.divergent = false;

  // Half step for momentum using current gradient
  ratiod_linalg::axpy(0.5 * epsilon, grad, p, n);

  // Full step for position: q += eps * C * p
  if (!mass.adapted) {
    // Identity mass: q += eps * p
    ratiod_linalg::axpy(epsilon, p, q, n);
  } else if (mass.type == MassMatrixType::DIAG) {
    // Diagonal: q[i] += eps * inv_mass[i] * p[i]
    ratiod_linalg::axpy_weighted(epsilon, mass.inv_mass_diag.data(), p, q, n);
  } else {
    // Dense: q += eps * C * p  (uses scratch buffer in workspace)
    ratiod_linalg::axpy_matvec(epsilon, mass.inv_mass_dense.data(), p, q, n);
  }

  // Compute gradient + log_prob at new position (fused: single O(N) pass)
  // Uses pre-resolved function pointer to skip 15+ branch dispatch per leapfrog step
  std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
  ws.gradient_fn(ws.params_buf, data, layout, ws.grad_buf, &ws.logp_at(slot));
  std::memcpy(grad, ws.grad_buf.data(), n * sizeof(double));
  result.log_prob = ws.logp_at(slot);

  // Half step for momentum using new gradient
  ratiod_linalg::axpy(0.5 * epsilon, grad, p, n);

  // Divergence check
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }
  for (int i = 0; i < n; i++) {
    if (std::abs(q[i]) > 1e10 || !std::isfinite(q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

// Zero-allocation recursive tree builder
// Uses workspace slot indices instead of vector copies
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng) {

  int n = ws.n;
  TreeStats stats;

  if (depth == 0) {
    // Base case: single leapfrog step in-place on input_slot
    LeapfrogInPlaceResult lf = leapfrog_step_inplace(
      ws, input_slot, direction * epsilon, mass, data, layout
    );

    double H_new = nuts_compute_hamiltonian_fast(
      lf.log_prob, ws.p_at(input_slot), mass, n
    );
    double delta_H = H_new - H0;

    // Both endpoints are the same slot (single node)
    stats.left_slot = input_slot;
    stats.right_slot = input_slot;
    stats.proposal_slot = input_slot;
    stats.log_prob_proposal = lf.log_prob;

    // Multinomial weight: log(weight) = H0 - H_new (relative, Stan-style)
    stats.sum_log_weight = H0 - H_new;

    // Divergence check
    stats.divergent = lf.divergent || (delta_H > delta_max);
    stats.stop = stats.divergent;
    stats.n_valid = stats.divergent ? 0 : 1;

    // Acceptance statistic
    double accept_stat = std::min(1.0, std::exp(-delta_H));
    if (!std::isfinite(accept_stat)) accept_stat = 0.0;
    stats.sum_accept_prob = accept_stat;
    stats.n_leapfrog = 1;

    // Generalized U-turn: track rho, p_sharp, p at this leaf
    const double* p_ptr = ws.p_at(input_slot);
    stats.rho.resize(n);
    stats.p_beg.resize(n);
    stats.p_end.resize(n);
    stats.p_sharp_beg.resize(n);
    stats.p_sharp_end.resize(n);

    std::memcpy(stats.rho.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_beg.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_end.data(), p_ptr, n * sizeof(double));

    // p_sharp = M^{-1} * p  — use full mass matrix for U-turn criterion.
    // Dense mass captures correlation structure; using diagonal p_sharp would
    // make NUTS unable to detect turns in correlated directions, causing
    // trees to grow to max depth on correlated posteriors (slopes, BYM2, HSGP).
    mass.inv_mass_times_p(p_ptr, stats.p_sharp_beg.data());
    std::memcpy(stats.p_sharp_end.data(), stats.p_sharp_beg.data(), n * sizeof(double));

    return stats;
  }

  // Recursive case: build inner subtree
  TreeStats inner = build_tree_fast(
    ws, input_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  stats = std::move(inner);

  if (stats.stop) return stats;

  // Copy the appropriate endpoint to a fresh slot for outer start
  int start_slot = ws.alloc_slot();
  if (start_slot < 0) {
    stats.stop = true;
    return stats;
  }
  if (direction == 1) {
    ws.copy_node(start_slot, stats.right_slot);
  } else {
    ws.copy_node(start_slot, stats.left_slot);
  }

  // Build outer subtree from the copy
  TreeStats outer = build_tree_fast(
    ws, start_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  // Combine results
  stats.n_leapfrog += outer.n_leapfrog;
  stats.sum_accept_prob += outer.sum_accept_prob;
  stats.divergent = stats.divergent || outer.divergent;

  // Multinomial sampling
  double new_sum_log_weight = nuts_log_sum_exp(stats.sum_log_weight, outer.sum_log_weight);
  double accept_prob_outer = std::exp(outer.sum_log_weight - new_sum_log_weight);
  if (!std::isfinite(accept_prob_outer)) accept_prob_outer = 0.0;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  if (unif(rng) < accept_prob_outer) {
    stats.proposal_slot = outer.proposal_slot;
    stats.log_prob_proposal = outer.log_prob_proposal;
  }

  stats.sum_log_weight = new_sum_log_weight;
  stats.n_valid = stats.n_valid + outer.n_valid;

  // === SAVE BOUNDARY VALUES AS COPIES BEFORE MOVES ===
  // "init" = inner (built first), "final" = outer (extends from init)
  // These copies are needed because moves below invalidate the originals
  // Uses pre-allocated depth-indexed merge buffers (no per-merge heap allocation)
  double* p_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_P_INIT_END);
  double* p_sharp_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_PSHARP_INIT_END);
  double* rho_init = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_INIT);
  double* rho_check = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_CHECK);

  const double* src_p = (direction == 1) ? stats.p_end.data() : stats.p_beg.data();
  std::memcpy(p_init_end, src_p, n * sizeof(double));
  const double* src_ps = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();
  std::memcpy(p_sharp_init_end, src_ps, n * sizeof(double));
  std::memcpy(rho_init, stats.rho.data(), n * sizeof(double));

  // Pointers to final's boundary (safe: these outer members are NOT moved)
  const double* p_final_beg_ptr = (direction == 1) ? outer.p_beg.data() : outer.p_end.data();
  const double* p_sharp_final_beg_ptr = (direction == 1) ? outer.p_sharp_beg.data() : outer.p_sharp_end.data();

  // === UPDATE TREE ENDPOINTS (moves invalidate init's boundary refs) ===
  if (direction == 1) {
    stats.right_slot = outer.right_slot;
    stats.p_sharp_end = std::move(outer.p_sharp_end);
    stats.p_end = std::move(outer.p_end);
  } else {
    stats.left_slot = outer.left_slot;
    stats.p_sharp_beg = std::move(outer.p_sharp_beg);
    stats.p_beg = std::move(outer.p_beg);
  }

  // Combine rho = rho_init + rho_final
  for (int i = 0; i < n; i++) {
    stats.rho[i] = rho_init[i] + outer.rho[i];
  }

  // === GENERALIZED U-TURN CRITERION (Stan-style, 3 juncture checks) ===
  // Check 1: Full merged trajectory — merged endpoints vs merged rho
  bool persist = compute_criterion(stats.p_sharp_beg.data(), stats.p_sharp_end.data(),
                                   stats.rho.data(), n);

  // After update, far endpoints depend on direction:
  // direction == 1: init's far = stats.beg (left, unchanged), final's far = stats.end (right, updated)
  // direction == -1: init's far = stats.end (right, unchanged), final's far = stats.beg (left, updated)
  const double* init_far_psharp = (direction == 1) ? stats.p_sharp_beg.data() : stats.p_sharp_end.data();
  const double* final_far_psharp = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();

  // Check 2: Init subtree + seam from final (rho = rho_init + p_final_beg)
  for (int i = 0; i < n; i++) {
    rho_check[i] = rho_init[i] + p_final_beg_ptr[i];
  }
  persist &= compute_criterion(init_far_psharp, p_sharp_final_beg_ptr,
                                rho_check, n);

  // Check 3: Seam from init + final subtree (rho = rho_final + p_init_end)
  for (int i = 0; i < n; i++) {
    rho_check[i] = outer.rho[i] + p_init_end[i];
  }
  persist &= compute_criterion(p_sharp_init_end, final_far_psharp,
                                rho_check, n);

  stats.stop = outer.stop || !persist;

  return stats;
}

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h
) {
  int p = static_cast<int>(params.size());
  hessian.resize(static_cast<size_t>(p) * p);

  // Base gradient
  std::vector<double> grad_base(p);
  compute_gradient(params, data, layout, grad_base);

  // Perturb each parameter and compute column of Hessian
  std::vector<double> params_pert = params;
  std::vector<double> grad_pert(p);
  for (int i = 0; i < p; i++) {
    double orig = params_pert[i];
    double hi = std::max(h, h * std::abs(orig));  // relative step for large params
    params_pert[i] = orig + hi;
    compute_gradient(params_pert, data, layout, grad_pert);
    for (int j = 0; j < p; j++) {
      hessian[static_cast<size_t>(i) * p + j] = (grad_pert[j] - grad_base[j]) / hi;
    }
    params_pert[i] = orig;
  }

  // Symmetrize: H = 0.5 * (H + H^T)
  for (int i = 0; i < p; i++) {
    for (int j = i + 1; j < p; j++) {
      double avg = 0.5 * (hessian[static_cast<size_t>(i) * p + j] +
                          hessian[static_cast<size_t>(j) * p + i]);
      hessian[static_cast<size_t>(i) * p + j] = avg;
      hessian[static_cast<size_t>(j) * p + i] = avg;
    }
  }
}

bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
) {
  // Map to Eigen (column-major)
  Eigen::Map<const Eigen::MatrixXd> H_map(neg_hessian.data(), p, p);

  // Eigendecomposition (symmetric)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(H_map);
  if (eigen.info() != Eigen::Success) return false;

  const auto& lambdas = eigen.eigenvalues();
  const auto& Q = eigen.eigenvectors();

  // Apply SoftAbs: f(λ) = λ * coth(α * λ)
  // Properties: always positive, f(|λ|>>0) ≈ |λ|, f(0) → 1/α
  Eigen::VectorXd softabs_inv_eig(p);
  for (int i = 0; i < p; i++) {
    double lam = lambdas(i);
    double al = alpha * lam;
    double f;
    if (std::abs(al) > 20.0) {
      f = std::abs(lam);
    } else if (std::abs(al) < 1e-10) {
      f = 1.0 / alpha;
    } else {
      f = lam * std::cosh(al) / std::sinh(al);
    }
    f = std::max(f, 1e-6);  // floor to ensure positive definiteness
    softabs_inv_eig(i) = 1.0 / f;
  }

  // Reconstruct G^{-1} = Q diag(1/f(λ)) Q^T
  Eigen::MatrixXd G_inv_mat = Q * softabs_inv_eig.asDiagonal() * Q.transpose();

  // Cholesky of G^{-1}
  Eigen::LLT<Eigen::MatrixXd> llt(G_inv_mat);
  if (llt.info() != Eigen::Success) return false;
  Eigen::MatrixXd L_mat = llt.matrixL();

  // Copy to output (column-major)
  G_inv.resize(static_cast<size_t>(p) * p);
  L_G_inv.resize(static_cast<size_t>(p) * p);
  Eigen::Map<Eigen::MatrixXd>(G_inv.data(), p, p) = G_inv_mat;
  Eigen::Map<Eigen::MatrixXd>(L_G_inv.data(), p, p) = L_mat;

  return true;
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
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;
  bool use_nuts = (L == 0);

  HMCResultCpp result;
  result.n_params_stored = n_params;
  result.samples_flat.resize(static_cast<size_t>(n_sample) * n_params);
  result.log_prob.resize(n_sample);
  result.accept_prob.resize(n_sample);
  result.n_leapfrog.resize(n_sample, L);
  result.divergent.resize(n_sample, 0);
  result.treedepth.resize(n_sample, 0);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;
  result.chain_id = chain_id;
  result.n_max_treedepth = 0;

  std::mt19937 rng(seed + chain_id * 12345);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  std::vector<double> q = q_init;

  // For NUTS: fuse initial log_post + gradient into single O(N) pass
  std::vector<double> current_grad(n_params);
  double log_prob_current;
  if (use_nuts) {
    compute_gradient(q, data, layout, current_grad, &log_prob_current);
  } else {
    log_prob_current = compute_log_post(q, data, layout);
  }

  double epsilon = find_reasonable_epsilon(q, data, layout, rng);

  // Compute target_boost for challenging model combinations
  // MSGP and GP with temporal are particularly challenging
  double target_boost = 0.0;
  if (data.has_multiscale_gp) {
    target_boost += 0.10;  // MSGP models need higher target acceptance
    if (layout.has_temporal) {
      target_boost += 0.05;  // MSGP + temporal is even more challenging
    }
  } else if (data.spatial_type == SpatialType::GP) {
    target_boost += 0.05;  // GP models moderately challenging
    if (layout.has_temporal) {
      target_boost += 0.05;  // GP + temporal combination
    }
  }
  DualAveraging da(epsilon, n_params, target_boost);

  // For NUTS: model-adaptive target acceptance
  // Store in nuts_target_accept for reuse at mass window boundaries (avoids bug
  // where da.target_accept was reset to 0.80 at each window reset).
  double nuts_target_accept = 0.80;
  if (use_nuts) {
    if (adapt_delta > 0) {
      // User override
      nuts_target_accept = adapt_delta;
    } else {
      // Auto-select based on model complexity
      nuts_target_accept = 0.80;  // Stan default base

      // BYM2: high correlation between ICAR phi + unstructured theta
      if (data.spatial_type == SpatialType::BYM2) {
        nuts_target_accept = 0.90;
      }
      // ICAR: correlated spatial params, less severe than BYM2
      else if (data.spatial_type == SpatialType::ICAR) {
        nuts_target_accept = 0.85;
      }

      // Correlated random slopes add funnel geometry
      if (data.has_re_correlated_slopes) {
        nuts_target_accept = std::max(nuts_target_accept, 0.90);
      }

      nuts_target_accept = std::min(0.99, nuts_target_accept);
    }
    da.target_accept = nuts_target_accept;
  }

  // current_grad already computed above (fused with log_prob for NUTS)

  // Mass matrix adaptation
  // Resolve AUTO metric: select dense vs diagonal based on model complexity.
  // Dense mass helps when posterior has strong parameter correlations that
  // reparameterization alone cannot decorrelate (correlated slopes, BYM2, HSGP, GP, SVC).
  //
  // BENCHMARKED 2026-02-27: Attempted removing has_re, has_temporal, is_hsgp,
  // is_temporal_gp, has_tvc from needs_dense. Result: ALL models got WORSE
  // (row 2 pg+RE: 1.25s→4.2s, row 27 pg+TVC: 68→87s, row 38 nb+HSGP: 210→266s).
  // Dense metric is essential even for non-centered RE — NUTS compensates for
  // missing correlation info with longer trajectories, net negative.
  //
  // REBENCHMARKED 2026-02-28: After fixing TVC tau prior gradient mismatch
  // (PC prior in log_post vs Gamma(2,0.5) in gradient), TVC no longer needs
  // dense mass. DIAG: NB+TVC=1.1s/0div, PG+TVC=0.9s/0div. DENSE: NB+TVC
  // fails (OAS shrinkage=0.000 → near-singular matrix → 250/250 divergences).
  // TVC removed from needs_dense.
  constexpr int DENSE_MAX_PARAMS = 200;
  MassMatrixType effective_metric = metric_type;
  if (effective_metric == MassMatrixType::AUTO) {
    // Dense mass matrix needed when posterior has strong parameter correlations:
    // - Any RE model has residual sigma_re ↔ likelihood correlation that
    //   non-centered parameterization doesn't fully remove.
    // - Temporal, spatial, and other structured effects also benefit from dense.
    // - TVC uses diagonal: gradient fix (2026-02-28) eliminated the tau prior
    //   inconsistency that was causing poor step sizes requiring dense compensation.
    bool needs_dense = layout.has_re ||
                       layout.has_re_correlated_slopes ||
                       layout.has_temporal ||
                       layout.is_bym2 ||
                       layout.is_hsgp ||
                       layout.is_gp ||
                       layout.is_multiscale_gp ||
                       layout.is_temporal_gp ||
                       layout.has_multiscale_temporal ||
                       layout.has_svc ||
                       layout.has_spatiotemporal ||
                       layout.has_latent;
    effective_metric = needs_dense ? MassMatrixType::DENSE : MassMatrixType::DIAG;
    if (verbose) {
      REprintf("  [METRIC] auto -> %s (p=%d",
               effective_metric == MassMatrixType::DENSE ? "dense" : "diag", n_params);
      if (needs_dense) {
        if (layout.has_re) REprintf(", re");
        if (layout.has_re_correlated_slopes) REprintf(", correlated_slopes");
        if (layout.has_temporal) REprintf(", temporal");
        if (layout.is_bym2) REprintf(", bym2");
        if (layout.is_hsgp) REprintf(", hsgp");
        if (layout.is_gp) REprintf(", gp");
        if (layout.is_multiscale_gp) REprintf(", msgp");
        if (layout.is_temporal_gp) REprintf(", temporal_gp");
        if (layout.has_multiscale_temporal) REprintf(", ms_temporal");
        if (layout.has_svc) REprintf(", svc");
        if (layout.has_spatiotemporal) REprintf(", spatiotemporal");
        if (layout.has_latent) REprintf(", latent");
        if (layout.has_tvc) REprintf(", tvc");
      }
      REprintf(")\n");
    }
  }
  // Auto-downgrade dense to diagonal when n_params too large
  // Dense needs O(p^2) storage and O(p^3) Cholesky; also needs p+5 warmup samples
  if (effective_metric == MassMatrixType::DENSE && n_params > DENSE_MAX_PARAMS) {
    if (verbose) {
      REprintf("  [DENSE] n_params=%d > %d: auto-downgrading to diagonal mass matrix\n",
               n_params, DENSE_MAX_PARAMS);
    }
    effective_metric = MassMatrixType::DIAG;
  }

  DenseMassMatrix mass;
  try {
    mass.init(n_params, effective_metric);
  } catch (const std::bad_alloc&) {
    if (effective_metric == MassMatrixType::DENSE) {
      if (verbose) {
        REprintf("  [DENSE] Allocation failed for p=%d, falling back to diagonal\n", n_params);
      }
      effective_metric = MassMatrixType::DIAG;
      mass.init(n_params, effective_metric);
    } else {
      throw;
    }
  }

  WelfordStats mass_stats(n_params);              // Always track diagonal
  WelfordCovStats cov_stats(n_params);            // Only used when dense
  bool use_mass_matrix = false;

  // L-BFGS mass matrix adaptation (warmup-only)
  // Uses L-BFGS to learn curvature during warmup, then switches to standard HMC
  bool use_lbfgs = data.has_multiscale_gp &&
                   data.multiscale_gp_data.sampler == ratiod_gp::MSGPSampler::LBFGS;
  ratiod_gp::LBFGSState lbfgs_state;
  std::vector<double> q_prev, grad_prev;
  bool lbfgs_initialized = false;
  bool lbfgs_warmup_done = false;  // After warmup, use standard HMC
  if (use_lbfgs) {
    lbfgs_state = ratiod_gp::LBFGSState(10, n_params);
    q_prev.resize(n_params);
    grad_prev.resize(n_params);
  }

  // Stan-style expanding warmup windows for mass matrix adaptation
  // Phase 1: [0, init_buffer) - step size adaptation only
  // Phase 2: [init_buffer, n_warmup - term_buffer) - mass matrix adaptation
  //   Windows double in size: 25, 50, 100, 200, ...
  //   Last window extends to fill remaining space
  // Phase 3: [n_warmup - term_buffer, n_warmup) - final step size tuning
  int init_buffer = 75;
  int term_buffer = 50;
  int init_window = 25;

  // Dense mass models with many params need longer step size tuning
  // after the dense mass is set. Scale term_buffer proportionally.
  if (effective_metric == MassMatrixType::DENSE && n_params > 100) {
    term_buffer = 75;  // 50% more final adaptation
  }

  // Adjust for short warmup
  if (n_warmup < init_buffer + term_buffer + init_window) {
    init_buffer = std::max(1, n_warmup / 5);
    term_buffer = std::max(1, n_warmup / 10);
    init_window = std::max(1, n_warmup - init_buffer - term_buffer);
  }

  // Compute mass adaptation window endpoints
  std::vector<int> mass_window_ends;
  {
    int adapt_end = n_warmup - term_buffer;
    if (adapt_end <= init_buffer) {
      // No room for mass adaptation windows
      mass_window_ends.push_back(std::max(1, adapt_end));
    } else {
      int next_end = init_buffer + init_window;
      int win_size = init_window;
      while (next_end < adapt_end) {
        int next_win = 2 * win_size;
        if (next_end + next_win > adapt_end) {
          // Extend current window to fill remaining space
          mass_window_ends.push_back(adapt_end);
          break;
        }
        mass_window_ends.push_back(next_end);
        win_size = next_win;
        next_end += win_size;
      }
      if (mass_window_ends.empty() || mass_window_ends.back() < adapt_end) {
        mass_window_ends.push_back(adapt_end);
      }
    }
  }
  int next_window_idx = 0;

  // Pre-allocate NUTS workspace (zero-allocation tree building)
  NUTSWorkspace nuts_ws;
  std::vector<double> _nuts_p;              // Momentum sampling buffer
  std::vector<double> _nuts_q_proposal;     // Persistent proposal (survives tree resets)
  std::vector<double> _nuts_grad_proposal;  // Persistent proposal gradient
  if (use_nuts) {
    nuts_ws.init(n_params, max_treedepth);
    nuts_ws.gradient_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    _nuts_p.resize(n_params);
    _nuts_q_proposal.resize(n_params);
    _nuts_grad_proposal.resize(n_params);
  }

  int sample_idx = 0;
  int n_accept = 0;
  int n_divergent = 0;
  // Adaptive NUTS→fixed-L switching: monitor early sampling for max treedepth
  int nuts_probe_window = std::min(20, n_sample);  // Check first 20 sampling iterations
  int nuts_probe_maxd = 0;  // Count of maxd hits in probe window
  bool nuts_probing = use_nuts && (L == 0);  // Only probe when using NUTS by default

  // SoftAbs divergence retry: compute local Hessian-based metric on divergent
  // trajectories and retry. Only active for BYM2/ICAR + dense mass (auto) or
  // when explicitly forced on.
  bool use_softabs_retry = false;
  if (riemannian == 1) {
    use_softabs_retry = true;
  } else if (riemannian == -1) {
    // Auto: enable for BYM2/ICAR with dense mass
    use_softabs_retry = (mass.type == MassMatrixType::DENSE &&
                         (data.spatial_type == SpatialType::BYM2 ||
                          data.spatial_type == SpatialType::ICAR));
  }
  // Disable if not using NUTS (SoftAbs retry only makes sense with NUTS)
  if (!use_nuts) use_softabs_retry = false;
  int softabs_retries = 0;
  int softabs_successes = 0;
  constexpr int SOFTABS_MAX_RETRIES = 3;  // Up to 3 retry attempts per divergence

  // Persistent SoftAbs metric (improvement #2): once computed, reuse for
  // all subsequent trajectories. Initialized at warmup→sampling transition
  // (improvement #4) or on first divergence, whichever comes first.
  bool softabs_metric_active = false;
  DenseMassMatrix softabs_persistent_mass;
  double softabs_persistent_eps = 0.0;
  if (use_softabs_retry) {
    softabs_persistent_mass.init(n_params, MassMatrixType::DENSE);
  }

  int warmup_total_leapfrog = 0;  // TEMP: diagnostic counter

  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);
    // Check if we've reached a mass adaptation window boundary
    if (is_warmup && next_window_idx < (int)mass_window_ends.size() &&
        iter == mass_window_ends[next_window_idx]) {
      bool dense_covariance_set = false;  // Track if DENSE covariance (not just diagonal) succeeded this window
      // Dense mass matrix: try full covariance first
      // OAS shrinkage guarantees PD even when n < p, so we can lower the
      // threshold from n_params+5.  For large p the original threshold is
      // unreachable during warmup (e.g. p=159, need 164 but only get 125).
      // New threshold: min(p+5, max(50, p/2))  — for p=159 this is 79.
      int dense_threshold = std::min(n_params + 5,
                                     std::max(50, n_params / 2));
      if (mass.type == MassMatrixType::DENSE && cov_stats.n >= dense_threshold) {
        auto cov = cov_stats.covariance();
        if (mass.update_from_covariance(cov.data(), cov_stats.n)) {
          use_mass_matrix = true;
          dense_covariance_set = true;
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): dense mass SET (n=%d, p=%d, OAS shrinkage=%.3f)\n",
                     next_window_idx, iter, cov_stats.n, n_params,
                     cov_stats.shrinkage_intensity);
          }
        } else {
          // Cholesky failed — mass auto-degraded to DIAG, use diagonal stats
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): Cholesky FAILED (cov_stats.n=%d, p=%d)\n",
                     next_window_idx, iter, cov_stats.n, n_params);
          }
          if (mass_stats.n >= 10) {
            mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
            use_mass_matrix = true;
          }
        }
      } else if (mass.type == MassMatrixType::DENSE) {
        // Not enough samples for dense yet — use diagonal as interim
        if (verbose) {
          REprintf("  [DENSE] Window %d (iter %d): not enough samples (cov_stats.n=%d, need=%d)\n",
                   next_window_idx, iter, cov_stats.n, dense_threshold);
        }
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
      } else if (mass_stats.n >= 10) {
        // Diagonal path
        mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
        use_mass_matrix = true;
      }
      mass_stats.reset();
      // For dense: only reset cov_stats when full covariance was successfully
      // computed THIS window. Otherwise keep accumulating across windows until
      // we have enough samples. This prevents the chicken-and-egg problem
      // where short windows never collect enough.
      // NOTE: We use dense_covariance_set (not mass.adapted) because
      // set_diagonal() also sets adapted=true, which would incorrectly
      // trigger a reset when we're still building up covariance samples.
      if (mass.type != MassMatrixType::DENSE || dense_covariance_set) {
        cov_stats.reset();
      }
      // Re-initialize step size with current mass matrix (A3)
      // Use dense-aware version when dense mass is adapted, so the step size
      // is calibrated for the rotated phase space (not just the diagonal).
      if (use_mass_matrix && mass.type == MassMatrixType::DENSE && mass.adapted) {
        epsilon = find_reasonable_epsilon_dense(q, data, layout, rng, mass);
      } else if (use_mass_matrix) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      } else {
        epsilon = find_reasonable_epsilon(q, data, layout, rng);
      }
      da = DualAveraging(epsilon, n_params, target_boost);
      if (use_nuts) da.target_accept = nuts_target_accept;  // Preserve model-adaptive target
      next_window_idx++;
    }

    // L-BFGS: transition from L-BFGS to standard HMC at end of warmup
    // Extract diagonal mass matrix from learned curvature
    if (use_lbfgs && !lbfgs_warmup_done && iter == n_warmup - 1 && lbfgs_initialized) {
      // Use gamma from L-BFGS as uniform scaling for mass matrix
      // gamma = (s^T y) / (y^T y) approximates average inverse Hessian scaling
      double gamma = lbfgs_state.gamma;
      if (gamma > 0.01 && gamma < 100.0) {
        // Set inv_mass = gamma * I (larger gamma = larger variance = larger step in that direction)
        std::vector<double> inv_m(n_params, gamma);
        std::vector<double> sqrt_m(n_params, 1.0 / std::sqrt(gamma));
        mass.set_diagonal(inv_m, sqrt_m);
        use_mass_matrix = true;
      }
      lbfgs_warmup_done = true;
    }

    // =========================================================================
    // NUTS or fixed-trajectory HMC
    // =========================================================================
    double alpha = 0.0;
    bool divergent = false;
    int iter_n_leapfrog = L;
    int iter_treedepth = 0;

    if (use_nuts && !(use_lbfgs && !lbfgs_warmup_done)) {
      // -----------------------------------------------------------------
      // NUTS: No-U-Turn Sampler (optimized zero-allocation path)
      // -----------------------------------------------------------------

      auto& p = _nuts_p;

      // Step size jitter (improvement #5): ±20% random noise per trajectory
      // Prevents systematic step-size resonances that cause divergences.
      // Only during post-warmup sampling — warmup needs stable epsilon for adaptation.
      double eps_iter = epsilon;
      if (!is_warmup) {
        double jitter = 1.0 + 0.2 * (2.0 * unif(rng) - 1.0);  // U[0.8, 1.2]
        eps_iter = epsilon * jitter;
      }

      // Sample momentum p ~ N(0, M) where M = C^{-1}
      mass.sample_momentum(p.data(), rng);

      // Initial Hamiltonian (pointer-based, no vector overhead)
      double H0 = nuts_compute_hamiltonian_fast(
        log_prob_current, p.data(), mass, n_params
      );
      double delta_max = 1000.0;

      // Load current state into workspace persistent slots
      nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);
      nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);

      // Initialize persistent proposal buffers (pre-allocated, no per-iter malloc)
      auto& q_proposal_data = _nuts_q_proposal;
      auto& grad_proposal_data = _nuts_grad_proposal;
      std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
      std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
      double log_prob_proposal = log_prob_current;
      double sum_log_weight = 0.0;  // Relative weights: log(exp(H0 - H0)) = 0

      int total_leapfrog = 0;
      double sum_accept_prob = 0.0;
      divergent = false;

      // Generalized U-turn tracking at top level (Stan-style)
      // rho = total momentum sum. rho_bck/rho_fwd = halves for 3-juncture checks.
      // At each iteration the entire old trajectory becomes one half,
      // the new subtree becomes the other half (Stan's approach).
      // Uses pre-allocated workspace vectors (no per-iteration heap allocation).
      auto& rho = nuts_ws.iter_rho;
      std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
      auto& rho_bck = nuts_ws.iter_rho_bck;
      auto& rho_fwd = nuts_ws.iter_rho_fwd;
      std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
      std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);

      // p_sharp = M^{-1} * p at initial point — full mass for correct U-turn geometry
      auto& p_sharp_init = nuts_ws.iter_p_sharp_init;
      mass.inv_mass_times_p(p.data(), p_sharp_init.data());

      // Boundary momenta: _end = far endpoint, _beg = origin-facing boundary
      // Stan naming: bck_end=bck_bck, bck_beg=bck_fwd, fwd_beg=fwd_bck, fwd_end=fwd_fwd
      auto& p_fwd_beg = nuts_ws.iter_p_fwd_beg;
      auto& p_fwd_end = nuts_ws.iter_p_fwd_end;
      auto& p_bck_beg = nuts_ws.iter_p_bck_beg;
      auto& p_bck_end = nuts_ws.iter_p_bck_end;
      std::memcpy(p_fwd_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_fwd_end.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_end.data(), p.data(), n_params * sizeof(double));
      auto& p_sharp_fwd_beg = nuts_ws.iter_p_sharp_fwd_beg;
      auto& p_sharp_fwd_end = nuts_ws.iter_p_sharp_fwd_end;
      auto& p_sharp_bck_beg = nuts_ws.iter_p_sharp_bck_beg;
      auto& p_sharp_bck_end = nuts_ws.iter_p_sharp_bck_end;
      std::memcpy(p_sharp_fwd_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_fwd_end.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_end.data(), p_sharp_init.data(), n_params * sizeof(double));

      // Build tree until U-turn or max depth
      for (int j = 0; j < max_treedepth; j++) {
        std::uniform_int_distribution<int> dir_dist(0, 1);
        int direction = 2 * dir_dist(rng) - 1;

        nuts_ws.reset_tree();

        int start_slot = nuts_ws.alloc_slot();
        if (start_slot < 0) break;
        if (direction == 1) {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
        } else {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
        }

        // Stan: relabel halves before building subtree
        // Entire old trajectory becomes one half; new subtree is the other
        if (direction == 1) {
          // Extending forward: old trajectory → backward half
          std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
        } else {
          // Extending backward: old trajectory → forward half
          std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
        }

        TreeStats subtree = build_tree_fast(
          nuts_ws, start_slot, direction, j,
          eps_iter, mass, H0, delta_max,
          data, layout, rng
        );

        total_leapfrog += subtree.n_leapfrog;
        sum_accept_prob += subtree.sum_accept_prob;

        if (subtree.divergent) {
          divergent = true;
        }

        if (!subtree.stop) {
          // Multinomial acceptance
          double log_sum_weight_subtree = subtree.sum_log_weight;
          double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);

          double accept_prob_subtree;
          if (log_sum_weight_subtree > new_sum_log_weight) {
            accept_prob_subtree = 1.0;
          } else {
            accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
          }
          if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

          std::uniform_real_distribution<double> unif01(0.0, 1.0);
          if (unif01(rng) < accept_prob_subtree) {
            std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            log_prob_proposal = subtree.log_prob_proposal;
          }

          sum_log_weight = new_sum_log_weight;
        }

        // Update direction endpoints and rho half from subtree
        // Use memcpy instead of std::move to preserve pre-allocated buffers
        if (direction == 1) {
          nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
          std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        } else {
          nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
          std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        }

        // Combine rho = rho_bck + rho_fwd
        for (int i = 0; i < n_params; i++) {
          rho[i] = rho_bck[i] + rho_fwd[i];
        }

        iter_treedepth = j + 1;

        // Generalized U-turn check at top level (3 junctures)
        if (subtree.stop) break;

        // Check 1: Full trajectory — far endpoints vs total rho
        bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                         rho.data(), n_params);

        // Check 2: Backward half + seam from forward (rho = rho_bck + p_fwd_beg)
        auto& rho_seam = nuts_ws.iter_rho_seam;
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_bck[i] + p_fwd_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                      rho_seam.data(), n_params);

        // Check 3: Seam from backward + forward half (rho = rho_fwd + p_bck_beg)
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_fwd[i] + p_bck_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                      rho_seam.data(), n_params);

        if (!persist) break;
      }

      // SoftAbs divergence retry (improvements #1, #2): if trajectory diverged,
      // compute local Hessian-based metric and retry up to SOFTABS_MAX_RETRIES
      // times, halving step size each attempt. On first successful metric
      // computation, persist it for all subsequent trajectories.
      if (divergent && !is_warmup && use_softabs_retry) {
        softabs_retries++;

        // Compute fresh Hessian at current position (p+1 gradient evals)
        std::vector<double> hessian_buf;
        compute_hessian_finite_diff(q, data, layout, hessian_buf);
        for (auto& v : hessian_buf) v = -v;  // Negate: -H = curvature

        std::vector<double> G_inv_buf, L_G_inv_buf;
        bool metric_ok = compute_softabs_metric(
          hessian_buf, n_params, 1.0, G_inv_buf, L_G_inv_buf
        );

        if (metric_ok) {
          // Update persistent SoftAbs metric for retry use only (improvement #2).
          // Do NOT override main mass/epsilon — warmup-adapted values work better
          // for general trajectories. SoftAbs is rescue-only.
          softabs_persistent_mass.set_from_metric(G_inv_buf, L_G_inv_buf);
          double eps_base = find_reasonable_epsilon_dense(
            q, data, layout, rng, softabs_persistent_mass);
          softabs_persistent_eps = eps_base;
          softabs_metric_active = true;

          // Multiple retry attempts (improvement #1): try up to 3 times
          // with halving step size each attempt
          for (int retry_attempt = 0; retry_attempt < SOFTABS_MAX_RETRIES; retry_attempt++) {
            double eps_retry = eps_base * std::pow(0.5, retry_attempt);

            // Sample new momentum and re-run NUTS trajectory
            softabs_persistent_mass.sample_momentum(p.data(), rng);
            double H0_retry = nuts_compute_hamiltonian_fast(
              log_prob_current, p.data(), softabs_persistent_mass, n_params
            );

            // Load current state into workspace
            nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                              q.data(), p.data(), current_grad.data(), log_prob_current);
            nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                              q.data(), p.data(), current_grad.data(), log_prob_current);

            std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
            log_prob_proposal = log_prob_current;
            sum_log_weight = 0.0;
            total_leapfrog = 0;
            sum_accept_prob = 0.0;
            bool retry_divergent = false;

            // Full NUTS tree with SoftAbs metric + 3-juncture U-turn
            std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
            std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
            std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);
            softabs_persistent_mass.inv_mass_times_p(p.data(), p_sharp_init.data());
            std::copy(p.begin(), p.end(), p_fwd_beg.begin());
            std::copy(p.begin(), p.end(), p_fwd_end.begin());
            std::copy(p.begin(), p.end(), p_bck_beg.begin());
            std::copy(p.begin(), p.end(), p_bck_end.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_fwd_beg.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_fwd_end.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_bck_beg.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_bck_end.begin());

            int retry_treedepth = 0;
            for (int j = 0; j < max_treedepth; j++) {
              std::uniform_int_distribution<int> dir_dist(0, 1);
              int direction = 2 * dir_dist(rng) - 1;

              nuts_ws.reset_tree();
              int start_slot = nuts_ws.alloc_slot();
              if (start_slot < 0) break;
              if (direction == 1) {
                nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
              } else {
                nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
              }

              if (direction == 1) {
                std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
                std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
              } else {
                std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
              }

              TreeStats subtree = build_tree_fast(
                nuts_ws, start_slot, direction, j,
                eps_retry, softabs_persistent_mass, H0_retry, 1000.0,
                data, layout, rng
              );

              total_leapfrog += subtree.n_leapfrog;
              sum_accept_prob += subtree.sum_accept_prob;
              if (subtree.divergent) retry_divergent = true;

              if (!subtree.stop) {
                double log_sum_weight_subtree = subtree.sum_log_weight;
                double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);
                double accept_prob_subtree;
                if (log_sum_weight_subtree > new_sum_log_weight) {
                  accept_prob_subtree = 1.0;
                } else {
                  accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
                }
                if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

                std::uniform_real_distribution<double> unif01(0.0, 1.0);
                if (unif01(rng) < accept_prob_subtree) {
                  std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                              n_params * sizeof(double));
                  std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                              n_params * sizeof(double));
                  log_prob_proposal = subtree.log_prob_proposal;
                }
                sum_log_weight = new_sum_log_weight;
              }

              if (direction == 1) {
                nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
                std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
              } else {
                nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
                std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
                std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
                std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
              }

              for (int i = 0; i < n_params; i++) {
                rho[i] = rho_bck[i] + rho_fwd[i];
              }
              retry_treedepth = j + 1;

              if (subtree.stop) break;

              bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                               rho.data(), n_params);
              auto& rho_seam_retry = nuts_ws.iter_rho_seam;
              for (int i = 0; i < n_params; i++) {
                rho_seam_retry[i] = rho_bck[i] + p_fwd_beg[i];
              }
              persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                            rho_seam_retry.data(), n_params);
              for (int i = 0; i < n_params; i++) {
                rho_seam_retry[i] = rho_fwd[i] + p_bck_beg[i];
              }
              persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                            rho_seam_retry.data(), n_params);
              if (!persist) break;
            }

            // If retry succeeded (no divergence), accept and stop retrying
            if (!retry_divergent) {
              divergent = false;
              iter_treedepth = retry_treedepth;
              softabs_successes++;
              alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
              iter_n_leapfrog = total_leapfrog;
              break;  // Success — stop retry loop
            }
            // Otherwise: try again with halved step size (next iteration)
          }  // end retry_attempt loop

          // If all retries failed, update stats from last attempt
          if (divergent) {
            alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
            iter_n_leapfrog = total_leapfrog;
          }
        }
        // else: metric computation failed, keep original divergent result
      }

      // Accept proposal: copy from persistent proposal buffers (memcpy, no alloc)
      std::memcpy(q.data(), q_proposal_data.data(), n_params * sizeof(double));
      std::memcpy(current_grad.data(), grad_proposal_data.data(), n_params * sizeof(double));
      log_prob_current = log_prob_proposal;
      n_accept++;

      // Average acceptance statistic for dual averaging
      alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
      iter_n_leapfrog = total_leapfrog;

      if (divergent) n_divergent++;
      if (iter_treedepth >= max_treedepth) result.n_max_treedepth++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);
        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
        }
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();
          if (verbose) {
            REprintf("  [DENSE] Warmup done: epsilon=%.6f, mass.type=%s, mass.adapted=%d\n",
                     epsilon, (mass.type == MassMatrixType::DENSE ? "DENSE" : "DIAG"),
                     (int)mass.adapted);
          }
          // Proactive SoftAbs at warmup→sampling transition (improvement #4):
          // Pre-compute SoftAbs metric so it's ready for retry attempts.
          // Do NOT override main mass/epsilon — warmup-adapted values are better
          // for general sampling. SoftAbs is only used as rescue on divergences.
          if (use_softabs_retry && !softabs_metric_active) {
            std::vector<double> hessian_warmup_end;
            compute_hessian_finite_diff(q, data, layout, hessian_warmup_end);
            for (auto& v : hessian_warmup_end) v = -v;

            std::vector<double> G_inv_init, L_G_inv_init;
            if (compute_softabs_metric(hessian_warmup_end, n_params, 1.0,
                                       G_inv_init, L_G_inv_init)) {
              softabs_persistent_mass.set_from_metric(G_inv_init, L_G_inv_init);
              softabs_persistent_eps = find_reasonable_epsilon_dense(
                q, data, layout, rng, softabs_persistent_mass);
              softabs_metric_active = true;
              // Note: main mass and epsilon are NOT overridden
              if (verbose) {
                REprintf("  [SoftAbs] Proactive metric pre-computed at warmup end: retry_eps=%.6f\n",
                         softabs_persistent_eps);
              }
            }
          }
        }
        // Print tree depth for last 10 warmup iterations
        if (verbose && iter >= n_warmup - 10) {
          REprintf("  [DENSE] warmup iter %d: treedepth=%d, epsilon=%.6f\n",
                   iter, iter_treedepth, epsilon);
        }
      }

      // Adaptive NUTS probe: warn if most early iterations hit max treedepth
      // (Stan's approach: warn but keep NUTS running — truncated NUTS picks
      // from up to 2^depth candidates, far better than HMC(L=10) with tiny epsilon)
      if (nuts_probing && !is_warmup && sample_idx < nuts_probe_window) {
        if (iter_treedepth >= max_treedepth) nuts_probe_maxd++;
        if (sample_idx == nuts_probe_window - 1) {
          nuts_probing = false;  // Probe window complete
          if (nuts_probe_maxd >= (nuts_probe_window * 8 + 9) / 10) {
            result.n_max_treedepth += 0;  // Already counted above
            if (verbose) {
              REprintf("  [NUTS] %d/%d initial sampling iterations hit max treedepth (%d). "
                       "Consider increasing max_treedepth or reparameterizing.\n",
                       nuts_probe_maxd, nuts_probe_window, max_treedepth);
            }
          }
        }
      }
    } else {
      // -----------------------------------------------------------------
      // Fixed-trajectory HMC (original code)
      // -----------------------------------------------------------------

      // Sample momentum and compute kinetic energy
      std::vector<double> p(n_params);
      double kinetic_current = 0.0;
      double H_current;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS: Sample p ~ N(0, B) where B ≈ 1/gamma * I (warmup only)
        std::vector<double> sqrt_diag = lbfgs_state.get_sqrt_B_diag();
        if ((int)sqrt_diag.size() == n_params) {
          for (int i = 0; i < n_params; i++) {
            p[i] = normal(rng) * sqrt_diag[i];
          }
          kinetic_current = lbfgs_state.kinetic_energy(p);
          H_current = -log_prob_current + kinetic_current;
        } else {
          mass.sample_momentum(p.data(), rng);
          kinetic_current = mass.kinetic_energy(p.data());
          H_current = -log_prob_current + kinetic_current;
        }
      } else {
        mass.sample_momentum(p.data(), rng);
        kinetic_current = mass.kinetic_energy(p.data());
        H_current = -log_prob_current + kinetic_current;
      }

      // Leapfrog integration
      std::vector<double> q_prop = q;
      std::vector<double> p_prop = p;

      // Determine effective L for this iteration
      int L_eff = L;
      if (use_nuts && use_lbfgs && !lbfgs_warmup_done) {
        // During L-BFGS warmup with NUTS mode, use fixed L=20
        L_eff = 20;
      }

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS leapfrog
        std::vector<double> grad(n_params);
        compute_gradient(q_prop, data, layout, grad);

        for (int l = 0; l < L_eff; l++) {
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          std::vector<double> Hp(n_params);
          lbfgs_state.multiply_H(p_prop, Hp);
          for (int i = 0; i < n_params; i++) {
            q_prop[i] += epsilon * Hp[i];
            if (!std::isfinite(q_prop[i])) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
          compute_gradient(q_prop, data, layout, grad);
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          for (int i = 0; i < n_params; i++) {
            if (!std::isfinite(p_prop[i]) || std::abs(p_prop[i]) > 1e10) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
        }
      } else {
        // Standard leapfrog
        for (int l = 0; l < L_eff; l++) {
          LeapfrogResult lf;
          if (use_mass_matrix) {
            lf = leapfrog_step_mass(q_prop, p_prop, epsilon, mass.inv_mass_diag, data, layout);
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
      }

      // Compute proposed Hamiltonian
      double log_prob_prop = compute_log_post(q_prop, data, layout);
      double kinetic_prop = 0.0;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        kinetic_prop = lbfgs_state.kinetic_energy(p_prop);
      } else {
        kinetic_prop = mass.kinetic_energy(p_prop.data());
      }
      double H_prop = -log_prob_prop + kinetic_prop;

      // Metropolis accept/reject
      alpha = std::min(1.0, std::exp(H_current - H_prop));
      if (!std::isfinite(alpha)) alpha = 0.0;

      std::uniform_real_distribution<double> unif01(0.0, 1.0);
      bool accepted = (unif01(rng) < alpha) && !divergent;
      if (accepted) {
        q = q_prop;
        log_prob_current = log_prob_prop;
        n_accept++;
        // Update cached gradient for transition to NUTS after L-BFGS warmup
        if (use_nuts) {
          compute_gradient(q, data, layout, current_grad);
        }
      }
      if (divergent) n_divergent++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);
        // Only collect mass stats during mass adaptation phase (A5)
        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
        }
        // On last warmup iteration, use averaged step size for sampling (A1)
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();
        }
      }

      // L-BFGS update: collect (s, y) pairs from accepted samples (warmup only)
      if (use_lbfgs && !lbfgs_warmup_done) {
        std::vector<double> grad_current(n_params);
        compute_gradient(q, data, layout, grad_current);

        if (!lbfgs_initialized) {
          q_prev = q;
          grad_prev = grad_current;
          lbfgs_initialized = true;
        } else if (accepted) {
          std::vector<double> s(n_params), y(n_params);
          for (int i = 0; i < n_params; i++) {
            s[i] = q[i] - q_prev[i];
            y[i] = grad_current[i] - grad_prev[i];
          }
          lbfgs_state.add_pair(s, y);
          q_prev = q;
          grad_prev = grad_current;
        }
      }

      iter_n_leapfrog = L_eff;
    }  // end fixed-trajectory HMC

    // Store sample (flat row-major storage, single memcpy)
    if (!is_warmup) {
      std::memcpy(result.sample_row(sample_idx), q.data(),
                  n_params * sizeof(double));
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;
      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // TEMP: diagnostic
    }

    // Note: verbose output disabled in parallel - not thread-safe
    // Progress will be reported after parallel region
  }

  result.epsilon = da.final_epsilon();

  // TEMP: diagnostic - warmup leapfrog count
  {
    int sampling_total_lf = 0;
    for (int i = 0; i < result.n_sample; i++) sampling_total_lf += result.n_leapfrog[i];
    REprintf("  [DIAG] Chain %d: warmup_LF=%d, sampling_LF=%d, total_LF=%d, epsilon=%.4f\n",
             chain_id + 1, warmup_total_leapfrog, sampling_total_lf,
             warmup_total_leapfrog + sampling_total_lf, result.epsilon);
  }

  if (verbose && (softabs_retries > 0 || softabs_metric_active)) {
    REprintf("  [SoftAbs] Chain %d: metric=%s, %d divergent retried (up to %d attempts), %d resolved (%d remained)\n",
             chain_id + 1,
             softabs_metric_active ? "active" : "inactive",
             softabs_retries, SOFTABS_MAX_RETRIES, softabs_successes,
             softabs_retries - softabs_successes);
  }

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
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
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
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  ParamLayout layout = compute_param_layout(data);
  int n_params = layout.total_params;

  // Use pure C++ containers in parallel region
  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Thread-safe autodiff: Each chain creates its own tape via TapeScope (RAII).
  // All gradient modes (N, A, A_t, H) are now thread-safe and can run in parallel.
  // The old global tape limitation has been removed.

#ifdef _OPENMP
  if (n_chains > 1) {
    // Run chains in parallel - all gradient modes are now thread-safe
    #pragma omp parallel for schedule(static) num_threads(n_chains)
    for (int c = 0; c < n_chains; c++) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init, data, layout,
        n_iter, n_warmup, L, c, seed, false, max_treedepth, metric_type, adapt_delta, riemannian
      );
    }
  } else {
    // Single chain - run sequentially with verbose output
    cpp_results[0] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );
  }
#else
  // Sequential fallback when OpenMP not available
  for (int c = 0; c < n_chains; c++) {
    cpp_results[c] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, c, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );
  }
#endif

  // Convert to R objects outside parallel region (single-threaded)
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);

    if (verbose && n_chains > 1) {
      // Print summary if we ran in parallel (verbose was disabled during parallel run)
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
    Rcpp::NumericVector y_num_cont,
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
    Rcpp::List tvc_params,  // Time-varying coefficients
    Rcpp::List svc_params,  // Spatially-varying coefficients
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    std::string gradient_mode_str = "auto",
    int max_treedepth = 10,
    std::string metric_str = "auto",
    double adapt_delta = -1.0,
    int riemannian = -1
) {
  using namespace ratiod_hmc;

  // Set global gradient mode from R parameter
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);

  // Parse metric type
  MassMatrixType metric_type = parse_metric_type(metric_str);

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

  // One-inflation parameters (for OI-binomial and ZOIB)
  Rcpp::NumericMatrix X_oi;
  SEXP xoi_sexp = zi_params["X_oi"];
  if (!Rf_isNull(xoi_sexp) && Rf_isMatrix(xoi_sexp)) {
    X_oi = Rcpp::as<Rcpp::NumericMatrix>(xoi_sexp);
  } else {
    // Create empty dummy matrix
    X_oi = Rcpp::NumericMatrix(1, 1);
    X_oi(0, 0) = 1.0;
  }
  int p_oi = 0;
  SEXP p_oi_sexp = zi_params["p_oi"];
  if (!Rf_isNull(p_oi_sexp)) {
    p_oi = Rcpp::as<int>(p_oi_sexp);
  }
  double oi_prior_sd = zi_prior_sd;  // Default to same as ZI
  SEXP oi_prior_sd_sexp = zi_params["oi_prior_sd"];
  if (!Rf_isNull(oi_prior_sd_sexp)) {
    oi_prior_sd = Rcpp::as<double>(oi_prior_sd_sexp);
  }

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
  data.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
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

  // RE parameterization: 0 = centered, 1 = non-centered
  // Non-centered uses z ~ N(0,1) prior, centered uses re ~ N(0, sigma^2)
  data.re_parameterization = Rcpp::as<int>(re_params["parameterization"]);

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

    // Build contiguous flat array: re_group_multi_flat[t * N + i]
    data.re_group_multi_flat.resize(n_re_terms * data.N);
    for (int t = 0; t < n_re_terms; t++) {
      for (int i = 0; i < data.N; i++) {
        data.re_group_multi_flat[t * data.N + i] = data.re_group_multi[t][i];
      }
    }
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
  } else if (model_type_str == "poisson_gamma") {
    data.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.model_type = ModelType::POISSON_GAMMA;  // fallback
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
  } else if (temporal_type_str == "gp") {
    data.temporal_type = TemporalType::GP;

    // GP-specific parameters
    data.has_temporal_gp = true;
    data.temporal_gp_data.n_obs = data.n_times;  // Use n_times, not N (total obs)
    data.temporal_gp_data.n_groups = n_temporal_groups;
    data.temporal_gp_data.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
    data.temporal_gp_data.group_index = temporal_group_idx;

    // Parse covariance type
    std::string cov_type_str = Rcpp::as<std::string>(temporal_params["cov_type"]);
    data.temporal_gp_data.cov_type = ratiod_temporal_gp::parse_temporal_cov_type(cov_type_str);
    data.temporal_gp_data.nu = Rcpp::as<double>(temporal_params["nu"]);
    data.temporal_gp_data.period = Rcpp::as<double>(temporal_params["period"]);
    data.temporal_gp_data.shared = temporal_shared;

    // GP priors
    data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
    data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
    data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
    data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);
  } else {
    data.temporal_type = TemporalType::NONE;
    data.has_temporal_gp = false;
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
  // Use explicit p_zi from R (not X_zi.ncol()) because OI-only models
  // pass a 1-column placeholder X_zi but p_zi=0
  {
    SEXP p_zi_sexp = zi_params["p_zi"];
    data.p_zi = (!Rf_isNull(p_zi_sexp)) ? Rcpp::as<int>(p_zi_sexp) : X_zi.ncol();
  }
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // One-inflation structure (for OI-binomial and ZOIB)
  data.p_oi = p_oi;
  data.oi_prior_sd = oi_prior_sd;
  if (p_oi > 0) {
    data.X_oi_flat.resize(data.N * data.p_oi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_oi; j++) {
        data.X_oi_flat[i * data.p_oi + j] = X_oi(i, j);
      }
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
  data.has_hsgp = false;

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

    // Parse ST type (accept both R-side "I"/"IV" and legacy "type_i"/"type_iv")
    if (st_type_str == "I" || st_type_str == "type_i") {
      data.spatiotemporal_data.type = STType::TYPE_I;
    } else if (st_type_str == "II" || st_type_str == "type_ii") {
      data.spatiotemporal_data.type = STType::TYPE_II;
    } else if (st_type_str == "III" || st_type_str == "type_iii") {
      data.spatiotemporal_data.type = STType::TYPE_III;
    } else if (st_type_str == "IV" || st_type_str == "type_iv") {
      data.spatiotemporal_data.type = STType::TYPE_IV;
    } else if (st_type_str == "separable") {
      data.spatiotemporal_data.type = STType::SEPARABLE;
    } else if (st_type_str == "nonsep_gp") {
      data.spatiotemporal_data.type = STType::NONSEP_GP;
    } else {
      Rcpp::stop("Unknown spatiotemporal type: '%s'. Expected one of: I, II, III, IV, separable, nonsep_gp",
                 st_type_str.c_str());
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

  // TVC (Temporally-Varying Coefficients) parameters
  bool has_tvc = tvc_params.size() > 0 && Rcpp::as<bool>(tvc_params["has_tvc"]);
  data.has_tvc = has_tvc;
  if (has_tvc) {
    // Extract TVC parameters (eager deep copies to prevent R GC issues)
    int tvc_n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
    int tvc_n_times = Rcpp::as<int>(tvc_params["n_times"]);
    int tvc_n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
    std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params["structure"]);
    bool tvc_shared = Rcpp::as<bool>(tvc_params["shared"]);
    bool tvc_cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
    std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
    std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
    std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
    std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
    double tvc_tau_shape = Rcpp::as<double>(tvc_params["tau_shape"]);
    double tvc_tau_rate = Rcpp::as<double>(tvc_params["tau_rate"]);

    // Populate TVC data structure
    data.tvc_data.n_obs = data.N;
    data.tvc_data.n_tvc = tvc_n_tvc;
    data.tvc_data.n_times = tvc_n_times;
    data.tvc_data.n_groups = tvc_n_groups;
    data.tvc_data.shared = tvc_shared;
    data.tvc_data.cyclic = tvc_cyclic;
    data.tvc_data.tvc_indices = tvc_indices;
    data.tvc_data.time_index = tvc_time_index;
    data.tvc_data.group_index = tvc_group_index;
    data.tvc_data.X_tvc = tvc_X_tvc;

    // Parse TVC temporal structure
    if (tvc_structure_str == "rw1") {
      data.tvc_data.structure = ratiod_temporal::TemporalType::RW1;
    } else if (tvc_structure_str == "rw2") {
      data.tvc_data.structure = ratiod_temporal::TemporalType::RW2;
    } else if (tvc_structure_str == "ar1") {
      data.tvc_data.structure = ratiod_temporal::TemporalType::AR1;
    } else if (tvc_structure_str == "iid") {
      data.tvc_data.structure = ratiod_temporal::TemporalType::IID;
    } else {
      data.tvc_data.structure = ratiod_temporal::TemporalType::RW1;  // Default
    }

    // Prior parameters
    data.tvc_tau_shape = tvc_tau_shape;
    data.tvc_tau_rate = tvc_tau_rate;

    // Pre-allocate gradient workspace buffers (avoids per-call heap allocation)
    data.tvc_data.init_workspace();
  } else {
    data.tvc_data.n_tvc = 0;
    data.tvc_data.n_times = 0;
    data.tvc_data.n_groups = 1;
  }

  // SVC (Spatially-Varying Coefficients) parameters
  bool has_svc = svc_params.size() > 0 && Rcpp::as<bool>(svc_params["has_svc"]);
  data.has_svc = has_svc;
  if (has_svc) {
    // Extract SVC parameters (eager deep copies to prevent R GC issues)
    int svc_n_svc = Rcpp::as<int>(svc_params["n_svc"]);
    int svc_nn = Rcpp::as<int>(svc_params["nn"]);
    bool svc_shared = Rcpp::as<bool>(svc_params["shared"]);
    std::string svc_cov_type_str = Rcpp::as<std::string>(svc_params["cov_type"]);
    std::vector<double> svc_coords = Rcpp::as<std::vector<double>>(svc_params["coords"]);
    std::vector<int> svc_indices = Rcpp::as<std::vector<int>>(svc_params["svc_indices"]);
    std::vector<double> svc_X_svc = Rcpp::as<std::vector<double>>(svc_params["X_svc"]);
    std::vector<int> svc_nn_idx = Rcpp::as<std::vector<int>>(svc_params["nn_idx"]);
    std::vector<double> svc_nn_dist = Rcpp::as<std::vector<double>>(svc_params["nn_dist"]);
    std::vector<int> svc_nn_order = Rcpp::as<std::vector<int>>(svc_params["nn_order"]);
    std::vector<int> svc_nn_order_inv = Rcpp::as<std::vector<int>>(svc_params["nn_order_inv"]);
    double svc_sigma2_scale = Rcpp::as<double>(svc_params["sigma2_prior_scale"]);
    double svc_phi_lower = Rcpp::as<double>(svc_params["phi_prior_lower"]);
    double svc_phi_upper = Rcpp::as<double>(svc_params["phi_prior_upper"]);

    // Populate SVC data structure
    data.svc_data.n_obs = data.N;
    data.svc_data.n_svc = svc_n_svc;
    data.svc_data.nn = svc_nn;
    data.svc_data.shared = svc_shared;
    data.svc_data.coords = svc_coords;
    data.svc_data.svc_indices = svc_indices;
    data.svc_data.X_svc = svc_X_svc;
    data.svc_data.nn_idx = svc_nn_idx;
    data.svc_data.nn_dist = svc_nn_dist;
    data.svc_data.nn_order = svc_nn_order;
    data.svc_data.nn_order_inv = svc_nn_order_inv;

    // Parse SVC covariance type
    if (svc_cov_type_str == "exponential") {
      data.svc_data.cov_type = ratiod_svc::CovType::EXPONENTIAL;
    } else if (svc_cov_type_str == "matern") {
      data.svc_data.cov_type = ratiod_svc::CovType::MATERN;
    } else if (svc_cov_type_str == "gaussian") {
      data.svc_data.cov_type = ratiod_svc::CovType::GAUSSIAN;
    } else if (svc_cov_type_str == "spherical") {
      data.svc_data.cov_type = ratiod_svc::CovType::SPHERICAL;
    } else {
      data.svc_data.cov_type = ratiod_svc::CovType::EXPONENTIAL;  // Default
    }

    // Prior parameters
    data.svc_sigma2_prior_scale = svc_sigma2_scale;
    data.svc_phi_prior_lower = svc_phi_lower;
    data.svc_phi_prior_upper = svc_phi_upper;
  } else {
    data.svc_data.n_svc = 0;
    data.svc_data.n_obs = data.N;
    data.svc_data.nn = 0;
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
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    // Determine sampler name: if any chain switched, report it
    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
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
    bool verbose,
    int max_treedepth = 10,
    double adapt_delta = -1.0
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
  std::vector<double> nn_neighbor_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_neighbor_dist"]);  // Phase 1.3

  int nn = Rcpp::as<int>(gp_params["nn"]);
  std::string cov_type_str = Rcpp::as<std::string>(gp_params["cov_type"]);
  double nu = Rcpp::as<double>(gp_params["nu"]);
  bool gp_shared = Rcpp::as<bool>(gp_params["shared"]);
  double gp_sigma2_prior_U = Rcpp::as<double>(gp_params["sigma2_prior_U"]);
  double gp_sigma2_prior_alpha = Rcpp::as<double>(gp_params["sigma2_prior_alpha"]);
  double gp_phi_prior_lower = Rcpp::as<double>(gp_params["phi_prior_lower"]);
  double gp_phi_prior_upper = Rcpp::as<double>(gp_params["phi_prior_upper"]);

  // GP solver configuration
  std::string gp_solver_str = Rcpp::as<std::string>(gp_params["solver"]);
  double gp_cg_tol = Rcpp::as<double>(gp_params["cg_tol"]);
  int gp_cg_maxiter = Rcpp::as<int>(gp_params["cg_maxiter"]);

  // Observation-to-location mapping (1-based from R, convert to 0-based)
  std::vector<int> gp_obs_to_loc_r = Rcpp::as<std::vector<int>>(gp_params["gp_obs_to_loc"]);
  int gp_n_unique = Rcpp::as<int>(gp_params["n_unique"]);

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
  std::vector<double> nn_neighbor_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_local"]);  // Phase 1.3
  std::vector<double> nn_neighbor_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_regional"]);  // Phase 1.3
  double range_local_lower = Rcpp::as<double>(ms_gp_params["range_local_lower"]);
  double range_local_upper = Rcpp::as<double>(ms_gp_params["range_local_upper"]);
  double range_regional_lower = Rcpp::as<double>(ms_gp_params["range_regional_lower"]);
  double range_regional_upper = Rcpp::as<double>(ms_gp_params["range_regional_upper"]);
  double ms_sigma2_local_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_U"]);
  double ms_sigma2_local_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_alpha"]);
  double ms_sigma2_regional_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_U"]);
  double ms_sigma2_regional_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_alpha"]);
  std::string msgp_sampler_str = Rcpp::as<std::string>(ms_gp_params["sampler"]);

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
  } else if (model_type_str == "poisson_gamma") {
    data.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.model_type = ModelType::POISSON_GAMMA;  // fallback
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
    data.has_hsgp = false;

    data.gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
    data.gp_data.nn = nn;
    data.gp_data.coords = coords_vec;  // Already std::vector from eager copy
    data.gp_data.nn_idx = nn_idx_vec;
    data.gp_data.nn_dist = nn_dist_vec;
    data.gp_data.nn_neighbor_dist = nn_neighbor_dist_vec;  // Phase 1.3: cached pairwise distances
    // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
    data.gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
    for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
      data.gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
    }
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

    // Set solver configuration
    data.gp_data.solver_config.solver = ratiod_gp::parse_gp_solver(gp_solver_str);
    data.gp_data.solver_config.cg_tol = gp_cg_tol;
    data.gp_data.solver_config.cg_maxiter = gp_cg_maxiter;
    data.gp_data.solver_config.n_obs = gp_n_unique;  // Unique locations

    data.gp_sigma2_prior_U = gp_sigma2_prior_U;
    data.gp_sigma2_prior_alpha = gp_sigma2_prior_alpha;
    data.gp_phi_prior_lower = gp_phi_prior_lower;
    data.gp_phi_prior_upper = gp_phi_prior_upper;

  } else if (gp_type_str == "multiscale_gp") {
    data.spatial_type = SpatialType::MULTISCALE_GP;
    data.has_gp = false;
    data.has_multiscale_gp = true;
    data.has_hsgp = false;

    data.multiscale_gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
    data.multiscale_gp_data.coords = coords_vec;  // Already std::vector from eager copy
    // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
    data.multiscale_gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
    for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
      data.multiscale_gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
    }

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
    data.multiscale_gp_data.nn_neighbor_dist_local = nn_neighbor_dist_local_vec;  // Phase 1.3

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
    data.multiscale_gp_data.nn_neighbor_dist_regional = nn_neighbor_dist_regional_vec;  // Phase 1.3

    // Range constraints
    data.multiscale_gp_data.range_local_lower = range_local_lower;
    data.multiscale_gp_data.range_local_upper = range_local_upper;
    data.multiscale_gp_data.range_regional_lower = range_regional_lower;
    data.multiscale_gp_data.range_regional_upper = range_regional_upper;

    data.multiscale_gp_data.cov_type = cov_type;
    data.multiscale_gp_data.nu = nu;
    data.multiscale_gp_data.shared = gp_shared;
    data.multiscale_gp_data.sampler = ratiod_gp::parse_msgp_sampler(msgp_sampler_str);

    data.ms_sigma2_local_prior_U = ms_sigma2_local_prior_U;
    data.ms_sigma2_local_prior_alpha = ms_sigma2_local_prior_alpha;
    data.ms_sigma2_regional_prior_U = ms_sigma2_regional_prior_U;
    data.ms_sigma2_regional_prior_alpha = ms_sigma2_regional_prior_alpha;

  } else if (gp_type_str == "hsgp") {
    data.spatial_type = SpatialType::HSGP;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = true;

    // HSGP parameters from gp_params
    int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
    double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
    bool hsgp_shared = gp_shared;

    // Setup HSGP data structure with precomputed basis functions
    ratiod_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                hsgp_shared, data.hsgp_data);

    data.hsgp_m_per_dim = hsgp_m;
    data.hsgp_boundary_factor = hsgp_c;

  } else {
    data.spatial_type = SpatialType::NONE;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;
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

  // Zero-inflation structure (GP interface: no OI support, use matrix directly)
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
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth,
      MassMatrixType::DIAG, adapt_delta, -1
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth,
      MassMatrixType::DIAG, adapt_delta, -1
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
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
  int max_treedepth = 10;
  if (args.containsElementNamed("max_treedepth")) {
    max_treedepth = Rcpp::as<int>(args["max_treedepth"]);
  }
  double adapt_delta = -1.0;
  if (args.containsElementNamed("adapt_delta")) {
    adapt_delta = Rcpp::as<double>(args["adapt_delta"]);
  }

  // Delegate to the original implementation
  return cpp_hmc_fit_gp(
    q_init, y_num, y_denom, y_denom_cont,
    X_num, X_denom, re_group, n_re_groups,
    model_type_str, gp_params, ms_gp_params, ms_temporal_params, rsr_params,
    sigma_beta, sigma_re_scale, phi_prior_shape, phi_prior_rate,
    zi_type_str, X_zi, zi_prior_sd,
    n_iter, n_warmup, L, n_chains, seed, n_threads, verbose, max_treedepth, adapt_delta
  );
}
