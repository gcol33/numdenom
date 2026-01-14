// hmc_sampler.h
// Full HMC/NUTS backend with spatial, temporal, and zero-inflation support
// Supports ICAR/BYM2 spatial effects, RW/AR1 temporal, and ZI/hurdle models

#ifndef QUOTR_HMC_SAMPLER_H
#define QUOTR_HMC_SAMPLER_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "hmc_temporal.h"
#include "hmc_zi.h"
#include "hmc_svc.h"
#include "hmc_gp.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_latent.h"
#include "hmc_spatiotemporal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_hmc {

using ratiod_temporal::TemporalType;
using ratiod_temporal::TemporalData;
using ratiod_zi::ZIType;
using ratiod_gp::GPData;
using ratiod_gp::MultiscaleGPData;
using ratiod_gp::CovType;
using ratiod_temporal::MultiscaleTemporalData;
using ratiod_spatiotemporal::STType;
using ratiod_spatiotemporal::SpatiotemporalData;
using ratiod_spatiotemporal::NonsepType;

// =====================================================================
// Model configuration
// =====================================================================

enum class ModelType { BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA };
enum class SpatialType { NONE, ICAR, BYM2, GP, MULTISCALE_GP };

// Model data container with spatial support
struct ModelData {
  // Response data
  std::vector<int> y_num;
  std::vector<int> y_denom;
  std::vector<double> y_denom_cont;

  // Design matrices (stored as flat vectors for cache efficiency)
  std::vector<double> X_num_flat;
  std::vector<double> X_denom_flat;
  int p_num, p_denom;

  // Random effects (supports multiple crossed RE terms with slopes)
  std::vector<int> re_group;  // 1-based group index (0 = no RE) - legacy single term
  int n_re_groups;            // Number of groups for first RE term

  // Multi-term RE structure
  int n_re_terms;                              // Number of RE terms (0 if none)
  std::vector<std::vector<int>> re_group_multi; // [term][obs] -> group index (1-based)
  std::vector<int> re_n_groups_multi;          // Groups per term
  std::vector<int> re_offsets;                 // Offset in flattened RE parameter vector per term
  int total_re_groups;                         // Sum of all groups across terms

  // Random slopes support
  bool has_re_slopes;                          // Whether any RE term has slopes
  bool has_re_correlated_slopes;               // Whether any RE term has correlated slopes
  std::vector<int> re_n_coefs;                 // Coefficients per group per term (1 = intercept only)
  std::vector<std::vector<double>> re_slope_matrices; // [term] -> flattened [N x n_slopes] slope design matrix
  std::vector<int> re_n_slopes;                // Number of slope variables per term
  std::vector<bool> re_correlated;             // Whether each term has correlated slopes
  std::vector<int> re_n_chol;                  // Cholesky parameters per term (k*(k-1)/2 for correlated, 0 otherwise)
  int total_re_params;                         // Total RE parameters (groups * coefs)
  int total_sigma_params;                      // Total variance parameters (sum of n_coefs)
  int total_chol_params;                       // Total Cholesky correlation parameters

  // Spatial structure
  SpatialType spatial_type;
  std::vector<int> spatial_group;    // Maps obs to spatial unit (1-based)
  int n_spatial_units;
  std::vector<int> adj_row_ptr;      // CSR format: row pointers
  std::vector<int> adj_col_idx;      // CSR format: column indices
  std::vector<int> n_neighbors;      // Number of neighbors per unit
  double bym2_scale_factor;          // For BYM2 scaling

  // Temporal structure
  TemporalType temporal_type;
  std::vector<int> temporal_time_idx;   // Maps obs to time point (1-based, 0 = no temporal)
  std::vector<int> temporal_group_idx;  // Maps obs to temporal group (1-based)
  int n_times;                          // Number of time points
  int n_temporal_groups;                // Number of temporal groups (1 if no grouping)
  int n_temporal_params;                // Total temporal parameters
  bool temporal_cyclic;                 // Whether RW is cyclic
  bool temporal_shared;                 // Whether effect is shared between num/denom
  double tau_temporal_shape;            // Gamma shape for temporal precision
  double tau_temporal_rate;             // Gamma rate for temporal precision

  // Zero-inflation structure
  ZIType zi_type;
  std::vector<double> X_zi_flat;        // Design matrix for ZI probability (flat)
  int p_zi;                             // Number of ZI predictors
  double zi_prior_sd;                   // Prior SD for ZI coefficients

  // SVC (Spatially-Varying Coefficients) structure
  ratiod_svc::SVCData svc_data;
  bool has_svc;
  double svc_sigma2_prior_scale;        // Half-Cauchy scale for sigma2
  double svc_phi_prior_lower;           // Uniform prior lower bound for phi
  double svc_phi_prior_upper;           // Uniform prior upper bound for phi

  // GP spatial structure (single-scale)
  GPData gp_data;
  bool has_gp;
  double gp_sigma2_prior_U;             // PC prior: P(sigma > U) = alpha
  double gp_sigma2_prior_alpha;
  double gp_phi_prior_lower;            // Uniform prior bounds for range
  double gp_phi_prior_upper;

  // Multi-scale GP spatial structure
  MultiscaleGPData multiscale_gp_data;
  bool has_multiscale_gp;
  double ms_sigma2_local_prior_U;
  double ms_sigma2_local_prior_alpha;
  double ms_sigma2_regional_prior_U;
  double ms_sigma2_regional_prior_alpha;

  // Multi-scale temporal structure
  MultiscaleTemporalData multiscale_temporal_data;
  bool has_multiscale_temporal;
  double ms_sigma2_trend_prior_U;
  double ms_sigma2_trend_prior_alpha;
  double ms_sigma2_seasonal_prior_U;
  double ms_sigma2_seasonal_prior_alpha;
  double ms_sigma2_short_prior_U;
  double ms_sigma2_short_prior_alpha;

  // RSR (Restricted Spatial Regression) structure
  bool has_rsr;
  std::vector<double> rsr_projection;   // P_perp matrix (n x n, flattened)
  int rsr_n;                            // Dimension of projection matrix

  // Latent factors for unmeasured confounders
  bool has_latent;
  int latent_n_factors;                 // Number of latent factors (K)
  bool latent_shared;                   // Whether factors enter both num and denom
  bool latent_scale;                    // Whether to standardize factors
  int latent_constraint;                // 0 = sum_to_zero, 1 = first_zero
  double latent_sigma_prior_rate;       // Exponential rate for PC prior on sigma

  // Spatiotemporal interaction
  bool has_spatiotemporal;
  SpatiotemporalData spatiotemporal_data;
  double st_sigma2_prior_U;             // PC prior for interaction variance
  double st_sigma2_prior_alpha;
  double st_phi_space_prior_lower;      // Uniform bounds for spatial range
  double st_phi_space_prior_upper;
  double st_phi_time_prior_lower;       // Uniform bounds for temporal range
  double st_phi_time_prior_upper;

  // Dimensions
  int N;

  // Prior parameters
  double sigma_beta;
  double sigma_re_scale;
  double phi_prior_shape;
  double phi_prior_rate;
  double tau_spatial_shape;
  double tau_spatial_rate;

  // Model type
  ModelType model_type;

  // Parallelization
  int n_threads;
};

// =====================================================================
// Parameter layout
// =====================================================================

// Parameter vector layout (order matters for cache efficiency):
// [beta_num, beta_denom, log_sigma_re?, re?, log_phi_num?, log_phi_denom?,
//  log_tau_spatial?, spatial?, log_sigma_bym2?, log_rho_bym2?, theta_bym2?]

struct ParamLayout {
  int beta_num_start, beta_num_end;
  int beta_denom_start, beta_denom_end;
  int log_sigma_re_idx;       // Legacy: index for single RE term
  int re_start, re_end;       // Legacy: bounds for single RE term

  // Multi-term RE layout (supports intercept-only terms)
  std::vector<int> log_sigma_re_multi;  // Index of log_sigma_re for each term
  std::vector<int> re_start_multi;      // Start index for each RE term
  std::vector<int> re_end_multi;        // End index for each RE term

  // Random slopes layout (when has_re_slopes is true)
  bool has_re_slopes;
  bool has_re_correlated_slopes;
  // For each term: number of coefficients (1 = intercept only, >1 = with slopes)
  std::vector<int> re_n_coefs_multi;
  // For each term: whether correlated (for | syntax) vs uncorrelated (for || syntax)
  std::vector<bool> re_correlated_multi;
  // For slopes: log_sigma_re_slopes[term] = [idx_intercept, idx_slope1, idx_slope2, ...]
  std::vector<std::vector<int>> log_sigma_re_slopes;
  // For correlated slopes: Cholesky lower-triangular parameters (off-diagonal only)
  // chol_re_multi[term] = [L_21, L_31, L_32, ...] (column-major lower triangle)
  std::vector<int> chol_re_start_multi;  // Start index for Cholesky params for each term
  std::vector<int> chol_re_end_multi;    // End index for Cholesky params for each term
  // re_coefs[term][group] = [re_intercept, re_slope1, re_slope2, ...]
  // We track start/end per term, and n_coefs per term to decode within
  int log_phi_num_idx;
  int log_phi_denom_idx;
  int log_tau_spatial_idx;
  int spatial_start, spatial_end;
  // BYM2 extras
  int log_sigma_bym2_idx;
  int logit_rho_bym2_idx;
  int theta_bym2_start, theta_bym2_end;

  // Temporal parameters
  int log_tau_temporal_idx;             // Log precision for temporal
  int logit_rho_ar1_idx;                // AR1 autocorrelation (logit scale)
  int temporal_start, temporal_end;     // Temporal effect parameters

  // Zero-inflation parameters
  int beta_zi_start, beta_zi_end;       // ZI regression coefficients

  // SVC parameters
  int log_sigma2_svc_start, log_sigma2_svc_end;  // Log spatial variance per SVC term
  int log_phi_svc_start, log_phi_svc_end;        // Log range parameter per SVC term
  int svc_w_start, svc_w_end;                    // SVC values (n_obs x n_svc)

  // GP spatial parameters
  int log_sigma2_gp_idx;                // Log spatial variance
  int log_phi_gp_idx;                   // Log range parameter
  int gp_w_start, gp_w_end;             // GP spatial effects (n_obs)

  // Multi-scale GP parameters
  int log_sigma2_gp_local_idx;          // Log variance for local scale
  int log_phi_gp_local_idx;             // Log range for local scale
  int log_sigma2_gp_regional_idx;       // Log variance for regional scale
  int log_phi_gp_regional_idx;          // Log range for regional scale
  int gp_local_start, gp_local_end;     // Local GP effects
  int gp_regional_start, gp_regional_end; // Regional GP effects

  // Multi-scale temporal parameters
  int log_sigma2_trend_idx;             // Log variance for trend
  int log_sigma2_seasonal_idx;          // Log variance for seasonal
  int log_sigma2_short_idx;             // Log variance for short-term
  int logit_rho_short_idx;              // AR1 autocorrelation for short-term
  int trend_start, trend_end;           // Trend effects (n_times)
  int seasonal_start, seasonal_end;     // Seasonal effects (period)
  int short_term_start, short_term_end; // Short-term effects (n_times)

  // Latent factor parameters
  int log_sigma_latent_start, log_sigma_latent_end;  // Log sigma per factor
  int latent_factor_start, latent_factor_end;        // Factor scores (N x K)

  // Spatiotemporal interaction parameters
  int log_tau_st_idx;                                // Log precision for ST interaction
  int log_tau_st2_idx;                               // Second precision (Type IV)
  int logit_rho_st_idx;                              // AR1 autocorrelation if ST temporal is AR1
  int log_phi_st_space_idx;                          // Log spatial range (GP-based)
  int log_phi_st_time_idx;                           // Log temporal range (GP-based)
  int st_delta_start, st_delta_end;                  // ST interaction effects (S * T)

  int total_params;

  bool has_re;
  bool has_phi_num;
  bool has_phi_denom;
  bool has_spatial;
  bool is_bym2;
  bool is_gp;
  bool is_multiscale_gp;
  bool has_temporal;
  bool is_ar1;
  bool has_multiscale_temporal;
  bool has_zi;
  bool has_svc;
  bool has_latent;
  bool has_spatiotemporal;
  bool is_st_gp;
};

ParamLayout compute_param_layout(const ModelData& data);
int get_n_params(const ModelData& data);

// =====================================================================
// Log-posterior computation (with OpenMP parallelization)
// =====================================================================

// Main log-posterior function
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);

// Numerical gradient (central difference)
void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
);

// Compute ICAR quadratic form: phi' Q phi
double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
);

// =====================================================================
// HMC/NUTS sampler structures
// =====================================================================

struct LeapfrogResult {
  std::vector<double> q;
  std::vector<double> p;
  double log_prob;
  bool divergent;
};

struct DualAveraging {
  double mu, log_epsilon_bar, H_bar;
  double gamma, t0, kappa;
  double target_accept;  // Target acceptance rate (dimension-adaptive)
  int m;

  // Default constructor with dimension-adaptive target
  DualAveraging(double epsilon_init = 1.0, int n_params = 1);
  double update(double alpha);
  double final_epsilon() const;

  // Compute dimension-adaptive target acceptance rate
  // Based on optimal scaling theory: lower for higher dimensions
  static double compute_target(int n_params) {
    // For d dimensions: target ≈ 0.65 for d=1, decreasing to ~0.55 for d≥50
    if (n_params <= 5) return 0.65;
    if (n_params <= 20) return 0.60;
    if (n_params <= 50) return 0.57;
    return 0.55;
  }
};

struct ChainState {
  std::vector<double> q;
  double log_prob;
  double epsilon;
  DualAveraging da;
  std::mt19937 rng;
  int n_divergent;
};

// Pure C++ result struct (safe for OpenMP parallel regions)
struct HMCResultCpp {
  std::vector<std::vector<double>> samples;  // [sample_idx][param_idx]
  std::vector<double> log_prob;
  std::vector<double> accept_prob;
  std::vector<int> n_leapfrog;
  std::vector<int> divergent;
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
};

// R-compatible result struct (create outside parallel regions)
struct HMCResult {
  Rcpp::NumericMatrix samples;
  Rcpp::NumericVector log_prob;
  Rcpp::NumericVector accept_prob;
  Rcpp::IntegerVector n_leapfrog;
  Rcpp::IntegerVector divergent;
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
};

// Convert C++ result to R result (call outside parallel region)
inline HMCResult cpp_to_r_result(const HMCResultCpp& cpp_result, int n_params) {
  HMCResult r_result;
  r_result.samples = Rcpp::NumericMatrix(cpp_result.n_sample, n_params);
  r_result.log_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.accept_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.n_leapfrog = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.divergent = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.epsilon = cpp_result.epsilon;
  r_result.n_warmup = cpp_result.n_warmup;
  r_result.n_sample = cpp_result.n_sample;
  r_result.chain_id = cpp_result.chain_id;

  for (int i = 0; i < cpp_result.n_sample; i++) {
    for (int j = 0; j < n_params; j++) {
      r_result.samples(i, j) = cpp_result.samples[i][j];
    }
    r_result.log_prob[i] = cpp_result.log_prob[i];
    r_result.accept_prob[i] = cpp_result.accept_prob[i];
    r_result.n_leapfrog[i] = cpp_result.n_leapfrog[i];
    r_result.divergent[i] = cpp_result.divergent[i];
  }

  return r_result;
}

// =====================================================================
// Sampler functions
// =====================================================================

// Single leapfrog step
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout
);

// Find reasonable initial step size
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
);

// Run single HMC chain (C++ version - safe for parallel)
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
);

// Run single HMC chain (R wrapper)
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
);

// Run multiple chains in parallel (across-chain parallelization)
std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose
);

} // namespace ratiod_hmc

#endif // QUOTR_HMC_SAMPLER_H
