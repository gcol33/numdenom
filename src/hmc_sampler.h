// hmc_sampler.h
// Full HMC/NUTS backend with spatial, temporal, and zero-inflation support
// Supports ICAR/BYM2 spatial effects, RW/AR1 temporal, and ZI/hurdle models

#ifndef QUOTR_HMC_SAMPLER_H
#define QUOTR_HMC_SAMPLER_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg_fast.h"
#include "hmc_temporal.h"
#include "hmc_temporal_gp.h"
#include "hmc_zi.h"
#include "hmc_svc.h"
#include "hmc_gp.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_latent.h"
#include "hmc_spatiotemporal.h"
#include "hmc_hsgp.h"
#include "hmc_tvc.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_hmc {

using ratiod_temporal::TemporalType;
using ratiod_temporal::TemporalData;
using ratiod_temporal_gp::TemporalGPData;
using ratiod_temporal_gp::TemporalCovType;
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

enum class ModelType { BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA, GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL };
enum class SpatialType { NONE, ICAR, BYM2, GP, MULTISCALE_GP, HSGP };

// Gradient computation methods
// AUTO: Use fastest available (H > A_r > A > A_t > N)
// NUMERICAL (N): Finite differences - slow but always works
// AUTODIFF_TAPE (A_t): Tape-based reverse-mode - slow due to heap allocation
// AUTODIFF_ARENA (A_r): Arena-based reverse-mode - fast O(N), ~10-30x over A_t
// AUTODIFF_FORWARD (A): Forward-mode dual numbers - O(p*N), thread-safe
// HANDCODED (H): Analytical gradients - fastest when available
enum class GradientMode { AUTO, NUMERICAL, AUTODIFF_TAPE, AUTODIFF_ARENA, AUTODIFF_FORWARD, HANDCODED };

// Mass matrix type for NUTS
enum class MassMatrixType { DIAG, DENSE, AUTO };

// Parse gradient mode from string
inline GradientMode parse_gradient_mode(const std::string& mode_str) {
    if (mode_str == "auto" || mode_str == "AUTO") return GradientMode::AUTO;
    if (mode_str == "N" || mode_str == "numerical") return GradientMode::NUMERICAL;
    if (mode_str == "A_t" || mode_str == "autodiff_tape") return GradientMode::AUTODIFF_TAPE;
    if (mode_str == "A_r" || mode_str == "arena" || mode_str == "autodiff_arena") return GradientMode::AUTODIFF_ARENA;
    if (mode_str == "A" || mode_str == "autodiff" || mode_str == "forward") return GradientMode::AUTODIFF_FORWARD;
    if (mode_str == "H" || mode_str == "handcoded" || mode_str == "analytical") return GradientMode::HANDCODED;
    return GradientMode::AUTO;  // Default
}

// Parse metric type from string
inline MassMatrixType parse_metric_type(const std::string& metric_str) {
    if (metric_str == "dense" || metric_str == "DENSE") return MassMatrixType::DENSE;
    if (metric_str == "auto" || metric_str == "AUTO") return MassMatrixType::AUTO;
    return MassMatrixType::DIAG;  // Default
}

// Set global gradient mode (defined in hmc_sampler.cpp)
void set_gradient_mode(GradientMode mode);

// Model data container with spatial support
struct ModelData {
  // Response data
  std::vector<int> y_num;
  std::vector<int> y_denom;
  std::vector<double> y_num_cont;    // For continuous numerator (gamma_gamma, lognormal)
  std::vector<double> y_denom_cont;  // For continuous denominator (poisson_gamma, gamma_gamma, lognormal)

  // Design matrices (stored as flat vectors for cache efficiency)
  std::vector<double> X_num_flat;
  std::vector<double> X_denom_flat;
  int p_num, p_denom;

  // Random effects (supports multiple crossed RE terms with slopes)
  std::vector<int> re_group;  // 1-based group index (0 = no RE) - legacy single term
  int n_re_groups;            // Number of groups for first RE term

  // Multi-term RE structure
  int n_re_terms;                              // Number of RE terms (0 if none)
  std::vector<std::vector<int>> re_group_multi; // [term][obs] -> group index (1-based) LEGACY
  std::vector<int> re_group_multi_flat;        // [term * N + obs] -> group index (1-based), contiguous
  std::vector<int> re_n_groups_multi;          // Groups per term
  std::vector<int> re_offsets;                 // Offset in flattened RE parameter vector per term
  int total_re_groups;                         // Sum of all groups across terms

  // Random slopes support
  bool has_re_slopes = false;                  // Whether any RE term has slopes
  bool has_re_correlated_slopes = false;       // Whether any RE term has correlated slopes
  std::vector<int> re_n_coefs;                 // Coefficients per group per term (1 = intercept only)
  std::vector<std::vector<double>> re_slope_matrices; // [term] -> flattened [N x n_slopes] slope design matrix
  std::vector<int> re_n_slopes;                // Number of slope variables per term
  std::vector<bool> re_correlated;             // Whether each term has correlated slopes
  std::vector<int> re_n_chol;                  // Cholesky parameters per term (k*(k-1)/2 for correlated, 0 otherwise)
  int total_re_params;                         // Total RE parameters (groups * coefs)
  int total_sigma_params;                      // Total variance parameters (sum of n_coefs)
  int total_chol_params;                       // Total Cholesky correlation parameters

  // RE parameterization: 0 = centered (default), 1 = non-centered
  // Non-centered stores z ~ N(0,1) instead of re ~ N(0, sigma^2)
  // and computes re = sigma * z (or re = diag(sigma) * L * z for correlated)
  int re_parameterization;                     // 0 = centered, 1 = non-centered

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

  // One-inflation structure (for OI-binomial and ZOIB models)
  std::vector<double> X_oi_flat;        // Design matrix for OI probability (flat)
  int p_oi;                             // Number of OI predictors
  double oi_prior_sd;                   // Prior SD for OI coefficients

  // SVC (Spatially-Varying Coefficients) structure
  ratiod_svc::SVCData svc_data;
  bool has_svc = false;
  double svc_sigma2_prior_scale = 1.0;  // Half-Cauchy scale for sigma2
  double svc_phi_prior_lower = 0.01;    // Uniform prior lower bound for phi
  double svc_phi_prior_upper = 10.0;    // Uniform prior upper bound for phi

  // GP spatial structure (single-scale)
  GPData gp_data;
  bool has_gp = false;
  double gp_sigma2_prior_U = 1.0;       // PC prior: P(sigma > U) = alpha
  double gp_sigma2_prior_alpha = 0.01;
  double gp_phi_prior_lower = 0.01;     // Uniform prior bounds for range
  double gp_phi_prior_upper = 10.0;

  // Multi-scale GP spatial structure
  MultiscaleGPData multiscale_gp_data;
  bool has_multiscale_gp = false;
  double ms_sigma2_local_prior_U = 1.0;
  double ms_sigma2_local_prior_alpha = 0.01;
  double ms_sigma2_regional_prior_U = 1.0;
  double ms_sigma2_regional_prior_alpha = 0.01;

  // Multi-scale temporal structure
  MultiscaleTemporalData multiscale_temporal_data;
  bool has_multiscale_temporal = false;
  double ms_sigma2_trend_prior_U = 1.0;
  double ms_sigma2_trend_prior_alpha = 0.01;
  double ms_sigma2_seasonal_prior_U = 1.0;
  double ms_sigma2_seasonal_prior_alpha = 0.01;
  double ms_sigma2_short_prior_U = 1.0;
  double ms_sigma2_short_prior_alpha = 0.01;

  // Temporal GP structure (for irregularly-spaced time series)
  TemporalGPData temporal_gp_data;
  bool has_temporal_gp = false;
  double temporal_gp_sigma2_prior_U = 1.0;    // PC prior: P(sigma > U) = alpha
  double temporal_gp_sigma2_prior_alpha = 0.01;
  double temporal_gp_phi_prior_lower = 0.01;  // Uniform prior bounds for range
  double temporal_gp_phi_prior_upper = 10.0;

  // HSGP (Hilbert Space GP) structure
  ratiod_hsgp::HSGPData hsgp_data;
  bool has_hsgp = false;
  int hsgp_m_per_dim = 15;             // Basis functions per dimension
  double hsgp_boundary_factor = 1.5;   // Boundary factor (c)

  // RSR (Restricted Spatial Regression) structure
  bool has_rsr = false;
  std::vector<double> rsr_projection;   // P_perp matrix (n x n, flattened)
  int rsr_n = 0;                        // Dimension of projection matrix

  // Latent factors for unmeasured confounders
  bool has_latent = false;
  int latent_n_factors = 0;             // Number of latent factors (K)
  bool latent_shared = true;            // Whether factors enter both num and denom
  bool latent_scale = true;             // Whether to standardize factors
  int latent_constraint = 0;            // 0 = sum_to_zero, 1 = first_zero
  double latent_sigma_prior_rate = 1.0; // Exponential rate for PC prior on sigma

  // Spatiotemporal interaction
  bool has_spatiotemporal = false;
  SpatiotemporalData spatiotemporal_data;
  double st_sigma2_prior_U = 1.0;       // PC prior for interaction variance
  double st_sigma2_prior_alpha = 0.01;
  double st_phi_space_prior_lower = 0.01;  // Uniform bounds for spatial range
  double st_phi_space_prior_upper = 10.0;
  double st_phi_time_prior_lower = 0.01;   // Uniform bounds for temporal range
  double st_phi_time_prior_upper = 10.0;

  // TVC (Temporally-Varying Coefficients) structure
  ratiod_tvc::TVCData tvc_data;
  bool has_tvc = false;
  double tvc_tau_shape = 1.0;           // Gamma shape for TVC precision
  double tvc_tau_rate = 0.01;           // Gamma rate for TVC precision

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
//  log_tau_spatial?, spatial?, log_sigma_bym2?, logit_rho_bym2?, theta_bym2?]

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
  bool has_re_slopes = false;
  bool has_re_correlated_slopes = false;
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
  // BYM2 extras (Riebler reparameterization: sigma_total, rho)
  // sigma_s = sigma_total * sqrt(rho), sigma_u = sigma_total * sqrt(1-rho)
  int log_sigma_bym2_idx;     // log(sigma_total): total spatial SD
  int logit_rho_bym2_idx;     // logit(rho): mixing parameter (structured fraction)
  int theta_bym2_start, theta_bym2_end;

  // Temporal parameters
  int log_tau_temporal_idx;             // Log precision for temporal
  int logit_rho_ar1_idx;                // AR1 autocorrelation (logit scale)
  int temporal_start, temporal_end;     // Temporal effect parameters

  // Temporal GP parameters (for irregularly-spaced time series)
  int log_sigma2_temporal_gp_idx;       // Log marginal variance
  int log_phi_temporal_gp_idx;          // Log length-scale

  // Zero-inflation parameters
  int beta_zi_start, beta_zi_end;       // ZI regression coefficients

  // One-inflation parameters (for OI-binomial and ZOIB)
  int beta_oi_start, beta_oi_end;       // OI regression coefficients

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

  // HSGP parameters
  int log_sigma2_hsgp_idx;              // Log spatial variance
  int log_lengthscale_hsgp_idx;         // Log lengthscale
  int hsgp_beta_start, hsgp_beta_end;   // HSGP basis coefficients (m^2)

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

  // TVC (Temporally-Varying Coefficients) parameters
  int log_tau_tvc_start, log_tau_tvc_end;            // Log precision per TVC term
  int logit_rho_tvc_start, logit_rho_tvc_end;        // AR1 correlations (if structure == AR1)
  int tvc_w_start, tvc_w_end;                        // TVC values (n_groups * n_tvc * n_times)

  int total_params = 0;

  bool has_re = false;
  bool has_phi_num = false;
  bool has_phi_denom = false;
  bool has_spatial = false;
  bool is_bym2 = false;
  bool is_gp = false;
  bool is_multiscale_gp = false;
  bool is_hsgp = false;
  bool has_temporal = false;
  bool is_ar1 = false;
  bool is_temporal_gp = false;
  bool has_multiscale_temporal = false;
  bool has_zi = false;
  bool has_oi = false;  // For OI-binomial and ZOIB models
  bool has_svc = false;
  bool has_latent = false;
  bool has_spatiotemporal = false;
  bool is_st_gp = false;
  bool has_tvc = false;
};

ParamLayout compute_param_layout(const ModelData& data);
int get_n_params(const ModelData& data);

// =====================================================================
// Log-posterior computation (with OpenMP parallelization)
// =====================================================================

// Main log-posterior function
// When skip_obs_loop=true, returns only prior+structural terms (O(p+S+T)),
// skipping the O(N) observation loop. Used by fused gradient+log_post computation.
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop = false
);

// Gradient computation (with optional fused log-posterior)
// When log_post_out is non-null, the log-posterior is computed alongside
// the gradient in a single pass, avoiding redundant O(N) computation.
void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
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

// NUTS-specific structures
struct LeapfrogResultWithGrad {
  std::vector<double> q, p, grad;
  double log_prob;
  bool divergent;
};

struct NUTSNode {
  std::vector<double> q, p, grad;
  double log_prob;
};

struct NUTSTreeResult {
  NUTSNode left, right;
  std::vector<double> q_proposal;
  std::vector<double> grad_proposal;
  double log_prob_proposal;
  int n_valid;
  bool divergent, stop;
  double sum_accept_prob;
  int n_leapfrog;
  double sum_log_weight;
};

// =====================================================================
// Optimized NUTS infrastructure (zero-allocation tree building)
// =====================================================================

// Result from in-place leapfrog (no vectors allocated)
struct LeapfrogInPlaceResult {
  double log_prob;
  bool divergent;
};

// Lightweight tree statistics (returned by build_tree_fast)
// Endpoints tracked via workspace slot indices, not owned vectors
struct TreeStats {
  int left_slot;            // Workspace slot index for left endpoint
  int right_slot;           // Workspace slot index for right endpoint
  int proposal_slot;        // Workspace slot index for proposal
  double sum_log_weight;
  double sum_accept_prob;
  double log_prob_proposal;
  int n_valid;
  int n_leapfrog;
  bool divergent;
  bool stop;

  // Generalized U-turn criterion (Betancourt 2017, Stan implementation)
  std::vector<double> rho;          // Cumulative momentum sum across trajectory
  std::vector<double> p_sharp_beg;  // M^{-1} * p at trajectory beginning
  std::vector<double> p_sharp_end;  // M^{-1} * p at trajectory end
  std::vector<double> p_beg;        // Raw momentum at trajectory beginning
  std::vector<double> p_end;        // Raw momentum at trajectory end
};

// Generalized U-turn criterion: check if momenta are aligned with integrated direction
// Returns true if trajectory should CONTINUE (no U-turn detected)
inline bool compute_criterion(const double* p_sharp_minus, const double* p_sharp_plus,
                              const double* rho, int n) {
  double dot_fwd = 0.0, dot_bwd = 0.0;
  for (int i = 0; i < n; i++) {
    dot_fwd += p_sharp_plus[i] * rho[i];
    dot_bwd += p_sharp_minus[i] * rho[i];
  }
  return (dot_fwd > 0.0) && (dot_bwd > 0.0);
}

// Pre-allocated buffer pool for NUTS tree building
// Eliminates all heap allocations in the build_tree hot path
struct NUTSWorkspace {
  int n;                    // n_params
  int max_depth;
  int stride;               // 3 * n (q + p + grad per slot)
  int total_slots;

  // Contiguous pool: [slot][q|p|grad], each n doubles wide
  std::vector<double> pool;
  std::vector<double> log_probs;  // One per slot

  // Slot allocator (stack-based, reset per top-level build_tree call)
  int next_slot;

  // Working buffers for compute_gradient (needs std::vector interface)
  std::vector<double> params_buf;
  std::vector<double> grad_buf;

  // Scratch buffer for dense mass matrix matvec (p doubles)
  std::vector<double> dense_scratch;

  // Pre-allocated merge buffers for build_tree_fast (depth-indexed, no per-merge alloc)
  // 4 buffers per depth level: rho_init, p_init_end, p_sharp_init_end, rho_check
  std::vector<double> merge_pool;
  static constexpr int MERGE_RHO_INIT = 0;
  static constexpr int MERGE_P_INIT_END = 1;
  static constexpr int MERGE_PSHARP_INIT_END = 2;
  static constexpr int MERGE_RHO_CHECK = 3;
  static constexpr int MERGE_BUFS_PER_DEPTH = 4;

  double* merge_buf(int depth, int buf_idx) {
    return &merge_pool[static_cast<size_t>(depth * MERGE_BUFS_PER_DEPTH + buf_idx) * n];
  }

  // Pre-allocated iteration-level vectors (reused across NUTS iterations)
  std::vector<double> iter_rho, iter_rho_bck, iter_rho_fwd;
  std::vector<double> iter_p_sharp_init;
  std::vector<double> iter_p_fwd_beg, iter_p_fwd_end;
  std::vector<double> iter_p_bck_beg, iter_p_bck_end;
  std::vector<double> iter_p_sharp_fwd_beg, iter_p_sharp_fwd_end;
  std::vector<double> iter_p_sharp_bck_beg, iter_p_sharp_bck_end;
  std::vector<double> iter_rho_seam;

  // Slot layout:
  //   Slots 0, 1: persistent trajectory endpoints (node_left, node_right)
  //   Slots 2+: allocated by build_tree_fast via alloc_slot()
  static constexpr int NODE_LEFT_SLOT = 0;
  static constexpr int NODE_RIGHT_SLOT = 1;
  static constexpr int TREE_START_SLOT = 2;

  void init(int np, int max_d) {
    n = np;
    max_depth = max_d;
    stride = 3 * np;
    // Max slots: 2 persistent + 2^max_depth for tree (generous upper bound)
    total_slots = 2 + (1 << max_d);
    pool.resize(static_cast<size_t>(total_slots) * stride);
    log_probs.resize(total_slots);
    params_buf.resize(np);
    grad_buf.resize(np);
    dense_scratch.resize(np);
    next_slot = TREE_START_SLOT;
    // Pre-allocate merge buffers: 4 per depth level
    merge_pool.resize(static_cast<size_t>(MERGE_BUFS_PER_DEPTH) * max_d * np, 0.0);
    // Pre-allocate iteration-level vectors
    iter_rho.resize(np);
    iter_rho_bck.resize(np);
    iter_rho_fwd.resize(np);
    iter_p_sharp_init.resize(np);
    iter_p_fwd_beg.resize(np);
    iter_p_fwd_end.resize(np);
    iter_p_bck_beg.resize(np);
    iter_p_bck_end.resize(np);
    iter_p_sharp_fwd_beg.resize(np);
    iter_p_sharp_fwd_end.resize(np);
    iter_p_sharp_bck_beg.resize(np);
    iter_p_sharp_bck_end.resize(np);
    iter_rho_seam.resize(np);
  }

  // Allocate a fresh slot (stack-based, no deallocation)
  // Returns -1 if workspace is exhausted
  int alloc_slot() {
    if (next_slot >= total_slots) return -1;
    return next_slot++;
  }

  // Reset allocator for new top-level build_tree call
  // Preserves persistent slots 0 and 1
  void reset_tree() {
    next_slot = TREE_START_SLOT;
  }

  // Access helpers (raw pointers into contiguous pool)
  double* q_at(int slot) { return &pool[slot * stride]; }
  double* p_at(int slot) { return &pool[slot * stride + n]; }
  double* grad_at(int slot) { return &pool[slot * stride + 2 * n]; }
  double& logp_at(int slot) { return log_probs[slot]; }

  // Copy full node (q + p + grad + log_prob) between slots
  void copy_node(int dst, int src) {
    std::memcpy(&pool[dst * stride], &pool[src * stride],
                stride * sizeof(double));
    log_probs[dst] = log_probs[src];
  }

  // Load node data from std::vector sources into a slot
  void load_node(int slot, const double* q, const double* p,
                 const double* grad, double log_prob) {
    std::memcpy(q_at(slot), q, n * sizeof(double));
    std::memcpy(p_at(slot), p, n * sizeof(double));
    std::memcpy(grad_at(slot), grad, n * sizeof(double));
    log_probs[slot] = log_prob;
  }
};

// =====================================================================
// Dense mass matrix for NUTS (encapsulates diag + dense state)
// =====================================================================

struct DenseMassMatrix {
  int n = 0;                              // Dimension
  MassMatrixType type = MassMatrixType::DIAG;
  bool adapted = false;

  // Diagonal (always available, used as fallback)
  std::vector<double> inv_mass_diag;      // M^{-1} diagonal = variance
  std::vector<double> sqrt_mass_diag;     // sqrt(M) diagonal = 1/sqrt(variance) for p sampling

  // Dense (only when type == DENSE)
  std::vector<double> inv_mass_dense;     // Full p×p M^{-1} = regularized sample covariance (column-major)
  std::vector<double> L_inv_mass;         // Cholesky factor L where LL^T = M^{-1} (column-major)

  // Scratch buffer for dense matvec results (avoids per-call allocation)
  std::vector<double> scratch;

  void init(int dim, MassMatrixType t) {
    n = dim;
    type = t;
    adapted = false;
    inv_mass_diag.assign(dim, 1.0);
    sqrt_mass_diag.assign(dim, 1.0);
    scratch.resize(dim);
    if (t == MassMatrixType::DENSE) {
      inv_mass_dense.assign(static_cast<size_t>(dim) * dim, 0.0);
      L_inv_mass.assign(static_cast<size_t>(dim) * dim, 0.0);
      // Initialize as identity
      for (int i = 0; i < dim; i++) {
        inv_mass_dense[static_cast<size_t>(i) * dim + i] = 1.0;
        L_inv_mass[static_cast<size_t>(i) * dim + i] = 1.0;
      }
    }
  }

  // Update dense mass matrix from sample covariance
  // Returns true on success, false on Cholesky failure (degrades to diagonal)
  // Uses Eigen LLT for Cholesky decomposition
  bool update_from_covariance(const double* cov, int n_samples);

  // Sample momentum: p ~ N(0, M) where M = C^{-1}
  // DIAG: p[i] = z * sqrt_mass_diag[i]
  // DENSE: solve L^T * p = z  (back-substitution)
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
    } else {
      // Dense: p = L^{-T} z where LL^T = C (inv_mass)
      // We need p ~ N(0, C^{-1}), so sample z ~ N(0, I), then p = L^{-T} z
      std::vector<double> z(n);
      for (int i = 0; i < n; i++) {
        z[i] = normal(rng);
      }
      ratiod_linalg::tri_solve_upper_transpose(L_inv_mass.data(), z.data(), p, n);
    }
  }

  // Kinetic energy: 0.5 * p^T * C * p  where C = M^{-1}
  double kinetic_energy(const double* p) const {
    if (type == MassMatrixType::DIAG || !adapted) {
      return 0.5 * ratiod_linalg::weighted_norm_squared(p, inv_mass_diag.data(), n);
    } else {
      return 0.5 * ratiod_linalg::quadratic_form(p, inv_mass_dense.data(), n);
    }
  }

  // Compute C * p (for leapfrog position update: q += eps * C * p)
  // Result written to `result` buffer
  void inv_mass_times_p(const double* p, double* result) const {
    if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
    } else {
      ratiod_linalg::symmatvec(inv_mass_dense.data(), p, result, n);
    }
  }

  // Compute diag(C) * p — uses diagonal only, even when dense is available.
  // Kept for backwards compatibility / debugging. The NUTS U-turn criterion
  // now uses inv_mass_times_p() for correct geometry with dense mass.
  void inv_mass_diag_times_p(const double* p, double* result) const {
    for (int i = 0; i < n; i++) {
      result[i] = inv_mass_diag[i] * p[i];
    }
  }

  // Set metric directly from precomputed G^{-1} and its Cholesky L.
  // Used by SoftAbs per-trajectory metric retry. No shrinkage applied.
  void set_from_metric(const std::vector<double>& g_inv,
                       const std::vector<double>& l_g_inv) {
    inv_mass_dense = g_inv;
    L_inv_mass = l_g_inv;
    for (int i = 0; i < n; i++) {
      inv_mass_diag[i] = g_inv[static_cast<size_t>(i) * n + i];
      sqrt_mass_diag[i] = 1.0 / std::sqrt(std::max(inv_mass_diag[i], 1e-10));
    }
    adapted = true;
  }

  // Set diagonal mass from WelfordStats output (same interface as before)
  // When type==DENSE, also populate the dense matrices as diagonal so that
  // the dense code paths (sample_momentum, kinetic_energy, inv_mass_times_p)
  // produce correct results even before full covariance is available.
  void set_diagonal(const std::vector<double>& inv_m, const std::vector<double>& sqrt_m) {
    inv_mass_diag = inv_m;
    sqrt_mass_diag = sqrt_m;
    adapted = true;

    if (type == MassMatrixType::DENSE && !inv_mass_dense.empty()) {
      // Populate dense matrices as diagonal so dense code paths work correctly
      std::fill(inv_mass_dense.begin(), inv_mass_dense.end(), 0.0);
      std::fill(L_inv_mass.begin(), L_inv_mass.end(), 0.0);
      for (int i = 0; i < n; i++) {
        inv_mass_dense[static_cast<size_t>(i) * n + i] = inv_m[i];
        // L where LL^T = inv_mass (diagonal): L[i,i] = sqrt(inv_mass[i])
        L_inv_mass[static_cast<size_t>(i) * n + i] = std::sqrt(inv_m[i]);
      }
    }
  }
};

// =====================================================================
// Online full covariance estimator (Welford's algorithm for outer products)
// =====================================================================

class WelfordCovStats {
public:
  int dim;
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // dim×dim: running sum of (x - mean_old)(x - mean_new)^T

  WelfordCovStats() : dim(0), n(0) {}

  explicit WelfordCovStats(int d) : dim(d), n(0),
    mean(d, 0.0), M2(static_cast<size_t>(d) * d, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    std::vector<double> delta(dim);
    for (int i = 0; i < dim; i++) {
      delta[i] = x[i] - mean[i];
      mean[i] += delta[i] / n;
    }
    // Outer product update: M2 += (x - mean_new) * delta^T
    for (int i = 0; i < dim; i++) {
      double dx_new = x[i] - mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        M2[static_cast<size_t>(j) * dim + i] += val;
        if (i != j) {
          M2[static_cast<size_t>(i) * dim + j] += val;  // Symmetric
        }
      }
    }
  }

  // Get covariance with Oracle Approximating Shrinkage (OAS)
  // (Chen, Wiesel, Eldar, Hero 2010)
  //
  // Σ_shrunk = (1 - ρ) * S + ρ * (tr(S)/p) * I
  //
  // OAS automatically adapts shrinkage intensity:
  //   - When n < p (rank-deficient): ρ → 1, shrinks heavily toward scaled identity
  //   - When n >> p: ρ → 0, recovers the unregularized sample covariance
  // Guarantees positive definiteness when ρ > 0.
  //
  // shrinkage_intensity is set as a side-effect for logging.
  mutable double shrinkage_intensity = 0.0;

  std::vector<double> covariance() const {
    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    if (n < 2) {
      // Return identity
      for (int i = 0; i < dim; i++) {
        cov[static_cast<size_t>(i) * dim + i] = 1.0;
      }
      shrinkage_intensity = 1.0;
      return cov;
    }

    // Step 1: Compute sample covariance S = M2 / (n - 1)
    double inv_nm1 = 1.0 / (n - 1);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        cov[static_cast<size_t>(j) * dim + i] =
            M2[static_cast<size_t>(j) * dim + i] * inv_nm1;
      }
      // Ensure diagonal is positive
      double& diag = cov[static_cast<size_t>(i) * dim + i];
      diag = std::max(1e-6, diag);
    }

    // Step 2: Compute OAS shrinkage intensity
    // tr(S) and tr(S^2)
    double trS = 0.0;
    double trS2 = 0.0;
    for (int i = 0; i < dim; i++) {
      trS += cov[static_cast<size_t>(i) * dim + i];
    }
    // tr(S^2) = sum of all S_ij^2 (Frobenius norm squared)
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        double s_ij = cov[static_cast<size_t>(j) * dim + i];
        trS2 += s_ij * s_ij;
      }
    }

    // OAS formula (Eq. 23 from Chen et al. 2010):
    // ρ = clamp( ((1 - 2/p) * tr(S²) + tr(S)²) /
    //            ((n + 1 - 2/p) * (tr(S²) - tr(S)²/p)), 0, 1 )
    double p = static_cast<double>(dim);
    double nf = static_cast<double>(n);
    double trS_sq = trS * trS;
    double numer = (1.0 - 2.0 / p) * trS2 + trS_sq;
    double denom = (nf + 1.0 - 2.0 / p) * (trS2 - trS_sq / p);

    double rho;
    if (std::abs(denom) < 1e-12) {
      // Denominator ~0 means S ≈ c*I already, minimal shrinkage needed
      rho = 0.0;
    } else {
      rho = std::max(0.0, std::min(1.0, numer / denom));
    }

    // When n < p, sample covariance is rank-deficient (rank n-1 < p).
    // OAS can underestimate the needed shrinkage for singular matrices.
    // Floor ρ at (1 - n/p) to fill the rank gap: this ensures the
    // p-(n-1) zero eigenvalues get lifted to ρ * tr(S)/p > 0.
    if (n < dim) {
      double rho_floor = 1.0 - nf / p;
      rho = std::max(rho, rho_floor);
    }
    shrinkage_intensity = rho;

    // Step 3: Apply shrinkage: Σ = (1 - ρ) * S + ρ * (tr(S)/p) * I
    double target_diag = trS / p;  // Scaled identity target
    double one_minus_rho = 1.0 - rho;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        size_t idx = static_cast<size_t>(j) * dim + i;
        if (i == j) {
          cov[idx] = one_minus_rho * cov[idx] + rho * target_diag;
        } else {
          cov[idx] = one_minus_rho * cov[idx];
        }
      }
    }

    return cov;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

struct DualAveraging {
  double mu, log_epsilon_bar, H_bar;
  double gamma, t0, kappa;
  double target_accept;  // Target acceptance rate (dimension-adaptive)
  int m;

  // Default constructor with dimension-adaptive target
  // target_boost: additional boost to target acceptance for challenging models
  //               (e.g., +0.10 for MSGP+temporal combinations)
  DualAveraging(double epsilon_init = 1.0, int n_params = 1, double target_boost = 0.0);
  double update(double alpha);
  double final_epsilon() const;

  // Compute dimension-adaptive target acceptance rate
  // Higher targets (closer to 1) = smaller step sizes = fewer divergences
  // but slower exploration. Stan default is 0.80.
  // target_boost: additional boost for challenging model combinations
  static double compute_target(int n_params, double target_boost = 0.0) {
    // Use higher targets (0.75-0.85) to avoid divergences in
    // challenging models (ICAR, hierarchical negbin, etc.)
    double base_target;
    if (n_params <= 5) base_target = 0.85;
    else if (n_params <= 20) base_target = 0.82;
    else if (n_params <= 50) base_target = 0.80;
    else if (n_params <= 100) base_target = 0.78;
    else base_target = 0.75;

    // Apply boost, cap at 0.99
    return std::min(0.99, base_target + target_boost);
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
  std::vector<double> samples_flat;  // n_sample × n_params, row-major contiguous
  int n_params_stored = 0;
  std::vector<double> log_prob;
  std::vector<double> accept_prob;
  std::vector<int> n_leapfrog;
  std::vector<int> divergent;
  std::vector<int> treedepth;    // Actual tree depth per iteration (NUTS only)
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  int n_max_treedepth = 0;       // Count of iterations hitting max treedepth
  std::string sampler;           // Sampler name (e.g., "NUTS", "HMC", "NUTS->HMC(L=10)")

  // Row access for flat storage
  double* sample_row(int i) { return &samples_flat[i * n_params_stored]; }
  const double* sample_row(int i) const { return &samples_flat[i * n_params_stored]; }
};

// R-compatible result struct (create outside parallel regions)
struct HMCResult {
  Rcpp::NumericMatrix samples;
  Rcpp::NumericVector log_prob;
  Rcpp::NumericVector accept_prob;
  Rcpp::IntegerVector n_leapfrog;
  Rcpp::IntegerVector treedepth;
  Rcpp::IntegerVector divergent;
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  std::string sampler;
};

// Convert C++ result to R result (call outside parallel region)
inline HMCResult cpp_to_r_result(const HMCResultCpp& cpp_result, int n_params) {
  HMCResult r_result;
  r_result.samples = Rcpp::NumericMatrix(cpp_result.n_sample, n_params);
  r_result.log_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.accept_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.n_leapfrog = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.treedepth = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.divergent = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.epsilon = cpp_result.epsilon;
  r_result.n_warmup = cpp_result.n_warmup;
  r_result.n_sample = cpp_result.n_sample;
  r_result.chain_id = cpp_result.chain_id;
  r_result.sampler = cpp_result.sampler;

  for (int i = 0; i < cpp_result.n_sample; i++) {
    const double* row = cpp_result.sample_row(i);
    for (int j = 0; j < n_params; j++) {
      r_result.samples(i, j) = row[j];
    }
    r_result.log_prob[i] = cpp_result.log_prob[i];
    r_result.accept_prob[i] = cpp_result.accept_prob[i];
    r_result.n_leapfrog[i] = cpp_result.n_leapfrog[i];
    r_result.treedepth[i] = cpp_result.treedepth[i];
    r_result.divergent[i] = cpp_result.divergent[i];
  }

  return r_result;
}

// NUTS helper function declarations
double nuts_log_sum_exp(double a, double b);
double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n);
bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n);
LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout);
NUTSTreeResult build_tree(const NUTSNode& node, int direction, int depth,
                          double epsilon, const std::vector<double>& inv_mass,
                          bool use_mass, double H0, double delta_max,
                          const ModelData& data, const ParamLayout& layout,
                          std::mt19937& rng);

// Optimized NUTS: zero-allocation in-place leapfrog + buffer pool tree building
// Pointer-based Hamiltonian (no vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n);
// Pointer-based U-turn check
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n);
// In-place leapfrog step operating on workspace slot
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout);
// Zero-allocation recursive tree builder
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng);

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

// Mass-aware version: uses diagonal mass matrix for leapfrog and kinetic energy
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
);

// Dense-mass-aware version: uses full DenseMassMatrix for momentum, leapfrog, and kinetic energy
double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
);

// Run single HMC chain (C++ version - safe for parallel)
// riemannian: -1=auto (retry divergences with SoftAbs for BYM2/ICAR),
//              1=force on, 0=force off
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
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
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
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
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
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
);

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

// Compute full Hessian via finite differences of the H-mode gradient.
// H[i,j] = (grad_j(q + h*e_i) - grad_j(q)) / h
// Cost: (p+1) gradient evaluations.
void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h = 1e-5
);

// Compute SoftAbs metric from negative Hessian.
// G = Q diag(f(λ_i)) Q^T where f(λ) = λ * coth(α * λ)
// Returns G^{-1} and its Cholesky L. Returns false on failure.
bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
);

} // namespace ratiod_hmc

#endif // QUOTR_HMC_SAMPLER_H
