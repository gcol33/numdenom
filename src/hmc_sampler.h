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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace quotr_hmc {

using quotr_temporal::TemporalType;
using quotr_temporal::TemporalData;
using quotr_zi::ZIType;

// =====================================================================
// Model configuration
// =====================================================================

enum class ModelType { BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA };
enum class SpatialType { NONE, ICAR, BYM2 };

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

  // Random effects
  std::vector<int> re_group;  // 1-based group index (0 = no RE)
  int n_re_groups;

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
  int log_sigma_re_idx;
  int re_start, re_end;
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

  int total_params;

  bool has_re;
  bool has_phi_num;
  bool has_phi_denom;
  bool has_spatial;
  bool is_bym2;
  bool has_temporal;
  bool is_ar1;
  bool has_zi;
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
  int m;

  DualAveraging(double epsilon_init = 1.0);
  double update(double alpha);
  double final_epsilon() const;
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

} // namespace quotr_hmc

#endif // QUOTR_HMC_SAMPLER_H
