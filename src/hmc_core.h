// hmc_core.h
// Hamiltonian Monte Carlo with NUTS (No-U-Turn Sampler)
// Custom implementation for ratiod - no Stan dependency

#ifndef QUOTR_HMC_CORE_H
#define QUOTR_HMC_CORE_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "autodiff.h"

namespace ratiod {

// Forward declarations
struct HMCResult;
struct NUTSResult;

// ---------------------------------------------------------------------
// Log-probability and gradient computation for ratiod models
// ---------------------------------------------------------------------

// Model types
enum class ModelType {
  BINOMIAL,
  NEGBIN_NEGBIN,
  POISSON_GAMMA
};

// Model data container
struct ModelData {
  // Response data
  std::vector<int> y_num;       // Numerator counts
  std::vector<int> y_denom;     // Denominator (trials or counts)
  std::vector<double> y_denom_cont;  // For Gamma denominator

  // Design matrices
  Rcpp::NumericMatrix X_num;    // Numerator design matrix
  Rcpp::NumericMatrix X_denom;  // Denominator design matrix

  // Random effects indexing
  std::vector<int> re_group;    // RE group indices (1-based)
  int n_re_groups;

  // Dimensions
  int N;
  int p_num;
  int p_denom;

  // Prior parameters
  double sigma_beta;            // Prior SD for fixed effects
  double sigma_re_scale;        // Half-Cauchy scale for RE SD
  double phi_prior_shape;       // Gamma shape for overdispersion
  double phi_prior_rate;        // Gamma rate for overdispersion

  // Model type
  ModelType model_type;
};

// Parameter vector layout:
// [beta_num (p_num), beta_denom (p_denom), log_sigma_re, re (n_re), log_phi_num, log_phi_denom]

// Get parameter dimensions
inline int get_n_params(const ModelData& data) {
  int n = data.p_num + data.p_denom;
  if (data.n_re_groups > 0) {
    n += 1 + data.n_re_groups;  // log_sigma_re + RE values
  }
  if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    n += 2;  // log_phi_num, log_phi_denom
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    n += 1;  // log_shape (for Gamma)
  }
  return n;
}

// ---------------------------------------------------------------------
// Log-probability computation with autodiff
// ---------------------------------------------------------------------

// Compute log-posterior using autodiff
ad::Var compute_log_prob_ad(
    const std::vector<ad::Var>& params,
    const ModelData& data
);

// Compute log-posterior and gradient (returns gradient via out parameter)
double compute_log_prob_grad(
    const std::vector<double>& params,
    const ModelData& data,
    std::vector<double>& grad
);

// Just log-posterior (no gradient)
double compute_log_prob(
    const std::vector<double>& params,
    const ModelData& data
);

// ---------------------------------------------------------------------
// Leapfrog integrator
// ---------------------------------------------------------------------

struct LeapfrogResult {
  std::vector<double> q;  // Position
  std::vector<double> p;  // Momentum
  double log_prob;        // Log probability at new position
  bool divergent;         // Did we hit numerical issues?
};

LeapfrogResult leapfrog(
    const std::vector<double>& q_init,
    const std::vector<double>& p_init,
    double epsilon,
    int L,
    const ModelData& data
);

// Single leapfrog step
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data
);

// ---------------------------------------------------------------------
// NUTS: No-U-Turn Sampler
// ---------------------------------------------------------------------

struct NUTSState {
  std::vector<double> q;
  std::vector<double> p;
  double log_prob;
  std::vector<double> grad;
};

struct TreeResult {
  // Leftmost and rightmost states
  NUTSState left;
  NUTSState right;

  // Proposal state
  std::vector<double> q_proposal;
  double log_prob_proposal;

  // Tree statistics
  int n_valid;           // Number of valid states in tree
  bool stop;             // Should we stop building?
  double alpha;          // Acceptance probability sum
  int n_alpha;           // Number of acceptance probabilities
  int n_divergent;       // Number of divergent transitions in tree

  TreeResult() : n_valid(0), stop(false), alpha(0.0), n_alpha(0), n_divergent(0) {}
};

// Build NUTS tree recursively
TreeResult build_tree(
    const NUTSState& state,
    int direction,       // +1 or -1
    int depth,
    double epsilon,
    double log_u,        // Slice variable
    const ModelData& data,
    double delta_max,    // Max energy error
    std::mt19937& rng
);

// Check U-turn condition
bool check_uturn(
    const std::vector<double>& q_minus,
    const std::vector<double>& q_plus,
    const std::vector<double>& p_minus,
    const std::vector<double>& p_plus
);

// ---------------------------------------------------------------------
// Step size adaptation (dual averaging)
// ---------------------------------------------------------------------

struct DualAveraging {
  double mu;             // Target log step size
  double log_epsilon_bar;
  double H_bar;
  double gamma;
  double t0;
  double kappa;
  int m;                 // Iteration counter

  DualAveraging(double epsilon_init = 1.0)
    : mu(std::log(epsilon_init)),
      log_epsilon_bar(std::log(epsilon_init)),
      H_bar(0.0),
      gamma(0.05),
      t0(10.0),
      kappa(0.75),
      m(0) {}

  // Update step size based on acceptance probability
  double update(double alpha) {
    m++;
    double w = 1.0 / (m + t0);
    H_bar = (1.0 - w) * H_bar + w * (0.65 - alpha);  // Target 65%

    double log_epsilon = mu - std::sqrt(static_cast<double>(m)) / gamma * H_bar;
    // Clamp log_epsilon to reasonable range (epsilon between ~1e-4 and ~1.0)
    log_epsilon = std::max(-9.2, std::min(log_epsilon, 0.0));
    double epsilon = std::exp(log_epsilon);

    double m_w = std::pow(static_cast<double>(m), -kappa);
    log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;

    return epsilon;
  }

  double final_epsilon() const {
    return std::exp(log_epsilon_bar);
  }
};

// Find reasonable initial step size
double find_reasonable_epsilon(
    const std::vector<double>& q_init,
    const ModelData& data,
    std::mt19937& rng
);

// ---------------------------------------------------------------------
// Mass matrix adaptation
// ---------------------------------------------------------------------

struct MassMatrixAdapter {
  int dim;
  int n_samples;
  std::vector<double> mean;
  std::vector<double> M2;  // For Welford's algorithm
  std::vector<double> inv_mass;  // Diagonal inverse mass matrix

  MassMatrixAdapter(int d)
    : dim(d), n_samples(0), mean(d, 0.0), M2(d, 0.0), inv_mass(d, 1.0) {}

  void add_sample(const std::vector<double>& q) {
    n_samples++;
    for (int i = 0; i < dim; i++) {
      double delta = q[i] - mean[i];
      mean[i] += delta / n_samples;
      double delta2 = q[i] - mean[i];
      M2[i] += delta * delta2;
    }
  }

  void finalize() {
    if (n_samples > 1) {
      for (int i = 0; i < dim; i++) {
        double var = M2[i] / (n_samples - 1);
        inv_mass[i] = 1.0 / std::max(var, 1e-8);
      }
    }
  }

  void reset() {
    n_samples = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

// ---------------------------------------------------------------------
// HMC Result structure
// ---------------------------------------------------------------------

struct HMCResult {
  Rcpp::NumericMatrix samples;     // Parameter samples
  Rcpp::NumericVector log_prob;    // Log probabilities
  Rcpp::NumericVector accept_prob; // Acceptance probabilities
  Rcpp::IntegerVector n_leapfrog;  // Number of leapfrog steps (for NUTS)
  Rcpp::IntegerVector divergent;   // Divergent transitions
  double epsilon;                  // Final step size
  int n_warmup;
  int n_sample;
};

// ---------------------------------------------------------------------
// Main sampling functions
// ---------------------------------------------------------------------

// Standard HMC sampler
HMCResult hmc_sample(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    double epsilon,
    int L,
    bool adapt,
    bool verbose,
    unsigned int seed = 0
);

// NUTS sampler (preferred)
HMCResult nuts_sample(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int max_treedepth,
    bool adapt,
    bool verbose,
    unsigned int seed = 0
);

} // namespace ratiod

#endif // QUOTR_HMC_CORE_H
