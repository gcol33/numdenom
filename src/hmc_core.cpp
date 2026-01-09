// hmc_core.cpp
// Implementation of HMC/NUTS sampler for ratiod

#include "hmc_core.h"
#include <Rcpp.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace Rcpp;

namespace ratiod {

// ---------------------------------------------------------------------
// Log-probability computation
// ---------------------------------------------------------------------

// Compute log-posterior using autodiff for gradients
ad::Var compute_log_prob_ad(
    const std::vector<ad::Var>& params,
    const ModelData& data
) {
  using namespace ad;

  int idx = 0;

  // Extract parameters
  // Fixed effects for numerator
  std::vector<Var> beta_num(data.p_num);
  for (int j = 0; j < data.p_num; j++) {
    beta_num[j] = params[idx++];
  }

  // Fixed effects for denominator
  std::vector<Var> beta_denom(data.p_denom);
  for (int j = 0; j < data.p_denom; j++) {
    beta_denom[j] = params[idx++];
  }

  // Random effects
  Var log_sigma_re(0.0);
  std::vector<Var> re;
  if (data.n_re_groups > 0) {
    log_sigma_re = params[idx++];
    re.resize(data.n_re_groups);
    for (int g = 0; g < data.n_re_groups; g++) {
      re[g] = params[idx++];
    }
  }

  // Overdispersion parameters
  Var log_phi_num(0.0), log_phi_denom(0.0);
  if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    log_phi_num = params[idx++];
    log_phi_denom = params[idx++];
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    log_phi_num = params[idx++];  // Gamma shape parameter
  }

  // Transform constrained parameters
  Var sigma_re = exp(log_sigma_re);
  Var phi_num = exp(log_phi_num);
  Var phi_denom = exp(log_phi_denom);

  // Compute log-posterior
  Var log_post(0.0);

  // ----- PRIORS -----

  // Fixed effects: N(0, sigma_beta^2)
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.p_num; j++) {
    log_post = log_post - 0.5 * tau_beta * beta_num[j] * beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    log_post = log_post - 0.5 * tau_beta * beta_denom[j] * beta_denom[j];
  }

  // Random effects SD: Half-Cauchy(0, scale)
  if (data.n_re_groups > 0) {
    // log p(sigma) = -log(scale) - log(1 + (sigma/scale)^2) - log(pi/2)
    Var ratio = sigma_re / data.sigma_re_scale;
    log_post = log_post - log(1.0 + ratio * ratio);
    // Jacobian for log transform
    log_post = log_post + log_sigma_re;
  }

  // Random effects: N(0, sigma_re^2)
  if (data.n_re_groups > 0) {
    Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    for (int g = 0; g < data.n_re_groups; g++) {
      log_post = log_post - 0.5 * tau_re * re[g] * re[g];
      log_post = log_post + 0.5 * log(tau_re);
    }
    log_post = log_post - 0.5 * data.n_re_groups * std::log(2.0 * M_PI);
  }

  // Overdispersion: Gamma prior
  if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    // Gamma(shape, rate) on phi
    log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi_num
                        - data.phi_prior_rate * phi_num + log_phi_num;
    log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi_denom
                        - data.phi_prior_rate * phi_denom + log_phi_denom;
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi_num
                        - data.phi_prior_rate * phi_num + log_phi_num;
  }

  // ----- LIKELIHOOD -----

  for (int i = 0; i < data.N; i++) {
    // Linear predictor for numerator
    Var eta_num(0.0);
    for (int j = 0; j < data.p_num; j++) {
      eta_num = eta_num + data.X_num(i, j) * beta_num[j];
    }

    // Linear predictor for denominator
    Var eta_denom(0.0);
    for (int j = 0; j < data.p_denom; j++) {
      eta_denom = eta_denom + data.X_denom(i, j) * beta_denom[j];
    }

    // Add shared random effect (if present)
    if (data.n_re_groups > 0 && data.re_group[i] > 0) {
      int g = data.re_group[i] - 1;  // 0-based
      eta_num = eta_num + re[g];
      eta_denom = eta_denom + re[g];
    }

    // Compute likelihood contribution based on model type
    if (data.model_type == ModelType::BINOMIAL) {
      // Binomial: y_num ~ Binomial(y_denom, inv_logit(eta_num))
      // log p = y * eta - n * log(1 + exp(eta))
      int y = data.y_num[i];
      int n = data.y_denom[i];

      // Numerically stable log-likelihood
      Var log_lik = eta_num * y;
      log_lik = log_lik - n * softplus(eta_num);

      log_post = log_post + log_lik;

    } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
      // NegBin-NegBin: both numerator and denominator are counts
      // y_num ~ NegBin(exp(eta_num), phi_num)
      // y_denom ~ NegBin(exp(eta_denom), phi_denom)

      int y_n = data.y_num[i];
      int y_d = data.y_denom[i];

      Var mu_num = exp(eta_num);
      Var mu_denom = exp(eta_denom);

      // Numerator likelihood: lgamma(y+phi) - lgamma(phi) - lgamma(y+1)
      //                     + phi*log(phi/(mu+phi)) + y*log(mu/(mu+phi))
      // Note: lgamma(y+phi) needs gradient w.r.t phi, so use (phi + y) not Var(y + phi.val())
      Var log_lik_num = lgamma(phi_num + static_cast<double>(y_n)) - lgamma(phi_num)
                      + phi_num * log(phi_num / (mu_num + phi_num))
                      + static_cast<double>(y_n) * log(mu_num / (mu_num + phi_num));

      Var log_lik_denom = lgamma(phi_denom + static_cast<double>(y_d)) - lgamma(phi_denom)
                        + phi_denom * log(phi_denom / (mu_denom + phi_denom))
                        + static_cast<double>(y_d) * log(mu_denom / (mu_denom + phi_denom));

      log_post = log_post + log_lik_num + log_lik_denom;

    } else if (data.model_type == ModelType::POISSON_GAMMA) {
      // Poisson-Gamma: count / continuous effort
      // y_num ~ Poisson(exp(eta_num))
      // y_denom_cont ~ Gamma(shape, shape/exp(eta_denom))

      int y_n = data.y_num[i];
      double y_d = data.y_denom_cont[i];

      Var mu_num = exp(eta_num);
      Var mu_denom = exp(eta_denom);

      // Poisson log-likelihood: y*log(mu) - mu - lgamma(y+1)
      Var log_lik_num = y_n * eta_num - mu_num;

      // Gamma log-likelihood: shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape)
      // rate = shape / mu_denom
      Var rate = phi_num / mu_denom;
      Var log_lik_denom = phi_num * log(rate) + (phi_num - 1.0) * std::log(y_d)
                        - rate * y_d - lgamma(phi_num);

      log_post = log_post + log_lik_num + log_lik_denom;
    }
  }

  return log_post;
}

// Compute log-posterior and gradient
double compute_log_prob_grad(
    const std::vector<double>& params,
    const ModelData& data,
    std::vector<double>& grad
) {
  using namespace ad;

  // Initialize tape
  init_tape();

  // Create AD variables
  std::vector<Var> ad_params = make_vars(params);

  // Compute log-posterior
  Var log_post = compute_log_prob_ad(ad_params, data);

  // Backward pass
  log_post.backward();

  // Extract gradients
  grad = get_adjoints(ad_params);

  double result = log_post.val();

  // Clear tape
  clear_tape();

  return result;
}

// Just log-posterior (no gradient)
double compute_log_prob(
    const std::vector<double>& params,
    const ModelData& data
) {
  std::vector<double> dummy_grad;
  return compute_log_prob_grad(params, data, dummy_grad);
}

// ---------------------------------------------------------------------
// Leapfrog integrator
// ---------------------------------------------------------------------

LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_log_prob_grad(result.q, data, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  for (int i = 0; i < n; i++) {
    result.q[i] += epsilon * result.p[i];
  }

  // Half step for momentum
  result.log_prob = compute_log_prob_grad(result.q, data, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Check for numerical issues
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }

  return result;
}

LeapfrogResult leapfrog(
    const std::vector<double>& q_init,
    const std::vector<double>& p_init,
    double epsilon,
    int L,
    const ModelData& data
) {
  LeapfrogResult result;
  result.q = q_init;
  result.p = p_init;
  result.divergent = false;

  for (int l = 0; l < L; l++) {
    result = leapfrog_step(result.q, result.p, epsilon, data);
    if (result.divergent) break;
  }

  return result;
}

// ---------------------------------------------------------------------
// NUTS implementation
// ---------------------------------------------------------------------

bool check_uturn(
    const std::vector<double>& q_minus,
    const std::vector<double>& q_plus,
    const std::vector<double>& p_minus,
    const std::vector<double>& p_plus
) {
  int n = q_minus.size();
  double forward = 0.0, backward = 0.0;

  for (int i = 0; i < n; i++) {
    double diff = q_plus[i] - q_minus[i];
    forward += diff * p_plus[i];
    backward += diff * p_minus[i];
  }

  return (forward < 0.0) || (backward < 0.0);
}

TreeResult build_tree(
    const NUTSState& state,
    int direction,
    int depth,
    double epsilon,
    double log_u,
    const ModelData& data,
    double delta_max,
    std::mt19937& rng
) {
  TreeResult result;

  if (depth == 0) {
    // Base case: single leapfrog step
    LeapfrogResult lf = leapfrog_step(state.q, state.p, direction * epsilon, data);

    // Kinetic energy
    double kinetic = 0.0;
    for (size_t i = 0; i < lf.p.size(); i++) {
      kinetic += 0.5 * lf.p[i] * lf.p[i];
    }
    double H_new = -lf.log_prob + kinetic;

    double kinetic_init = 0.0;
    for (size_t i = 0; i < state.p.size(); i++) {
      kinetic_init += 0.5 * state.p[i] * state.p[i];
    }
    double H_init = -state.log_prob + kinetic_init;

    // Slice variable check
    bool valid = (log_u <= -H_new);

    // Divergence check
    bool divergent = (H_new - H_init > delta_max) || lf.divergent;

    // Set up result
    std::vector<double> grad_new(lf.q.size());
    double lp_new = compute_log_prob_grad(lf.q, data, grad_new);

    NUTSState new_state;
    new_state.q = lf.q;
    new_state.p = lf.p;
    new_state.log_prob = lp_new;
    new_state.grad = grad_new;

    result.left = new_state;
    result.right = new_state;
    result.q_proposal = lf.q;
    result.log_prob_proposal = lp_new;
    result.n_valid = valid ? 1 : 0;
    result.stop = divergent;
    result.n_divergent = divergent ? 1 : 0;

    // Acceptance probability
    double alpha = std::min(1.0, std::exp(H_init - H_new));
    if (!std::isfinite(alpha)) alpha = 0.0;
    result.alpha = alpha;
    result.n_alpha = 1;

    return result;
  }

  // Recursive case: build two subtrees
  TreeResult sub1 = build_tree(state, direction, depth - 1, epsilon, log_u, data, delta_max, rng);

  if (sub1.stop) {
    return sub1;
  }

  // Extend in same direction
  NUTSState extend_from = (direction > 0) ? sub1.right : sub1.left;
  TreeResult sub2 = build_tree(extend_from, direction, depth - 1, epsilon, log_u, data, delta_max, rng);

  // Combine results
  result.n_valid = sub1.n_valid + sub2.n_valid;
  result.alpha = sub1.alpha + sub2.alpha;
  result.n_alpha = sub1.n_alpha + sub2.n_alpha;
  result.n_divergent = sub1.n_divergent + sub2.n_divergent;

  // Biased progressive sampling
  if (result.n_valid > 0) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double prob = static_cast<double>(sub2.n_valid) / result.n_valid;
    if (unif(rng) < prob) {
      result.q_proposal = sub2.q_proposal;
      result.log_prob_proposal = sub2.log_prob_proposal;
    } else {
      result.q_proposal = sub1.q_proposal;
      result.log_prob_proposal = sub1.log_prob_proposal;
    }
  } else {
    result.q_proposal = sub1.q_proposal;
    result.log_prob_proposal = sub1.log_prob_proposal;
  }

  // Update endpoints
  if (direction > 0) {
    result.left = sub1.left;
    result.right = sub2.right;
  } else {
    result.left = sub2.left;
    result.right = sub1.right;
  }

  // Check for U-turn
  result.stop = sub2.stop || check_uturn(
    result.left.q, result.right.q,
    result.left.p, result.right.p
  );

  return result;
}

// Find reasonable initial step size
double find_reasonable_epsilon(
    const std::vector<double>& q_init,
    const ModelData& data,
    std::mt19937& rng
) {
  int n = q_init.size();

  // Start with epsilon = 1
  double epsilon = 1.0;

  // Sample random momentum
  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<double> p(n);
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng);
  }

  // Compute initial Hamiltonian
  std::vector<double> grad(n);
  double log_prob_init = compute_log_prob_grad(q_init, data, grad);
  double kinetic_init = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_init += 0.5 * p[i] * p[i];
  }
  double H_init = -log_prob_init + kinetic_init;

  // Single leapfrog step
  LeapfrogResult lf = leapfrog_step(q_init, p, epsilon, data);

  double kinetic_new = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_new += 0.5 * lf.p[i] * lf.p[i];
  }
  double H_new = -lf.log_prob + kinetic_new;

  double delta_H = H_new - H_init;
  int direction = (delta_H > std::log(0.5)) ? -1 : 1;

  // Double or halve epsilon until acceptance prob crosses 0.5
  while (true) {
    if (direction > 0) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon > 1e6 || epsilon < 1e-6) break;

    lf = leapfrog_step(q_init, p, epsilon, data);
    kinetic_new = 0.0;
    for (int i = 0; i < n; i++) {
      kinetic_new += 0.5 * lf.p[i] * lf.p[i];
    }
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    if (direction > 0 && delta_H > std::log(0.5)) break;
    if (direction < 0 && delta_H < std::log(0.5)) break;
  }

  return std::max(1e-4, std::min(epsilon, 1.0));
}

// ---------------------------------------------------------------------
// Main NUTS sampler
// ---------------------------------------------------------------------

HMCResult nuts_sample(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int max_treedepth,
    bool adapt,
    bool verbose,
    unsigned int seed
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;

  HMCResult result;
  result.samples = NumericMatrix(n_sample, n_params);
  result.log_prob = NumericVector(n_sample);
  result.accept_prob = NumericVector(n_sample);
  result.n_leapfrog = IntegerVector(n_sample);
  result.divergent = IntegerVector(n_sample);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;

  // Initialize RNG with seed (0 means use random device)
  std::mt19937 rng;
  if (seed == 0) {
    std::random_device rd;
    rng.seed(rd());
  } else {
    rng.seed(seed);
  }
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Current state
  std::vector<double> q = q_init;
  std::vector<double> grad(n_params);
  double log_prob_current = compute_log_prob_grad(q, data, grad);

  // Step size adaptation
  double epsilon = adapt ? find_reasonable_epsilon(q, data, rng) : 0.1;
  DualAveraging da(epsilon);

  // Mass matrix adaptation
  MassMatrixAdapter mass_adapter(n_params);

  double delta_max = 1000.0;  // Max energy difference for divergence

  int sample_idx = 0;
  int n_divergent = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);

    // Sample momentum
    std::vector<double> p(n_params);
    for (int i = 0; i < n_params; i++) {
      p[i] = normal(rng);
    }

    // Initial Hamiltonian
    double kinetic = 0.0;
    for (int i = 0; i < n_params; i++) {
      kinetic += 0.5 * p[i] * p[i];
    }
    double H_init = -log_prob_current + kinetic;

    // Slice variable
    double log_u = std::log(unif(rng)) - H_init;

    // Initialize tree
    NUTSState init_state;
    init_state.q = q;
    init_state.p = p;
    init_state.log_prob = log_prob_current;
    init_state.grad = grad;

    NUTSState left = init_state;
    NUTSState right = init_state;

    std::vector<double> q_proposal = q;
    double log_prob_proposal = log_prob_current;
    int n_valid = 1;
    bool stop = false;
    double sum_alpha = 0.0;
    int n_alpha = 0;
    int tree_depth = 0;
    int iter_n_divergent = 0;  // Divergences in this iteration

    // Build tree
    while (!stop && tree_depth < max_treedepth) {
      // Choose direction
      int direction = (unif(rng) < 0.5) ? -1 : 1;

      // Build subtree
      NUTSState extend_from = (direction > 0) ? right : left;
      TreeResult subtree = build_tree(extend_from, direction, tree_depth,
                                      epsilon, log_u, data, delta_max, rng);

      if (!subtree.stop) {
        // Accept proposal with probability n_valid_subtree / n_valid_total
        double accept_prob = static_cast<double>(subtree.n_valid) /
                            (n_valid + subtree.n_valid);
        if (unif(rng) < accept_prob) {
          q_proposal = subtree.q_proposal;
          log_prob_proposal = subtree.log_prob_proposal;
        }
      }

      // Update tree endpoints
      if (direction > 0) {
        right = subtree.right;
      } else {
        left = subtree.left;
      }

      n_valid += subtree.n_valid;
      sum_alpha += subtree.alpha;
      n_alpha += subtree.n_alpha;
      iter_n_divergent += subtree.n_divergent;
      stop = subtree.stop || check_uturn(left.q, right.q, left.p, right.p);

      tree_depth++;
    }

    // Track divergences (at least one in tree = this transition is divergent)
    bool divergent = (iter_n_divergent > 0);
    if (divergent) n_divergent++;

    // Accept/reject
    q = q_proposal;
    log_prob_current = compute_log_prob_grad(q, data, grad);

    // Adaptation during warmup
    if (is_warmup && adapt) {
      double avg_alpha = (n_alpha > 0) ? sum_alpha / n_alpha : 0.0;
      epsilon = da.update(avg_alpha);
      mass_adapter.add_sample(q);
    }

    // Store sample (after warmup)
    if (!is_warmup) {
      for (int i = 0; i < n_params; i++) {
        result.samples(sample_idx, i) = q[i];
      }
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = (n_alpha > 0) ? sum_alpha / n_alpha : 0.0;
      result.n_leapfrog[sample_idx] = (1 << tree_depth) - 1;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      sample_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 100 == 0) {
      double avg_accept = (n_alpha > 0) ? sum_alpha / n_alpha : 0.0;
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter;
      if (is_warmup) {
        Rcpp::Rcout << " (warmup)";
      }
      Rcpp::Rcout << " - epsilon: " << epsilon
                  << ", accept: " << avg_accept
                  << ", depth: " << tree_depth
                  << std::endl;
    }
  }

  // Final step size
  result.epsilon = adapt ? da.final_epsilon() : epsilon;

  if (verbose) {
    Rcpp::Rcout << "Sampling complete. Final epsilon: " << result.epsilon
                << ", divergent: " << n_divergent << std::endl;
  }

  return result;
}

// ---------------------------------------------------------------------
// Standard HMC sampler (simpler, fixed trajectory length)
// ---------------------------------------------------------------------

HMCResult hmc_sample(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    double epsilon,
    int L,
    bool adapt,
    bool verbose,
    unsigned int seed
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;

  HMCResult result;
  result.samples = NumericMatrix(n_sample, n_params);
  result.log_prob = NumericVector(n_sample);
  result.accept_prob = NumericVector(n_sample);
  result.n_leapfrog = IntegerVector(n_sample, L);
  result.divergent = IntegerVector(n_sample, 0);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;

  // Initialize RNG with seed (0 means use random device)
  std::mt19937 rng;
  if (seed == 0) {
    std::random_device rd;
    rng.seed(rd());
  } else {
    rng.seed(seed);
  }
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Current state
  std::vector<double> q = q_init;
  std::vector<double> grad(n_params);
  double log_prob_current = compute_log_prob_grad(q, data, grad);

  // Step size adaptation
  DualAveraging da(epsilon);

  int sample_idx = 0;
  int n_accept = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);

    // Sample momentum
    std::vector<double> p(n_params);
    for (int i = 0; i < n_params; i++) {
      p[i] = normal(rng);
    }

    // Current Hamiltonian
    double kinetic_current = 0.0;
    for (int i = 0; i < n_params; i++) {
      kinetic_current += 0.5 * p[i] * p[i];
    }
    double H_current = -log_prob_current + kinetic_current;

    // Leapfrog integration
    LeapfrogResult lf = leapfrog(q, p, epsilon, L, data);

    // Proposed Hamiltonian
    double kinetic_proposed = 0.0;
    for (int i = 0; i < n_params; i++) {
      kinetic_proposed += 0.5 * lf.p[i] * lf.p[i];
    }
    double H_proposed = -lf.log_prob + kinetic_proposed;

    // Metropolis accept/reject
    double alpha = std::min(1.0, std::exp(H_current - H_proposed));
    if (!std::isfinite(alpha)) alpha = 0.0;

    bool accepted = (unif(rng) < alpha);
    if (accepted && !lf.divergent) {
      q = lf.q;
      log_prob_current = lf.log_prob;
      n_accept++;
    }

    // Adaptation
    if (is_warmup && adapt) {
      epsilon = da.update(alpha);
    }

    // Store sample
    if (!is_warmup) {
      for (int i = 0; i < n_params; i++) {
        result.samples(sample_idx, i) = q[i];
      }
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.divergent[sample_idx] = lf.divergent ? 1 : 0;
      sample_idx++;
    }

    if (verbose && (iter + 1) % 100 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter;
      if (is_warmup) Rcpp::Rcout << " (warmup)";
      Rcpp::Rcout << " - epsilon: " << epsilon
                  << ", accept rate: " << (double)n_accept / (iter + 1)
                  << std::endl;
    }
  }

  result.epsilon = adapt ? da.final_epsilon() : epsilon;

  return result;
}

} // namespace ratiod

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_nuts_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    std::string model_type,
    double sigma_beta = 10.0,
    double sigma_re_scale = 2.5,
    double phi_prior_shape = 1.0,
    double phi_prior_rate = 0.01,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    bool adapt = true,
    bool verbose = true,
    unsigned int seed = 0
) {
  // Set up model data
  ratiod::ModelData data;
  data.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());
  data.X_num = X_num;
  data.X_denom = X_denom;
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;
  data.N = y_num.size();
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;

  // Parse model type
  if (model_type == "binomial") {
    data.model_type = ratiod::ModelType::BINOMIAL;
  } else if (model_type == "negbin_negbin") {
    data.model_type = ratiod::ModelType::NEGBIN_NEGBIN;
  } else if (model_type == "poisson_gamma") {
    data.model_type = ratiod::ModelType::POISSON_GAMMA;
  } else {
    Rcpp::stop("Unknown model type: " + model_type);
  }

  // Initial parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Run NUTS
  ratiod::HMCResult result = ratiod::nuts_sample(
    q0, data, n_iter, n_warmup, max_treedepth, adapt, verbose, seed
  );

  return Rcpp::List::create(
    Rcpp::Named("samples") = result.samples,
    Rcpp::Named("log_prob") = result.log_prob,
    Rcpp::Named("accept_prob") = result.accept_prob,
    Rcpp::Named("n_leapfrog") = result.n_leapfrog,
    Rcpp::Named("divergent") = result.divergent,
    Rcpp::Named("epsilon") = result.epsilon,
    Rcpp::Named("n_warmup") = result.n_warmup,
    Rcpp::Named("n_sample") = result.n_sample
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_hmc_basic(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    std::string model_type,
    double sigma_beta = 10.0,
    double sigma_re_scale = 2.5,
    double phi_prior_shape = 1.0,
    double phi_prior_rate = 0.01,
    int n_iter = 2000,
    int n_warmup = 1000,
    double epsilon = 0.1,
    int L = 10,
    bool adapt = true,
    bool verbose = true,
    unsigned int seed = 0
) {
  // Set up model data
  ratiod::ModelData data;
  data.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());
  data.X_num = X_num;
  data.X_denom = X_denom;
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;
  data.N = y_num.size();
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;

  // Parse model type
  if (model_type == "binomial") {
    data.model_type = ratiod::ModelType::BINOMIAL;
  } else if (model_type == "negbin_negbin") {
    data.model_type = ratiod::ModelType::NEGBIN_NEGBIN;
  } else if (model_type == "poisson_gamma") {
    data.model_type = ratiod::ModelType::POISSON_GAMMA;
  } else {
    Rcpp::stop("Unknown model type: " + model_type);
  }

  // Initial parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Run HMC
  ratiod::HMCResult result = ratiod::hmc_sample(
    q0, data, n_iter, n_warmup, epsilon, L, adapt, verbose, seed
  );

  return Rcpp::List::create(
    Rcpp::Named("samples") = result.samples,
    Rcpp::Named("log_prob") = result.log_prob,
    Rcpp::Named("accept_prob") = result.accept_prob,
    Rcpp::Named("n_leapfrog") = result.n_leapfrog,
    Rcpp::Named("divergent") = result.divergent,
    Rcpp::Named("epsilon") = result.epsilon,
    Rcpp::Named("n_warmup") = result.n_warmup,
    Rcpp::Named("n_sample") = result.n_sample
  );
}
