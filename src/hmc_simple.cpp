// hmc_simple.cpp
// Simplified HMC/NUTS implementation using numerical gradients
// More stable than autodiff version, suitable for production use

#include <Rcpp.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

using namespace Rcpp;

namespace quotr_hmc_simple {

// Model types
enum class ModelType { BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA };

// Model data container
struct ModelData {
  std::vector<int> y_num;
  std::vector<int> y_denom;
  std::vector<double> y_denom_cont;
  NumericMatrix X_num;
  NumericMatrix X_denom;
  std::vector<int> re_group;
  int n_re_groups;
  int N, p_num, p_denom;
  double sigma_beta, sigma_re_scale;
  double phi_prior_shape, phi_prior_rate;
  ModelType model_type;
};

// ---------------------------------------------------------------------
// Likelihood functions (no autodiff, just compute values)
// ---------------------------------------------------------------------

double log_lik_binomial(int y, int n, double eta) {
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    return y * eta - n * std::log(1.0 + std::exp(eta));
  }
}

double log_lik_negbin(int y, double mu, double phi) {
  if (mu <= 0 || phi <= 0) return -1e10;
  return R::lgammafn(y + phi) - R::lgammafn(phi) - R::lgammafn(y + 1.0)
       + phi * std::log(phi / (mu + phi))
       + y * std::log(mu / (mu + phi));
}

double log_lik_poisson(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - R::lgammafn(y + 1.0);
}

double log_lik_gamma(double y, double shape, double mu) {
  if (y <= 0 || shape <= 0 || mu <= 0) return -1e10;
  double rate = shape / mu;
  return shape * std::log(rate) + (shape - 1.0) * std::log(y) - rate * y - R::lgammafn(shape);
}

// ---------------------------------------------------------------------
// Log-posterior computation
// ---------------------------------------------------------------------

double compute_log_post(const std::vector<double>& params, const ModelData& data) {
  int idx = 0;

  // Extract fixed effects
  std::vector<double> beta_num(data.p_num);
  for (int j = 0; j < data.p_num; j++) {
    beta_num[j] = params[idx++];
  }

  std::vector<double> beta_denom(data.p_denom);
  for (int j = 0; j < data.p_denom; j++) {
    beta_denom[j] = params[idx++];
  }

  // Random effects
  double log_sigma_re = 0.0;
  std::vector<double> re;
  if (data.n_re_groups > 0) {
    log_sigma_re = params[idx++];
    re.resize(data.n_re_groups);
    for (int g = 0; g < data.n_re_groups; g++) {
      re[g] = params[idx++];
    }
  }

  // Overdispersion
  double log_phi_num = 0.0, log_phi_denom = 0.0;
  if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    log_phi_num = params[idx++];
    log_phi_denom = params[idx++];
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    log_phi_num = params[idx++];
  }

  double sigma_re = std::exp(log_sigma_re);
  double phi_num = std::exp(log_phi_num);
  double phi_denom = std::exp(log_phi_denom);

  double log_post = 0.0;

  // ----- PRIORS -----

  // Fixed effects: N(0, sigma_beta^2)
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.p_num; j++) {
    log_post -= 0.5 * tau_beta * beta_num[j] * beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    log_post -= 0.5 * tau_beta * beta_denom[j] * beta_denom[j];
  }

  // Random effects SD: Half-Cauchy
  if (data.n_re_groups > 0) {
    double ratio = sigma_re / data.sigma_re_scale;
    log_post -= std::log(1.0 + ratio * ratio);
    log_post += log_sigma_re;  // Jacobian
  }

  // Random effects: N(0, sigma_re^2)
  if (data.n_re_groups > 0) {
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    for (int g = 0; g < data.n_re_groups; g++) {
      log_post -= 0.5 * tau_re * re[g] * re[g];
      log_post += 0.5 * std::log(tau_re);
    }
    log_post -= 0.5 * data.n_re_groups * std::log(2.0 * M_PI);
  }

  // Overdispersion priors: Gamma
  if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_num - data.phi_prior_rate * phi_num + log_phi_num;
    log_post += (data.phi_prior_shape - 1.0) * log_phi_denom - data.phi_prior_rate * phi_denom + log_phi_denom;
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_num - data.phi_prior_rate * phi_num + log_phi_num;
  }

  // ----- LIKELIHOOD -----

  for (int i = 0; i < data.N; i++) {
    // Linear predictors
    double eta_num = 0.0, eta_denom = 0.0;
    for (int j = 0; j < data.p_num; j++) {
      eta_num += data.X_num(i, j) * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
      eta_denom += data.X_denom(i, j) * beta_denom[j];
    }

    // Add random effect
    if (data.n_re_groups > 0 && data.re_group[i] > 0) {
      int g = data.re_group[i] - 1;
      eta_num += re[g];
      eta_denom += re[g];
    }

    // Likelihood
    if (data.model_type == ModelType::BINOMIAL) {
      log_post += log_lik_binomial(data.y_num[i], data.y_denom[i], eta_num);
    } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);
      log_post += log_lik_negbin(data.y_num[i], mu_num, phi_num);
      log_post += log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
    } else {
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);
      log_post += log_lik_poisson(data.y_num[i], mu_num);
      log_post += log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
    }
  }

  return log_post;
}

// Numerical gradient
void compute_gradient(const std::vector<double>& params, const ModelData& data,
                      std::vector<double>& grad) {
  int n = params.size();
  grad.resize(n);

  double h = 1e-5;
  std::vector<double> params_plus = params;
  std::vector<double> params_minus = params;

  for (int i = 0; i < n; i++) {
    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = compute_log_post(params_plus, data);
    double f_minus = compute_log_post(params_minus, data);

    grad[i] = (f_plus - f_minus) / (2.0 * h);

    params_plus[i] = params[i];
    params_minus[i] = params[i];
  }
}

// ---------------------------------------------------------------------
// Leapfrog integrator
// ---------------------------------------------------------------------

struct LeapfrogResult {
  std::vector<double> q, p;
  double log_prob;
  bool divergent;
};

LeapfrogResult leapfrog_step(const std::vector<double>& q, const std::vector<double>& p,
                              double epsilon, const ModelData& data) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_gradient(result.q, data, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  for (int i = 0; i < n; i++) {
    result.q[i] += epsilon * result.p[i];
  }

  // Half step for momentum
  result.log_prob = compute_log_post(result.q, data);
  compute_gradient(result.q, data, grad);
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

// ---------------------------------------------------------------------
// Dual averaging for step size adaptation
// ---------------------------------------------------------------------

struct DualAveraging {
  double mu, log_epsilon_bar, H_bar;
  double gamma, t0, kappa;
  int m;

  DualAveraging(double epsilon_init = 1.0)
    : mu(std::log(epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
      gamma(0.05), t0(10.0), kappa(0.75), m(0) {}

  double update(double alpha) {
    m++;
    double w = 1.0 / (m + t0);
    H_bar = (1.0 - w) * H_bar + w * (0.65 - alpha);
    double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
    // Clamp log_epsilon to reasonable range (epsilon between ~1e-4 and ~1.0)
    log_epsilon = std::max(-9.2, std::min(log_epsilon, 0.0));
    double epsilon = std::exp(log_epsilon);
    double m_w = std::pow((double)m, -kappa);
    log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
    return epsilon;
  }

  double final_epsilon() const { return std::exp(log_epsilon_bar); }
};

// Find reasonable initial step size
double find_reasonable_epsilon(const std::vector<double>& q, const ModelData& data,
                                std::mt19937& rng) {
  int n = q.size();
  double epsilon = 0.1;

  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<double> p(n);
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng);
  }

  double log_prob_init = compute_log_post(q, data);
  double kinetic_init = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_init += 0.5 * p[i] * p[i];
  }
  double H_init = -log_prob_init + kinetic_init;

  LeapfrogResult lf = leapfrog_step(q, p, epsilon, data);
  double kinetic_new = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_new += 0.5 * lf.p[i] * lf.p[i];
  }
  double H_new = -lf.log_prob + kinetic_new;

  double delta_H = H_new - H_init;
  int direction = (delta_H > std::log(0.5)) ? -1 : 1;

  while (true) {
    epsilon *= (direction > 0) ? 2.0 : 0.5;
    if (epsilon > 1e3 || epsilon < 1e-6) break;

    lf = leapfrog_step(q, p, epsilon, data);
    kinetic_new = 0.0;
    for (int i = 0; i < n; i++) {
      kinetic_new += 0.5 * lf.p[i] * lf.p[i];
    }
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    if (direction > 0 && delta_H > std::log(0.5)) break;
    if (direction < 0 && delta_H < std::log(0.5)) break;
  }

  // Clamp to a more conservative range
  return std::max(1e-4, std::min(epsilon, 0.05));
}

} // namespace quotr_hmc_simple

// ---------------------------------------------------------------------
// Simple HMC sampler (fixed trajectory length)
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_hmc_simple(
    NumericVector q_init,
    IntegerVector y_num,
    IntegerVector y_denom,
    NumericVector y_denom_cont,
    NumericMatrix X_num,
    NumericMatrix X_denom,
    IntegerVector re_group,
    int n_re_groups,
    std::string model_type_str,
    double sigma_beta = 10.0,
    double sigma_re_scale = 2.5,
    double phi_prior_shape = 2.0,
    double phi_prior_rate = 0.1,
    int n_iter = 2000,
    int n_warmup = 1000,
    int L = 20,
    bool adapt = true,
    bool verbose = true,
    unsigned int seed = 0
) {
  // Set up model data
  quotr_hmc_simple::ModelData data;
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

  if (model_type_str == "binomial") {
    data.model_type = quotr_hmc_simple::ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.model_type = quotr_hmc_simple::ModelType::NEGBIN_NEGBIN;
  } else {
    data.model_type = quotr_hmc_simple::ModelType::POISSON_GAMMA;
  }

  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;

  NumericMatrix samples(n_sample, n_params);
  NumericVector log_prob_vec(n_sample);
  NumericVector accept_prob_vec(n_sample);
  IntegerVector n_leapfrog_vec(n_sample, L);
  IntegerVector divergent_vec(n_sample, 0);

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

  std::vector<double> q(q_init.begin(), q_init.end());
  double log_prob_current = quotr_hmc_simple::compute_log_post(q, data);

  // Find initial step size
  double epsilon = adapt ? quotr_hmc_simple::find_reasonable_epsilon(q, data, rng) : 0.01;
  quotr_hmc_simple::DualAveraging da(epsilon);

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
    std::vector<double> q_prop = q;
    std::vector<double> p_prop = p;
    bool divergent = false;

    for (int l = 0; l < L; l++) {
      quotr_hmc_simple::LeapfrogResult lf = quotr_hmc_simple::leapfrog_step(q_prop, p_prop, epsilon, data);
      q_prop = lf.q;
      p_prop = lf.p;
      if (lf.divergent) {
        divergent = true;
        break;
      }
    }

    double log_prob_prop = quotr_hmc_simple::compute_log_post(q_prop, data);
    double kinetic_prop = 0.0;
    for (int i = 0; i < n_params; i++) {
      kinetic_prop += 0.5 * p_prop[i] * p_prop[i];
    }
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

    // Adaptation
    if (is_warmup && adapt) {
      epsilon = da.update(alpha);
    }

    // Store sample
    if (!is_warmup) {
      for (int i = 0; i < n_params; i++) {
        samples(sample_idx, i) = q[i];
      }
      log_prob_vec[sample_idx] = log_prob_current;
      accept_prob_vec[sample_idx] = alpha;
      divergent_vec[sample_idx] = divergent ? 1 : 0;
      sample_idx++;
    }

    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter;
      if (is_warmup) Rcpp::Rcout << " (warmup)";
      Rcpp::Rcout << " - epsilon: " << epsilon
                  << ", accept rate: " << (double)n_accept / (iter + 1)
                  << std::endl;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("samples") = samples,
    Rcpp::Named("log_prob") = log_prob_vec,
    Rcpp::Named("accept_prob") = accept_prob_vec,
    Rcpp::Named("n_leapfrog") = n_leapfrog_vec,
    Rcpp::Named("divergent") = divergent_vec,
    Rcpp::Named("epsilon") = adapt ? da.final_epsilon() : epsilon,
    Rcpp::Named("n_warmup") = n_warmup,
    Rcpp::Named("n_sample") = n_sample
  );
}
