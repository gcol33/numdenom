// hmc_sampler.cpp
// Full HMC/NUTS backend with spatial, temporal, and ZI support
// Provides Stan-free Bayesian inference for all quotr models

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include "hmc_progress.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace quotr_hmc {

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

  // Random effects
  layout.has_re = (data.n_re_groups > 0);
  if (layout.has_re) {
    layout.log_sigma_re_idx = idx++;
    layout.re_start = idx;
    idx += data.n_re_groups;
    layout.re_end = idx;
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

  // Spatial effects
  layout.has_spatial = (data.spatial_type != SpatialType::NONE);
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

  // Random effects SD: Half-Cauchy(0, scale)
  if (layout.has_re) {
    double ratio = sigma_re / data.sigma_re_scale;
    log_post -= std::log(1.0 + ratio * ratio);
    log_post += log_sigma_re;  // Jacobian

    // Random effects: N(0, sigma_re^2)
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    for (int g = 0; g < data.n_re_groups; g++) {
      log_post -= 0.5 * tau_re * re[g] * re[g];
      log_post += 0.5 * std::log(tau_re);
    }
    log_post -= 0.5 * data.n_re_groups * std::log(2.0 * M_PI);
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
        double quad = quotr_temporal::rw1_quadratic_form(phi_g, T, data.temporal_cyclic);
        int rank = data.temporal_cyclic ? T : T - 1;
        log_post += 0.5 * rank * log_tau_temporal - 0.5 * tau_temporal * quad;
        // Soft sum-to-zero constraint
        log_post += quotr_temporal::sum_to_zero_penalty(phi_g, T, 0.001);

      } else if (data.temporal_type == TemporalType::RW2) {
        double quad = quotr_temporal::rw2_quadratic_form(phi_g, T, data.temporal_cyclic);
        int rank = data.temporal_cyclic ? T : T - 2;
        log_post += 0.5 * rank * log_tau_temporal - 0.5 * tau_temporal * quad;
        // Soft sum-to-zero constraint
        log_post += quotr_temporal::sum_to_zero_penalty(phi_g, T, 0.001);

      } else if (data.temporal_type == TemporalType::AR1) {
        log_post += quotr_temporal::ar1_log_density(phi_g, T, rho_ar1, tau_temporal);
      }
    }
  }

  // ============ LIKELIHOOD (parallelized) ============

  double log_lik = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:log_lik) schedule(static) \
          num_threads(data.n_threads)
  #endif
  for (int i = 0; i < data.N; i++) {
    // Linear predictor for numerator (using optimized dot product)
    double eta_num = quotr_linalg::dot_product(
        &data.X_num_flat[i * data.p_num], beta_num, data.p_num);

    // Linear predictor for denominator (using optimized dot product)
    double eta_denom = quotr_linalg::dot_product(
        &data.X_denom_flat[i * data.p_denom], beta_denom, data.p_denom);

    // Add random effect (shared between num and denom)
    if (layout.has_re && data.re_group[i] > 0) {
      int g = data.re_group[i] - 1;
      eta_num += re[g];
      eta_denom += re[g];
    }

    // Add spatial effect
    if (layout.has_spatial && data.spatial_group[i] > 0) {
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
    if (layout.has_temporal && data.temporal_time_idx[i] > 0) {
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

    // Compute ZI linear predictor if applicable (using optimized dot product)
    double logit_zi = 0.0;
    if (layout.has_zi) {
      logit_zi = quotr_linalg::dot_product(
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
        ll_i = quotr_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
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
        ll_i = quotr_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
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
// Numerical gradient (parallelized)
// =====================================================================

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
  int n = params.size();
  grad.resize(n);

  double h = 1e-5;

  // Parallelize gradient computation
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(data.n_threads)
  #endif
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
// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init)
  : mu(std::log(epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (0.65 - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range (epsilon between 1e-5 and 0.05)
  log_epsilon = std::max(-12.0, std::min(log_epsilon, -3.0));
  double epsilon = std::exp(log_epsilon);
  double m_w = std::pow((double)m, -kappa);
  log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
  return epsilon;
}

double DualAveraging::final_epsilon() const {
  return std::exp(log_epsilon_bar);
}

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

double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
) {
  int n = q.size();
  double epsilon = 0.1;

  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<double> p(n);
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng);
  }

  double log_prob_init = compute_log_post(q, data, layout);
  double kinetic_init = 0.5 * quotr_linalg::norm_squared(p.data(), n);
  double H_init = -log_prob_init + kinetic_init;

  LeapfrogResult lf = leapfrog_step(q, p, epsilon, data, layout);
  double kinetic_new = 0.5 * quotr_linalg::norm_squared(lf.p.data(), n);
  double H_new = -lf.log_prob + kinetic_new;

  double delta_H = H_new - H_init;
  int direction = (delta_H > std::log(0.5)) ? -1 : 1;

  while (true) {
    epsilon *= (direction > 0) ? 2.0 : 0.5;
    if (epsilon > 1e3 || epsilon < 1e-6) break;

    lf = leapfrog_step(q, p, epsilon, data, layout);
    kinetic_new = 0.5 * quotr_linalg::norm_squared(lf.p.data(), n);
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    if (direction > 0 && delta_H > std::log(0.5)) break;
    if (direction < 0 && delta_H < std::log(0.5)) break;
  }

  // Clamp to a more conservative range
  return std::max(1e-4, std::min(epsilon, 0.05));
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
  DualAveraging da(epsilon);

  int sample_idx = 0;
  int n_accept = 0;
  int n_divergent = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);

    // Sample momentum
    std::vector<double> p(n_params);
    for (int i = 0; i < n_params; i++) {
      p[i] = normal(rng);
    }

    // Current Hamiltonian (using optimized norm computation)
    double kinetic_current = 0.5 * quotr_linalg::norm_squared(p.data(), n_params);
    double H_current = -log_prob_current + kinetic_current;

    // Leapfrog integration
    std::vector<double> q_prop = q;
    std::vector<double> p_prop = p;
    bool divergent = false;

    for (int l = 0; l < L; l++) {
      LeapfrogResult lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout);
      q_prop = lf.q;
      p_prop = lf.p;
      if (lf.divergent) {
        divergent = true;
        break;
      }
    }

    double log_prob_prop = compute_log_post(q_prop, data, layout);
    double kinetic_prop = 0.5 * quotr_linalg::norm_squared(p_prop.data(), n_params);
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

    // Adaptation
    if (is_warmup) {
      epsilon = da.update(alpha);
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
  // Run C++ version
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, false
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcout << "Chain " << (chain_id + 1) << " complete. "
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

#ifdef _OPENMP
  // Run chains in parallel - C++ containers only, no R objects
  #pragma omp parallel for schedule(static) num_threads(n_chains)
  for (int c = 0; c < n_chains; c++) {
    cpp_results[c] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, c, seed, false  // verbose=false in parallel
    );
  }
#else
  // Sequential fallback
  for (int c = 0; c < n_chains; c++) {
    cpp_results[c] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, c, seed, false
    );
  }
#endif

  // Convert to R objects outside parallel region (single-threaded)
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);

    if (verbose) {
      int n_div = 0;
      for (int i = 0; i < cpp_results[c].n_sample; i++) {
        n_div += cpp_results[c].divergent[i];
      }
      Rcpp::Rcout << "Chain " << (c + 1) << " complete. "
                  << "Divergent: " << n_div << std::endl;
    }
  }

  return results;
}

} // namespace quotr_hmc

// =====================================================================
// R EXPORTS
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    std::string model_type_str,
    std::string spatial_type_str,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double bym2_scale_factor,
    std::string temporal_type_str,
    Rcpp::IntegerVector temporal_time_idx,
    Rcpp::IntegerVector temporal_group_idx,
    int n_times,
    int n_temporal_groups,
    int n_temporal_params,
    bool temporal_cyclic,
    bool temporal_shared,
    double tau_temporal_shape,
    double tau_temporal_rate,
    double sigma_beta,
    double sigma_re_scale,
    double phi_prior_shape,
    double phi_prior_rate,
    double tau_spatial_shape,
    double tau_spatial_rate,
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
  using namespace quotr_hmc;

  // Set up model data
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

  // Random effects
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;

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

  data.spatial_group = std::vector<int>(spatial_group.begin(), spatial_group.end());
  data.n_spatial_units = n_spatial_units;
  data.adj_row_ptr = std::vector<int>(adj_row_ptr.begin(), adj_row_ptr.end());
  data.adj_col_idx = std::vector<int>(adj_col_idx.begin(), adj_col_idx.end());
  data.n_neighbors = std::vector<int>(n_neighbors.begin(), n_neighbors.end());
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

  data.temporal_time_idx = std::vector<int>(temporal_time_idx.begin(), temporal_time_idx.end());
  data.temporal_group_idx = std::vector<int>(temporal_group_idx.begin(), temporal_group_idx.end());
  data.n_times = n_times;
  data.n_temporal_groups = n_temporal_groups;
  data.n_temporal_params = n_temporal_params;
  data.temporal_cyclic = temporal_cyclic;
  data.temporal_shared = temporal_shared;
  data.tau_temporal_shape = tau_temporal_shape;
  data.tau_temporal_rate = tau_temporal_rate;

  // Zero-inflation structure
  data.zi_type = quotr_zi::parse_zi_type(zi_type_str);
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

  // Initialize parameters
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
int cpp_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}
