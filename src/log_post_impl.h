// log_post_impl.h
// Templated compute_log_post that works with both double and ad::Var
// Single implementation for both evaluation and autodiff gradients
//
// NOTE: This header must be included AFTER hmc_sampler.h which defines
// ModelData, ParamLayout, ModelType, TemporalType

#ifndef RATIOD_LOG_POST_IMPL_H
#define RATIOD_LOG_POST_IMPL_H

#include <vector>
#include "autodiff_utils.h"
#include "hmc_svc_autodiff.h"  // Templated SVC functions
#include "hmc_tvc_autodiff.h"  // Templated TVC functions
#include "hmc_latent_autodiff.h"  // Templated latent factor functions

// Expects these to be defined by including hmc_sampler.h first:
// - ratiod_hmc::ModelData
// - ratiod_hmc::ParamLayout
// - ratiod_hmc::ModelType
// - ratiod_hmc::TemporalType

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using ratiod_hmc::ModelType;
using ratiod_hmc::TemporalType;
using ratiod_zi::ZIType;

namespace ratiod {

using namespace math;

// ============================================================================
// Templated log-posterior computation
// T = double for evaluation, T = ad::Var for autodiff gradients
// ============================================================================

template<typename T>
T compute_log_post_impl(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
    T log_post = T(0.0);

    // ========================================================================
    // Extract parameters
    // ========================================================================

    // Fixed effects
    std::vector<T> beta_num(data.p_num);
    std::vector<T> beta_denom(data.p_denom);
    for (int j = 0; j < data.p_num; j++) {
        beta_num[j] = params[layout.beta_num_start + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        beta_denom[j] = params[layout.beta_denom_start + j];
    }

    // Random effects
    // For non-centered parameterization (re_parameterization == 1):
    //   - params store z ~ N(0,1)
    //   - actual RE = sigma_re * z
    // For centered parameterization (re_parameterization == 0):
    //   - params store re ~ N(0, sigma_re^2) directly
    T sigma_re = T(1.0);
    std::vector<T> re;
    if (layout.has_re) {
        T log_sigma_re = params[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        int n_re = layout.re_end - layout.re_start;
        re.resize(n_re);

        if (data.re_parameterization == 1) {
            // Non-centered: params are z values, compute re = sigma * z
            for (int g = 0; g < n_re; g++) {
                T z_g = params[layout.re_start + g];
                re[g] = sigma_re * z_g;
            }
        } else {
            // Centered: params are actual RE values
            for (int g = 0; g < n_re; g++) {
                re[g] = params[layout.re_start + g];
            }
        }
    }

    // Overdispersion (phi for Gamma denominator in poisson_gamma)
    T phi_num = T(1.0);
    if (layout.has_phi_num) {
        T log_phi = params[layout.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
    }

    T phi_denom = T(1.0);
    if (layout.has_phi_denom) {
        T log_phi = params[layout.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
    }

    // Spatial effects
    std::vector<T> phi_spatial;
    T tau_spatial = T(1.0);
    T sigma_s_bym2 = T(1.0);
    T sigma_u_bym2 = T(1.0);
    std::vector<T> theta_bym2;

    if (layout.has_spatial) {
        int n_spatial = data.n_spatial_units;
        phi_spatial.resize(n_spatial);
        for (int s = 0; s < n_spatial; s++) {
            phi_spatial[s] = params[layout.spatial_start + s];
        }

        if (layout.is_bym2) {
            // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
            T sigma_total_bym2 = safe_exp(params[layout.log_sigma_bym2_idx]);
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2 = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            sigma_s_bym2 = sigma_total_bym2 * sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total_bym2 * sqrt(T(1.0) - rho_bym2);

            theta_bym2.resize(n_spatial);
            for (int s = 0; s < n_spatial; s++) {
                theta_bym2[s] = params[layout.theta_bym2_start + s];
            }
        } else {
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial = safe_exp(log_tau);
        }
    }

    // Temporal effects
    std::vector<T> phi_temporal;
    T tau_temporal = T(1.0);
    T rho_ar1 = T(0.5);
    T sigma2_temporal_gp = T(1.0);
    T phi_temporal_gp = T(1.0);  // lengthscale

    if (layout.has_temporal) {
        // Extract temporal effects (common to all temporal types)
        int n_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            phi_temporal[t] = params[layout.temporal_start + t];
        }

        if (layout.is_temporal_gp) {
            // Temporal GP: sigma2 and phi (lengthscale) parameters
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            T log_phi = params[layout.log_phi_temporal_gp_idx];
            sigma2_temporal_gp = safe_exp(log_sigma2);
            phi_temporal_gp = safe_exp(log_phi);
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            tau_temporal = safe_exp(log_tau);

            if (layout.is_ar1) {
                T logit_rho = params[layout.logit_rho_ar1_idx];
                rho_ar1 = inv_logit(logit_rho);
            }
        }
    }

    // ========================================================================
    // PRIORS
    // ========================================================================

    // Fixed effects: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        log_post = log_post + log_prior_normal(beta_num[j], tau_beta);
    }
    for (int j = 0; j < data.p_denom; j++) {
        log_post = log_post + log_prior_normal(beta_denom[j], tau_beta);
    }

    // Random effects prior with Half-Cauchy on sigma_re
    // For non-centered: z ~ N(0,1), so prior is on params (z values) with tau=1
    // For centered: re ~ N(0, sigma_re^2), so prior uses tau = 1/sigma_re^2
    if (layout.has_re) {
        // Half-Cauchy prior on sigma_re (same for both parameterizations)
        T log_sigma_re = params[layout.log_sigma_re_idx];
        log_post = log_post + log_prior_half_cauchy(log_sigma_re, data.sigma_re_scale);

        // Prior on RE/z values
        int n_re = layout.re_end - layout.re_start;
        if (data.re_parameterization == 1) {
            // Non-centered: z ~ N(0, 1), so tau = 1
            // Prior on z values (the stored params)
            for (int g = 0; g < n_re; g++) {
                T z_g = params[layout.re_start + g];
                log_post = log_post + log_prior_normal(z_g, 1.0);
                // Normalization constant for tau=1: 0.5 * log(1) = 0
            }
        } else {
            // Centered: re ~ N(0, sigma_re^2)
            T tau_re = T(1.0) / (sigma_re * sigma_re + T(1e-10));
            for (int g = 0; g < n_re; g++) {
                log_post = log_post + log_prior_normal(re[g], get_value(tau_re));
                // Add normalization constant
                log_post = log_post + T(0.5) * safe_log(tau_re);
            }
        }
    }

    // Overdispersion: Gamma prior
    if (layout.has_phi_num) {
        T log_phi = params[layout.log_phi_num_idx];
        log_post = log_post + log_prior_gamma(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }
    if (layout.has_phi_denom) {
        T log_phi = params[layout.log_phi_denom_idx];
        log_post = log_post + log_prior_gamma(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }

    // Spatial priors
    if (layout.has_spatial) {
        if (layout.is_bym2) {
            // BYM2 Riebler: Half-Cauchy on sigma_total
            T log_sigma = params[layout.log_sigma_bym2_idx];
            log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);

            // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
            // log p(logit_rho) = log(rho) + log(1-rho)
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2_prior = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            log_post = log_post + log(rho_bym2_prior)
                                + log(T(1.0) - rho_bym2_prior);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial[i] * phi_spatial[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial[i] * phi_spatial[j];
                    }
                }
            }
            log_post = log_post - T(0.5) * quad_form;

            // N(0, I) prior on theta
            for (int s = 0; s < data.n_spatial_units; s++) {
                log_post = log_post - T(0.5) * theta_bym2[s] * theta_bym2[s];
            }
        } else {
            // ICAR: Gamma prior on tau
            T log_tau = params[layout.log_tau_spatial_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_spatial_shape, data.tau_spatial_rate);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial[i] * phi_spatial[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial[i] * phi_spatial[j];
                    }
                }
            }
            int J = data.n_spatial_units;
            T log_tau_sp = params[layout.log_tau_spatial_idx];
            log_post = log_post + T(0.5 * (J - 1)) * log_tau_sp - T(0.5) * tau_spatial * quad_form;
        }
    }

    // Temporal priors
    if (layout.has_temporal) {
        int T_times = data.n_times;

        if (layout.is_temporal_gp) {
            // Temporal GP: PC prior on sigma2, uniform on phi (lengthscale)
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            T log_phi = params[layout.log_phi_temporal_gp_idx];

            // PC prior on sigma2 (favor smaller variance)
            // log p(sigma2) = log(rate) - rate * sqrt(sigma2) - log(2 * sqrt(sigma2))
            double rate = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
            T sigma_gp = safe_sqrt(sigma2_temporal_gp);
            log_post = log_post + T(std::log(rate)) - T(rate) * sigma_gp - safe_log(T(2.0) * sigma_gp);
            log_post = log_post + log_sigma2;  // Jacobian for log transform

            // Uniform prior on phi within bounds - just check bounds
            double phi_val = get_value(phi_temporal_gp);
            if (phi_val < data.temporal_gp_phi_prior_lower || phi_val > data.temporal_gp_phi_prior_upper) {
                return T(-INFINITY);
            }
            log_post = log_post - T(std::log(data.temporal_gp_phi_prior_upper - data.temporal_gp_phi_prior_lower));
            log_post = log_post + log_phi;  // Jacobian

            // GP log-likelihood for temporal effects using exponential covariance
            // For efficiency, use state-space representation (AR1 approximation)
            for (int g = 0; g < data.n_temporal_groups; g++) {
                // First time point: marginal N(0, sigma2)
                T f0 = phi_temporal[g * T_times];
                log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2_temporal_gp);
                log_post = log_post - T(0.5) * f0 * f0 / sigma2_temporal_gp;

                // Subsequent time points: conditional on previous
                for (int t = 1; t < T_times; t++) {
                    T f_prev = phi_temporal[g * T_times + t - 1];
                    T f_curr = phi_temporal[g * T_times + t];

                    // Time difference (assumes consecutive integer times scaled)
                    double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                    T rho = safe_exp(T(-dt) / phi_temporal_gp);
                    T cond_var = sigma2_temporal_gp * (T(1.0) - rho * rho);

                    // Ensure positive variance
                    T cond_var_safe = safe_max(cond_var, T(1e-10));
                    T cond_mean = rho * f_prev;
                    T resid = f_curr - cond_mean;

                    log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var_safe);
                    log_post = log_post - T(0.5) * resid * resid / cond_var_safe;
                }
            }
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_temporal_shape, data.tau_temporal_rate);

            if (data.temporal_type == TemporalType::RW1) {
                // RW1: sum of (phi[t] - phi[t-1])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 1; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t] - phi_temporal[g * T_times + t - 1];
                        quad_form = quad_form + diff * diff;
                    }
                }
                log_post = log_post + T(0.5 * (T_times - 1) * data.n_temporal_groups) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal * quad_form;

            } else if (data.temporal_type == TemporalType::RW2) {
                // RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 2; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t]
                               - T(2.0) * phi_temporal[g * T_times + t - 1]
                               + phi_temporal[g * T_times + t - 2];
                        quad_form = quad_form + diff * diff;
                    }
                }
                log_post = log_post + T(0.5 * (T_times - 2) * data.n_temporal_groups) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal * quad_form;

            } else if (data.temporal_type == TemporalType::AR1) {
                // AR1: phi[t] | phi[t-1] ~ N(rho * phi[t-1], 1/tau)
                // Jacobian for logit -> rho transformation: log(rho+1) + log(1-rho)
                log_post = log_post + safe_log(rho_ar1 + T(1.0)) + safe_log(T(1.0) - rho_ar1);

                for (int g = 0; g < data.n_temporal_groups; g++) {
                    // First time point: phi[0] ~ N(0, 1/(tau*(1-rho^2)))
                    T var_stationary = T(1.0) / (tau_temporal * (T(1.0) - rho_ar1 * rho_ar1));
                    log_post = log_post - T(0.5) * phi_temporal[g * T_times] * phi_temporal[g * T_times] / var_stationary;

                    // Subsequent: phi[t] | phi[t-1]
                    for (int t = 1; t < T_times; t++) {
                        T resid = phi_temporal[g * T_times + t] - rho_ar1 * phi_temporal[g * T_times + t - 1];
                        log_post = log_post - T(0.5) * tau_temporal * resid * resid;
                    }
                }
            }
        }
    }

    // SVC (Spatially-Varying Coefficients) parameters and priors
    std::vector<T> svc_sigma2;
    std::vector<T> svc_phi;
    std::vector<T> svc_w_flat;
    std::vector<T> svc_eta;

    if (layout.has_svc && data.svc_data.n_svc > 0) {
        int n_svc = data.svc_data.n_svc;
        int n_obs = data.svc_data.n_obs;

        // Extract sigma2 (spatial variance) parameters
        svc_sigma2.resize(n_svc);
        for (int j = 0; j < n_svc; j++) {
            T log_sigma2 = params[layout.log_sigma2_svc_start + j];
            svc_sigma2[j] = safe_exp(log_sigma2);

            // Half-Cauchy prior on sigma = sqrt(sigma2)
            T sigma = safe_sqrt(svc_sigma2[j]);
            double scale = data.svc_sigma2_prior_scale;
            log_post = log_post - safe_log(T(1.0) + sigma * sigma / T(scale * scale));
            log_post = log_post + log_sigma2;  // Jacobian for log transform
        }

        // Extract phi (spatial range) parameters
        svc_phi.resize(n_svc);
        for (int j = 0; j < n_svc; j++) {
            T log_phi = params[layout.log_phi_svc_start + j];
            svc_phi[j] = safe_exp(log_phi);

            // Uniform prior on phi (transformed to log scale)
            double phi_val = get_value(svc_phi[j]);
            if (phi_val < data.svc_phi_prior_lower || phi_val > data.svc_phi_prior_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi;  // Jacobian for log transform
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
            std::vector<T> w_j(n_obs);
            for (int k = 0; k < n_obs; k++) {
                w_j[k] = svc_w_flat[j * n_obs + k];
            }

            // NNGP log-likelihood
            log_post = log_post + ratiod_svc_ad::nngp_log_lik(w_j, svc_sigma2[j], svc_phi[j], data.svc_data);
        }

        // Soft sum-to-zero constraint for identifiability
        log_post = log_post + ratiod_svc_ad::svc_sum_to_zero_penalty(svc_w_flat, data.svc_data, 1.0);

        // Precompute SVC contribution to linear predictor
        svc_eta.resize(n_obs, T(0.0));
        ratiod_svc_ad::compute_svc_eta(svc_w_flat, data.svc_data, svc_eta);
    }

    // TVC (Temporally-Varying Coefficients) parameters and priors
    std::vector<T> tvc_tau;
    std::vector<T> tvc_rho;
    std::vector<T> tvc_w_flat;
    std::vector<T> tvc_eta;

    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        int n_tvc = data.tvc_data.n_tvc;
        int n_groups = data.tvc_data.n_groups;
        int n_times = data.tvc_data.n_times;
        int n_obs = data.tvc_data.n_obs;

        // Extract tau (precision) parameters
        tvc_tau.resize(n_tvc);
        for (int j = 0; j < n_tvc; j++) {
            T log_tau = params[layout.log_tau_tvc_start + j];
            tvc_tau[j] = safe_exp(log_tau);

            // Gamma prior on tau + Jacobian for log transform
            log_post = log_post + T(data.tvc_tau_shape - 1.0) * log_tau
                     - T(data.tvc_tau_rate) * tvc_tau[j]
                     + log_tau;  // Jacobian
        }

        // Extract rho (AR1 correlation) parameters if AR1 structure
        tvc_rho.resize(n_tvc, T(0.0));
        if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
            for (int j = 0; j < n_tvc; j++) {
                T logit_rho = params[layout.logit_rho_tvc_start + j];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho);
                tvc_rho[j] = T(2.0) * u - T(1.0);

                // Beta(2, 2) prior on u (favors moderate autocorrelation)
                // + Jacobian for logit transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Beta(2,2)
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Jacobian
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        tvc_w_flat.resize(n_tvc_params);
        for (int k = 0; k < n_tvc_params; k++) {
            tvc_w_flat[k] = params[layout.tvc_w_start + k];
        }

        // TVC temporal prior (RW1, RW2, or AR1)
        log_post = log_post + ratiod_tvc_ad::tvc_log_prior(
            tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho
        );

        // Soft sum-to-zero constraint for identifiability
        log_post = log_post + ratiod_tvc_ad::tvc_sum_to_zero_penalty(
            tvc_w_flat, data.tvc_data, 0.001
        );

        // Precompute TVC contribution to linear predictor
        tvc_eta.resize(n_obs, T(0.0));
        ratiod_tvc_ad::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
    }

    // Latent factors parameters and priors
    std::vector<T> latent_sigma;
    std::vector<T> latent_factors;
    std::vector<T> latent_eta;

    if (layout.has_latent && data.latent_n_factors > 0) {
        int K = data.latent_n_factors;
        int N = data.N;

        // Extract log_sigma parameters
        std::vector<T> log_sigma_latent(K);
        for (int k = 0; k < K; k++) {
            log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        }

        // Compute sigma from log_sigma
        latent_sigma.resize(K);
        ratiod_latent_ad::extract_sigma(latent_sigma, log_sigma_latent);

        // Extract factors and apply constraint
        int n_factor_params = N * K;
        latent_factors.resize(n_factor_params);
        for (int j = 0; j < n_factor_params; j++) {
            latent_factors[j] = params[layout.latent_factor_start + j];
        }

        // Apply identifiability constraint
        if (data.latent_constraint == 0) {  // SUM_TO_ZERO
            ratiod_latent_ad::apply_sum_to_zero(latent_factors, N, K);
        } else {  // FIRST_ZERO
            ratiod_latent_ad::apply_first_zero(latent_factors, N, K);
        }

        // Sigma prior: Exponential on sigma (PC prior)
        log_post = log_post + ratiod_latent_ad::latent_sigma_log_prior(
            log_sigma_latent, data.latent_sigma_prior_rate
        );

        // Factor prior: N(0, 1) on each factor score
        ratiod_latent::LatentConstraint constraint =
            (data.latent_constraint == 0) ? ratiod_latent::LatentConstraint::SUM_TO_ZERO
                                          : ratiod_latent::LatentConstraint::FIRST_ZERO;
        log_post = log_post + ratiod_latent_ad::latent_factor_log_prior(
            latent_factors, N, K, constraint
        );

        // Precompute latent factor contribution to linear predictor
        latent_eta.resize(N, T(0.0));
        ratiod_latent_ad::latent_contributions_all(latent_eta, latent_factors, latent_sigma, N, K);
    }

    // Zero-inflation / One-inflation parameters
    std::vector<T> beta_zi;
    std::vector<T> beta_oi;

    if (layout.has_zi && data.p_zi > 0) {
        beta_zi.resize(data.p_zi);
        for (int j = 0; j < data.p_zi; j++) {
            beta_zi[j] = params[layout.beta_zi_start + j];
        }
        // Prior on beta_zi: N(0, zi_prior_sd^2)
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd);
        for (int j = 0; j < data.p_zi; j++) {
            log_post = log_post + log_prior_normal(beta_zi[j], tau_zi);
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        beta_oi.resize(data.p_oi);
        for (int j = 0; j < data.p_oi; j++) {
            beta_oi[j] = params[layout.beta_oi_start + j];
        }
        // Prior on beta_oi: N(0, oi_prior_sd^2)
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd);
        for (int j = 0; j < data.p_oi; j++) {
            log_post = log_post + log_prior_normal(beta_oi[j], tau_oi);
        }
    }

    // ========================================================================
    // LIKELIHOOD
    // ========================================================================

    // Note: No OpenMP here - autodiff tape is not thread-safe
    // For double, this is slightly slower but correct

    for (int i = 0; i < data.N; i++) {
        // Compute linear predictors
        T eta_num = T(0.0);
        T eta_denom = T(0.0);

        for (int j = 0; j < data.p_num; j++) {
            eta_num = eta_num + data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom = eta_denom + data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Add random effects (shared between num and denom)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            eta_num = eta_num + re[g];
            eta_denom = eta_denom + re[g];
        }

        // Add spatial effects
        if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
            int s = data.spatial_group[i] - 1;
            T spatial_effect;

            if (layout.is_bym2) {
                T scaled_phi = phi_spatial[s] * T(data.bym2_scale_factor);
                spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s];
            } else {
                spatial_effect = phi_spatial[s];
            }

            eta_num = eta_num + spatial_effect;
            eta_denom = eta_denom + spatial_effect;
        }

        // Add temporal effects
        if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            T temporal_effect = phi_temporal[g * data.n_times + t];

            if (data.temporal_shared) {
                eta_num = eta_num + temporal_effect;
                eta_denom = eta_denom + temporal_effect;
            } else {
                eta_num = eta_num + temporal_effect;
            }
        }

        // Add SVC (Spatially-Varying Coefficients) effect
        if (layout.has_svc && !svc_eta.empty()) {
            T svc_effect = svc_eta[i];
            if (data.svc_data.shared) {
                eta_num = eta_num + svc_effect;
                eta_denom = eta_denom + svc_effect;
            } else {
                eta_num = eta_num + svc_effect;
            }
        }

        // Add TVC (Temporally-Varying Coefficients) effect
        if (layout.has_tvc && !tvc_eta.empty()) {
            T tvc_effect = tvc_eta[i];
            if (data.tvc_data.shared) {
                eta_num = eta_num + tvc_effect;
                eta_denom = eta_denom + tvc_effect;
            } else {
                eta_num = eta_num + tvc_effect;
            }
        }

        // Add latent factor effect
        if (layout.has_latent && !latent_eta.empty()) {
            T latent_effect = latent_eta[i];
            if (data.latent_shared) {
                eta_num = eta_num + latent_effect;
                eta_denom = eta_denom + latent_effect;
            } else {
                eta_num = eta_num + latent_effect;
            }
        }

        // Compute ZI/OI linear predictors if needed
        T logit_zi = T(0.0);
        T logit_oi = T(0.0);

        if (layout.has_zi && data.p_zi > 0) {
            for (int j = 0; j < data.p_zi; j++) {
                logit_zi = logit_zi + data.X_zi_flat[i * data.p_zi + j] * beta_zi[j];
            }
        }

        if (layout.has_oi && data.p_oi > 0) {
            for (int j = 0; j < data.p_oi; j++) {
                logit_oi = logit_oi + data.X_oi_flat[i * data.p_oi + j] * beta_oi[j];
            }
        }

        // Compute likelihood based on model type
        T ll_i = T(0.0);

        if (data.model_type == ModelType::BINOMIAL) {
            T p = inv_logit(eta_num);
            int y = data.y_num[i];
            int n = data.y_denom[i];

            // Handle different ZI types for binomial
            if (data.zi_type == ZIType::ZI_BINOMIAL) {
                ll_i = log_lik_zi_binomial(y, n, p, logit_zi);
            } else if (data.zi_type == ZIType::OI_BINOMIAL) {
                ll_i = log_lik_oi_binomial(y, n, p, logit_oi);
            } else if (data.zi_type == ZIType::ZOIB) {
                ll_i = log_lik_zoib(y, n, p, logit_zi, logit_oi);
            } else if (data.zi_type == ZIType::HURDLE_BINOMIAL) {
                ll_i = log_lik_hurdle_binomial(y, n, p, logit_zi);
            } else {
                ll_i = log_lik_binomial(y, n, p);
            }

        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);
            ll_i = log_lik_negbin(data.y_num[i], mu_num, phi_num);
            ll_i = ll_i + log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);

        } else if (data.model_type == ModelType::POISSON_GAMMA) {
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);
            ll_i = log_lik_poisson(data.y_num[i], mu_num);
            ll_i = ll_i + log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);

        } else if (data.model_type == ModelType::GAMMA_GAMMA) {
            // Gamma-Gamma: both responses are continuous Gamma
            // phi_num = shape_num, phi_denom = shape_denom
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);
            ll_i = log_lik_gamma_gamma(data.y_num_cont[i], data.y_denom_cont[i],
                                       mu_num, mu_denom, phi_num, phi_denom);

        } else if (data.model_type == ModelType::LOGNORMAL) {
            // Lognormal-Lognormal: both responses are continuous Lognormal
            // eta = mean on log scale, phi = sigma (std dev on log scale)
            ll_i = log_lik_lognormal_lognormal(data.y_num_cont[i], data.y_denom_cont[i],
                                               eta_num, eta_denom, phi_num, phi_denom);

        } else if (data.model_type == ModelType::BETA_BINOMIAL) {
            // Beta-Binomial: overdispersed binomial
            // phi_num = precision parameter (alpha + beta)
            T p = inv_logit(eta_num);
            int y = data.y_num[i];
            int n = data.y_denom[i];
            ll_i = log_lik_beta_binomial(y, n, p, phi_num);
        }

        log_post = log_post + ll_i;
    }

    return log_post;
}

}  // namespace ratiod

#endif  // RATIOD_LOG_POST_IMPL_H
