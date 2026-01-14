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

// Expects these to be defined by including hmc_sampler.h first:
// - ratiod_hmc::ModelData
// - ratiod_hmc::ParamLayout
// - ratiod_hmc::ModelType
// - ratiod_hmc::TemporalType

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using ratiod_hmc::ModelType;
using ratiod_hmc::TemporalType;

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
    T sigma_re = T(1.0);
    std::vector<T> re;
    if (layout.has_re) {
        T log_sigma_re = params[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        int n_re = layout.re_end - layout.re_start;
        re.resize(n_re);
        for (int g = 0; g < n_re; g++) {
            re[g] = params[layout.re_start + g];
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
    T sigma_bym2 = T(1.0);
    T rho_bym2 = T(0.5);
    std::vector<T> theta_bym2;

    if (layout.has_spatial) {
        int n_spatial = data.n_spatial_units;
        phi_spatial.resize(n_spatial);
        for (int s = 0; s < n_spatial; s++) {
            phi_spatial[s] = params[layout.spatial_start + s];
        }

        if (layout.is_bym2) {
            sigma_bym2 = safe_exp(params[layout.log_sigma_bym2_idx]);
            T logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = inv_logit(logit_rho);

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

    if (layout.has_temporal) {
        T log_tau = params[layout.log_tau_temporal_idx];
        tau_temporal = safe_exp(log_tau);

        int n_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            phi_temporal[t] = params[layout.temporal_start + t];
        }

        if (layout.is_ar1) {
            T logit_rho = params[layout.logit_rho_ar1_idx];
            rho_ar1 = inv_logit(logit_rho);
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

    // Random effects: N(0, sigma_re^2) with Half-Cauchy on sigma
    if (layout.has_re) {
        // Half-Cauchy prior on sigma_re
        T log_sigma_re = params[layout.log_sigma_re_idx];
        log_post = log_post + log_prior_half_cauchy(log_sigma_re, data.sigma_re_scale);

        // RE prior: N(0, sigma_re^2)
        T tau_re = T(1.0) / (sigma_re * sigma_re + T(1e-10));
        for (size_t g = 0; g < re.size(); g++) {
            log_post = log_post + log_prior_normal(re[g], get_value(tau_re));
            // Add normalization constant
            log_post = log_post + T(0.5) * safe_log(tau_re);
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
            // BYM2: Half-Cauchy on sigma, Beta on rho
            T log_sigma = params[layout.log_sigma_bym2_idx];
            log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);

            // Beta(0.5, 0.5) prior on rho
            T logit_rho = params[layout.logit_rho_bym2_idx];
            log_post = log_post + T(-0.5) * safe_log(rho_bym2) + T(-0.5) * safe_log(T(1.0) - rho_bym2);
            log_post = log_post + safe_log(rho_bym2) + safe_log(T(1.0) - rho_bym2);  // Jacobian

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
        T log_tau = params[layout.log_tau_temporal_idx];
        log_post = log_post + log_prior_gamma(log_tau, data.tau_temporal_shape, data.tau_temporal_rate);

        int T_times = data.n_times;

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
            T logit_rho = params[layout.logit_rho_ar1_idx];
            log_post = log_post + safe_log(rho_ar1 + T(1.0)) + safe_log(T(1.0) - rho_ar1);  // Jacobian

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
                spatial_effect = sigma_bym2 * (
                    safe_sqrt(rho_bym2) * scaled_phi +
                    safe_sqrt(T(1.0) - rho_bym2) * theta_bym2[s]
                );
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

        // Compute likelihood based on model type
        T ll_i = T(0.0);

        if (data.model_type == ModelType::BINOMIAL) {
            T p = inv_logit(eta_num);
            ll_i = log_lik_binomial(data.y_num[i], data.y_denom[i], p);

        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);
            ll_i = log_lik_negbin(data.y_num[i], mu_num, phi_num);
            ll_i = ll_i + log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);

        } else {  // POISSON_GAMMA
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);
            ll_i = log_lik_poisson(data.y_num[i], mu_num);
            ll_i = ll_i + log_lik_gamma(data.y_denom_cont[i], phi_num, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    return log_post;
}

}  // namespace ratiod

#endif  // RATIOD_LOG_POST_IMPL_H
