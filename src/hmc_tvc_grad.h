// hmc_tvc_grad.h
// Hand-coded gradients for TVC (Temporally-Varying Coefficients)
// Provides O(n) analytical gradients for RW1, RW2, and AR1 temporal structures

#ifndef RATIOD_HMC_TVC_GRAD_H
#define RATIOD_HMC_TVC_GRAD_H

#include <vector>
#include <cmath>
#include "hmc_tvc.h"

namespace ratiod_tvc {

// Structure to hold TVC gradient results
struct TVCGradients {
    std::vector<double> grad_w;       // Gradient w.r.t. TVC values
    std::vector<double> grad_log_tau; // Gradient w.r.t. log(tau)
    std::vector<double> grad_logit_rho; // Gradient w.r.t. logit(rho) (AR1 only)
};

// =============================================================================
// Analytical RW1 gradients
// =============================================================================

// RW1 prior: log p(w|tau) = 0.5 * (T-1) * log(tau) - 0.5 * tau * sum((w[t] - w[t-1])^2)
// d/d(w[t]) = -tau * (2*w[t] - w[t-1] - w[t+1]) for interior
//           = -tau * (w[t] - w[t+1]) for t=0
//           = -tau * (w[t] - w[t-1]) for t=T-1
inline void rw1_grad_w(const double* w, int n_times, double tau, double* grad_w) {
    for (int t = 0; t < n_times; t++) {
        if (t == 0) {
            grad_w[t] = -tau * (w[0] - w[1]);
        } else if (t == n_times - 1) {
            grad_w[t] = -tau * (w[t] - w[t-1]);
        } else {
            grad_w[t] = -tau * (2.0 * w[t] - w[t-1] - w[t+1]);
        }
    }
}

// d/d(log_tau) = 0.5 * (T-1) - 0.5 * tau * quad + log_tau_jacobian
//              = 0.5 * (T-1) - 0.5 * quad * tau  (Jacobian cancels in chain rule)
inline double rw1_grad_log_tau(const double* w, int n_times, double tau) {
    double quad = 0.0;
    for (int t = 1; t < n_times; t++) {
        double diff = w[t] - w[t-1];
        quad += diff * diff;
    }
    // d/d(log_tau) = 0.5*(T-1) - 0.5*tau*quad + tau (Jacobian: d tau / d log_tau = tau)
    // But we want d log_post / d log_tau, which includes Jacobian automatically in computation
    return 0.5 * (n_times - 1) - 0.5 * tau * quad;
}

// =============================================================================
// Analytical RW2 gradients
// =============================================================================

// RW2 prior: second differences d[t] = w[t] - 2*w[t-1] + w[t-2]
// log p(w|tau) = 0.5 * (T-2) * log(tau) - 0.5 * tau * sum(d[t]^2)

// The gradient is more complex. For w[t], it contributes to d[t], d[t+1], d[t+2].
// d[t] = w[t] - 2*w[t-1] + w[t-2]
// d/d(w[k]) of sum(d[t]^2) = 2 * sum over t where w[k] appears
// At t: coefficient for w[t] is 1, for w[t-1] is -2, for w[t-2] is 1
// So d/d(w[k]) = 2 * (d[k+2] * 1 - 2 * d[k+1] + d[k] * 1) if k >= 2
inline void rw2_grad_w(const double* w, int n_times, double tau, double* grad_w) {
    // Compute second differences
    std::vector<double> d(n_times);
    for (int t = 2; t < n_times; t++) {
        d[t] = w[t] - 2.0 * w[t-1] + w[t-2];
    }

    // Gradient: d/d(w[k]) [-0.5 * tau * sum(d[t]^2)]
    // = -tau * sum_t d[t] * (d d[t] / d w[k])
    for (int k = 0; k < n_times; k++) {
        grad_w[k] = 0.0;
        // d[t] depends on w[t], w[t-1], w[t-2]
        // So w[k] affects d[k] (coef=1), d[k+1] (coef=-2), d[k+2] (coef=1)
        if (k >= 2 && k < n_times) {
            grad_w[k] += -tau * d[k] * 1.0;
        }
        if (k >= 1 && k+1 < n_times) {
            grad_w[k] += -tau * d[k+1] * (-2.0);
        }
        if (k+2 < n_times) {
            grad_w[k] += -tau * d[k+2] * 1.0;
        }
    }
}

inline double rw2_grad_log_tau(const double* w, int n_times, double tau) {
    double quad = 0.0;
    for (int t = 2; t < n_times; t++) {
        double d = w[t] - 2.0 * w[t-1] + w[t-2];
        quad += d * d;
    }
    return 0.5 * (n_times - 2) - 0.5 * tau * quad;
}

// =============================================================================
// Analytical AR1 gradients
// =============================================================================

// AR1 prior:
// w[0] ~ N(0, 1/(tau*(1-rho^2)))
// w[t] | w[t-1] ~ N(rho * w[t-1], 1/tau) for t > 0

// Gradient w.r.t. w[t]:
// t=0: d/d(w[0]) = -tau * (1-rho^2) * w[0] - tau * (w[1] - rho*w[0]) * (-rho)
// t>0, t<T-1: d/d(w[t]) = -tau * (w[t] - rho*w[t-1]) + tau * rho * (w[t+1] - rho*w[t])
// t=T-1: d/d(w[T-1]) = -tau * (w[T-1] - rho*w[T-2])
inline void ar1_grad_w(const double* w, int n_times, double tau, double rho, double* grad_w) {
    double one_m_rho2 = 1.0 - rho * rho;

    if (n_times == 1) {
        grad_w[0] = -tau * one_m_rho2 * w[0];
        return;
    }

    // First time point
    double resid_1 = w[1] - rho * w[0];
    grad_w[0] = -tau * one_m_rho2 * w[0] + tau * rho * resid_1;

    // Interior time points
    for (int t = 1; t < n_times - 1; t++) {
        double resid_t = w[t] - rho * w[t-1];
        double resid_tp1 = w[t+1] - rho * w[t];
        grad_w[t] = -tau * resid_t + tau * rho * resid_tp1;
    }

    // Last time point
    double resid_T = w[n_times-1] - rho * w[n_times-2];
    grad_w[n_times-1] = -tau * resid_T;
}

// Gradient w.r.t. log(tau)
inline double ar1_grad_log_tau(const double* w, int n_times, double tau, double rho) {
    double one_m_rho2 = 1.0 - rho * rho;

    // Stationary part
    double grad = -0.5;  // From -0.5*log(2*pi*var_stationary)
    grad += 0.5 * (n_times - 1);  // From normalization constants for t>0

    // Quadratic terms
    double quad = one_m_rho2 * w[0] * w[0];
    for (int t = 1; t < n_times; t++) {
        double resid = w[t] - rho * w[t-1];
        quad += resid * resid;
    }
    grad -= 0.5 * tau * quad;

    return grad;
}

// Gradient w.r.t. logit(rho) where u = inv_logit(logit_rho), rho = 2*u - 1
// d/d(logit_rho) = d/d(rho) * d(rho)/d(u) * d(u)/d(logit_rho)
//                = d/d(rho) * 2 * u * (1-u)
inline double ar1_grad_logit_rho(const double* w, int n_times, double tau, double rho) {
    double one_m_rho2 = 1.0 - rho * rho;

    // d log p / d rho from stationary variance
    double grad_rho = rho / one_m_rho2 * w[0] * w[0];

    // d log p / d rho from AR terms
    for (int t = 1; t < n_times; t++) {
        double resid = w[t] - rho * w[t-1];
        grad_rho += tau * resid * w[t-1];
    }

    // Transform to logit_rho
    // u = (rho + 1) / 2, logit_rho = logit(u)
    // d(rho)/d(logit_rho) = d(rho)/d(u) * d(u)/d(logit_rho)
    //                     = 2 * u * (1-u)
    double u = (rho + 1.0) / 2.0;
    double d_rho_d_logit = 2.0 * u * (1.0 - u);

    return grad_rho * d_rho_d_logit;
}

// =============================================================================
// Full TVC prior gradients
// =============================================================================

// Compute gradients for all TVC parameters from the temporal prior
inline void tvc_prior_gradients(
    const std::vector<double>& w_flat,  // n_groups * n_tvc * n_times
    const TVCData& tvc_data,
    const std::vector<double>& tau,     // Length n_tvc
    const std::vector<double>& rho,     // Length n_tvc (AR1 only)
    TVCGradients& grads                 // Output
) {
    int n_groups = tvc_data.n_groups;
    int n_tvc = tvc_data.n_tvc;
    int n_times = tvc_data.n_times;

    int n_w = n_groups * n_tvc * n_times;
    grads.grad_w.assign(n_w, 0.0);
    grads.grad_log_tau.assign(n_tvc, 0.0);
    grads.grad_logit_rho.assign(n_tvc, 0.0);

    for (int g = 0; g < n_groups; g++) {
        for (int j = 0; j < n_tvc; j++) {
            // Extract w for this group/term
            int offset = (g * n_tvc + j) * n_times;
            const double* w_jg = &w_flat[offset];

            std::vector<double> grad_w_jg(n_times);
            double grad_log_tau_j = 0.0;
            double grad_logit_rho_j = 0.0;

            if (tvc_data.structure == TemporalType::RW1) {
                rw1_grad_w(w_jg, n_times, tau[j], grad_w_jg.data());
                grad_log_tau_j = rw1_grad_log_tau(w_jg, n_times, tau[j]);

            } else if (tvc_data.structure == TemporalType::RW2) {
                rw2_grad_w(w_jg, n_times, tau[j], grad_w_jg.data());
                grad_log_tau_j = rw2_grad_log_tau(w_jg, n_times, tau[j]);

            } else if (tvc_data.structure == TemporalType::AR1) {
                ar1_grad_w(w_jg, n_times, tau[j], rho[j], grad_w_jg.data());
                grad_log_tau_j = ar1_grad_log_tau(w_jg, n_times, tau[j], rho[j]);
                grad_logit_rho_j = ar1_grad_logit_rho(w_jg, n_times, tau[j], rho[j]);

            } else {
                // IID: N(0, 1/tau)
                for (int t = 0; t < n_times; t++) {
                    grad_w_jg[t] = -tau[j] * w_jg[t];
                }
                double quad = 0.0;
                for (int t = 0; t < n_times; t++) {
                    quad += w_jg[t] * w_jg[t];
                }
                grad_log_tau_j = 0.5 * n_times - 0.5 * tau[j] * quad;
            }

            // Accumulate gradients
            for (int t = 0; t < n_times; t++) {
                grads.grad_w[offset + t] += grad_w_jg[t];
            }
            grads.grad_log_tau[j] += grad_log_tau_j;
            grads.grad_logit_rho[j] += grad_logit_rho_j;
        }
    }

    // Add soft sum-to-zero penalty gradients
    double lambda = 0.001;
    for (int g = 0; g < n_groups; g++) {
        for (int j = 0; j < n_tvc; j++) {
            int offset = (g * n_tvc + j) * n_times;
            double sum = 0.0;
            for (int t = 0; t < n_times; t++) {
                sum += w_flat[offset + t];
            }
            for (int t = 0; t < n_times; t++) {
                grads.grad_w[offset + t] -= lambda * sum;
            }
        }
    }
}

} // namespace ratiod_tvc

#endif // RATIOD_HMC_TVC_GRAD_H
