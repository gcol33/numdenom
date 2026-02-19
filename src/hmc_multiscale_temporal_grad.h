// hmc_multiscale_temporal_grad.h
// Hand-coded gradients for Multi-scale Temporal decomposition
// Provides O(n) analytical gradients for trend + seasonal + short-term components

#ifndef RATIOD_HMC_MULTISCALE_TEMPORAL_GRAD_H
#define RATIOD_HMC_MULTISCALE_TEMPORAL_GRAD_H

#include <vector>
#include <cmath>
#include "hmc_temporal_multiscale.h"

namespace ratiod_temporal_grad {

using ratiod_temporal::MultiscaleTemporalData;
using ratiod_temporal::TemporalType;

// Structure to hold multiscale temporal gradient results
struct MultiscaleTemporalGradients {
    std::vector<double> grad_trend;           // Gradient w.r.t. trend effects
    std::vector<double> grad_seasonal;        // Gradient w.r.t. seasonal effects
    std::vector<double> grad_short_term;      // Gradient w.r.t. short-term effects
    double grad_log_sigma2_trend;             // Gradient w.r.t. log(sigma2_trend)
    double grad_log_sigma2_seasonal;          // Gradient w.r.t. log(sigma2_seasonal)
    double grad_log_sigma2_short;             // Gradient w.r.t. log(sigma2_short)
    double grad_logit_rho_short;              // Gradient w.r.t. logit(rho_short) (AR1 only)
};

// =============================================================================
// RW1 gradients (non-cyclic)
// =============================================================================

// RW1: log p(phi|sigma2) = -0.5 * (T-1) * log(2*pi*sigma2)
//                        - 0.5 / sigma2 * sum((phi[t] - phi[t-1])^2)
inline void rw1_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    for (int t = 0; t < n; t++) {
        if (t == 0) {
            grad_phi[t] = -inv_sigma2 * (phi[0] - phi[1]);
        } else if (t == n - 1) {
            grad_phi[t] = -inv_sigma2 * (phi[t] - phi[t-1]);
        } else {
            grad_phi[t] = -inv_sigma2 * (2.0 * phi[t] - phi[t-1] - phi[t+1]);
        }
    }
}

inline double rw1_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 1; t < n; t++) {
        double diff = phi[t] - phi[t-1];
        quad += diff * diff;
    }
    // d/d(log_sigma2) = -0.5 * (n-1) + 0.5 / sigma2 * quad
    // Jacobian: sigma2 = exp(log_sigma2), so multiply by sigma2
    return (-0.5 * (n - 1) + 0.5 * quad / sigma2) * sigma2;
}

// =============================================================================
// RW1 gradients (cyclic - for seasonal)
// =============================================================================

// Cyclic RW1: adds connection from last to first
inline void rw1_cyclic_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    if (n < 2) {
        for (int t = 0; t < n; t++) grad_phi[t] = 0.0;
        return;
    }

    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    // All interior differences: phi[t] - phi[t-1]
    for (int t = 0; t < n; t++) {
        grad_phi[t] = 0.0;

        // From diff[t+1] = phi[t+1] - phi[t]: coef = 1 for phi[t+1], -1 for phi[t]
        // From diff[t] = phi[t] - phi[t-1]: coef = 1 for phi[t], -1 for phi[t-1]

        int t_prev = (t - 1 + n) % n;
        int t_next = (t + 1) % n;

        // Contribution from diff at t: phi[t] - phi[t_prev]
        grad_phi[t] -= inv_sigma2 * (phi[t] - phi[t_prev]);

        // Contribution from diff at t_next: phi[t_next] - phi[t]
        grad_phi[t] += inv_sigma2 * (phi[t_next] - phi[t]);
    }
}

inline double rw1_cyclic_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 0; t < n; t++) {
        int t_prev = (t - 1 + n) % n;
        double diff = phi[t] - phi[t_prev];
        quad += diff * diff;
    }
    // d/d(log_sigma2) = -0.5 * n + 0.5 / sigma2 * quad
    // Jacobian: multiply by sigma2
    return (-0.5 * n + 0.5 * quad / sigma2) * sigma2;
}

// =============================================================================
// RW2 gradients
// =============================================================================

inline void rw2_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    if (n < 3) {
        for (int t = 0; t < n; t++) grad_phi[t] = 0.0;
        return;
    }

    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    // Compute second differences
    std::vector<double> d(n);
    for (int t = 2; t < n; t++) {
        d[t] = phi[t] - 2.0 * phi[t-1] + phi[t-2];
    }

    // Gradient w.r.t. phi[k]
    for (int k = 0; k < n; k++) {
        grad_phi[k] = 0.0;
        // d[t] depends on phi[t], phi[t-1], phi[t-2]
        // So phi[k] affects d[k] (coef=1), d[k+1] (coef=-2), d[k+2] (coef=1)
        if (k >= 2 && k < n) {
            grad_phi[k] += -inv_sigma2 * d[k] * 1.0;
        }
        if (k >= 1 && k+1 < n && k+1 >= 2) {
            grad_phi[k] += -inv_sigma2 * d[k+1] * (-2.0);
        }
        if (k+2 < n) {
            grad_phi[k] += -inv_sigma2 * d[k+2] * 1.0;
        }
    }
}

inline double rw2_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 2; t < n; t++) {
        double d = phi[t] - 2.0 * phi[t-1] + phi[t-2];
        quad += d * d;
    }
    return (-0.5 * (n - 2) + 0.5 * quad / sigma2) * sigma2;
}

// =============================================================================
// AR1 gradients
// =============================================================================

inline void ar1_grad_phi(const double* phi, int n, double sigma2, double rho, double* grad_phi) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    double one_m_rho2 = 1.0 - rho * rho + 1e-10;
    double inv_var_stationary = one_m_rho2 * inv_sigma2;

    if (n == 1) {
        grad_phi[0] = -inv_var_stationary * phi[0];
        return;
    }

    // First time point
    double resid_1 = phi[1] - rho * phi[0];
    grad_phi[0] = -inv_var_stationary * phi[0] + inv_sigma2 * rho * resid_1;

    // Interior time points
    for (int t = 1; t < n - 1; t++) {
        double resid_t = phi[t] - rho * phi[t-1];
        double resid_tp1 = phi[t+1] - rho * phi[t];
        grad_phi[t] = -inv_sigma2 * resid_t + inv_sigma2 * rho * resid_tp1;
    }

    // Last time point
    double resid_T = phi[n-1] - rho * phi[n-2];
    grad_phi[n-1] = -inv_sigma2 * resid_T;
}

inline double ar1_grad_log_sigma2(const double* phi, int n, double sigma2, double rho) {
    double one_m_rho2 = 1.0 - rho * rho + 1e-10;

    // Normalization constant gradients
    double grad = -0.5;  // From stationary variance term
    grad -= 0.5 * (n - 1);  // From conditional variance terms

    // Quadratic terms
    double quad = one_m_rho2 * phi[0] * phi[0];
    for (int t = 1; t < n; t++) {
        double resid = phi[t] - rho * phi[t-1];
        quad += resid * resid;
    }
    grad += 0.5 * quad / sigma2;

    // Jacobian for log transform
    return grad * sigma2;
}

inline double ar1_grad_logit_rho(const double* phi, int n, double sigma2, double rho) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    double one_m_rho2 = 1.0 - rho * rho + 1e-10;

    // d log p / d rho from stationary variance
    double grad_rho = rho / one_m_rho2 * phi[0] * phi[0];

    // d log p / d rho from AR terms
    for (int t = 1; t < n; t++) {
        double resid = phi[t] - rho * phi[t-1];
        grad_rho += inv_sigma2 * resid * phi[t-1];
    }

    // Transform to logit_rho
    // u = (rho + 1) / 2, logit_rho = logit(u)
    // d(rho)/d(logit_rho) = 2 * u * (1-u)
    double u = (rho + 1.0) / 2.0;
    double d_rho_d_logit = 2.0 * u * (1.0 - u);

    return grad_rho * d_rho_d_logit;
}

// =============================================================================
// IID gradients
// =============================================================================

inline void iid_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    for (int t = 0; t < n; t++) {
        grad_phi[t] = -inv_sigma2 * phi[t];
    }
}

inline double iid_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 0; t < n; t++) {
        quad += phi[t] * phi[t];
    }
    return (-0.5 * n + 0.5 * quad / sigma2) * sigma2;
}

// =============================================================================
// PC prior gradient for sigma2 (on log_sigma2 scale)
// =============================================================================

// PC prior: log p(sigma) = log(rate) - rate * sigma - log(2*sigma)
// where rate = -log(alpha) / U
// d/d(log_sigma2) = d/d(sigma) * d(sigma)/d(log_sigma2)
//                 = (-rate - 1/sigma) * (sigma / 2)
//                 = -0.5 * rate * sigma - 0.5
inline double pc_prior_grad_log_sigma2(double sigma2, double U, double alpha) {
    double sigma = std::sqrt(sigma2 + 1e-10);
    double rate = -std::log(alpha + 1e-10) / (U + 1e-10);
    // d log_prior / d log_sigma2 = d log_prior / d sigma * d sigma / d log_sigma2
    // d log_prior / d sigma = -rate - 1/sigma
    // d sigma / d log_sigma2 = sigma / 2
    return (-rate - 1.0 / sigma) * (sigma / 2.0);
}

// =============================================================================
// Full multiscale temporal prior gradients
// =============================================================================

inline void multiscale_temporal_prior_gradients(
    const double* trend, int n_trend,
    const double* seasonal, int n_seasonal,
    const double* short_term, int n_short,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short,
    double rho_short,
    const MultiscaleTemporalData& temp_data,
    MultiscaleTemporalGradients& grads
) {
    // Initialize gradients
    grads.grad_trend.assign(n_trend, 0.0);
    grads.grad_seasonal.assign(n_seasonal, 0.0);
    grads.grad_short_term.assign(n_short, 0.0);
    grads.grad_log_sigma2_trend = 0.0;
    grads.grad_log_sigma2_seasonal = 0.0;
    grads.grad_log_sigma2_short = 0.0;
    grads.grad_logit_rho_short = 0.0;

    // Trend component
    if (n_trend > 0) {
        if (temp_data.trend_type == TemporalType::RW1) {
            rw1_grad_phi(trend, n_trend, sigma2_trend, grads.grad_trend.data());
            grads.grad_log_sigma2_trend = rw1_grad_log_sigma2(trend, n_trend, sigma2_trend);
        } else if (temp_data.trend_type == TemporalType::RW2) {
            rw2_grad_phi(trend, n_trend, sigma2_trend, grads.grad_trend.data());
            grads.grad_log_sigma2_trend = rw2_grad_log_sigma2(trend, n_trend, sigma2_trend);
        }
    }

    // Seasonal component (cyclic RW1)
    if (n_seasonal > 0 && temp_data.seasonal_period > 0) {
        rw1_cyclic_grad_phi(seasonal, n_seasonal, sigma2_seasonal, grads.grad_seasonal.data());
        grads.grad_log_sigma2_seasonal = rw1_cyclic_grad_log_sigma2(seasonal, n_seasonal, sigma2_seasonal);
    }

    // Short-term component
    if (n_short > 0) {
        if (temp_data.short_term_type == TemporalType::AR1) {
            ar1_grad_phi(short_term, n_short, sigma2_short, rho_short, grads.grad_short_term.data());
            grads.grad_log_sigma2_short = ar1_grad_log_sigma2(short_term, n_short, sigma2_short, rho_short);
            grads.grad_logit_rho_short = ar1_grad_logit_rho(short_term, n_short, sigma2_short, rho_short);
        } else if (temp_data.short_term_type == TemporalType::IID) {
            iid_grad_phi(short_term, n_short, sigma2_short, grads.grad_short_term.data());
            grads.grad_log_sigma2_short = iid_grad_log_sigma2(short_term, n_short, sigma2_short);
        }
    }
}

} // namespace ratiod_temporal_grad

#endif // RATIOD_HMC_MULTISCALE_TEMPORAL_GRAD_H
