// hmc_hsgp.h
// Hilbert Space Gaussian Process (HSGP) approximation
// Based on Riutort-Mayol et al. (2023) and Stan's implementation
//
// HSGP approximates GP as: f(x) = sum_j phi_j(x) * sqrt(S(lambda_j)) * beta_j
// where phi_j are Laplacian eigenfunctions and S is the spectral density

#ifndef RATIOD_HMC_HSGP_H
#define RATIOD_HMC_HSGP_H

#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ratiod_hsgp {

// HSGP data structure
struct HSGPData {
    int n_obs;           // Number of observations
    int n_dim;           // Number of dimensions (1 or 2)
    int m_per_dim;       // Basis functions per dimension
    int m_total;         // Total basis functions (m^d for d dimensions)

    double L1, L2;       // Boundary factors (domain is [-L, L])

    // Precomputed basis matrix: phi[i, j] = phi_j(x_i)
    // Stored as flat vector: phi_flat[i * m_total + j]
    std::vector<double> phi_flat;

    // Eigenvalues for each basis function
    std::vector<double> eigenvalues;

    // Scaled coordinates (mapped to [-1, 1])
    std::vector<double> coords_scaled;

    bool shared;         // Whether GP is shared between num/denom
};

// Spectral density for squared exponential kernel
// S(omega) = sigma^2 * sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2)
inline double spectral_density_se(double omega_sq, double sigma2, double lengthscale) {
    double ell = lengthscale;
    double ell2 = ell * ell;
    return sigma2 * std::sqrt(2.0 * M_PI) * ell * std::exp(-0.5 * ell2 * omega_sq);
}

// Derivative of spectral density w.r.t. sigma2
// dS/d(sigma2) = sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2) = S / sigma2
inline double dS_dsigma2(double omega_sq, double sigma2, double lengthscale) {
    return spectral_density_se(omega_sq, sigma2, lengthscale) / sigma2;
}

// Derivative of spectral density w.r.t. lengthscale
// S = sigma2 * sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2)
// dS/d(ell) = sigma2 * sqrt(2*pi) * [exp(...) + ell * (-ell * omega^2) * exp(...)]
//           = S * (1/ell - ell * omega^2)
inline double dS_dlengthscale(double omega_sq, double sigma2, double lengthscale) {
    double S = spectral_density_se(omega_sq, sigma2, lengthscale);
    double ell = lengthscale;
    return S * (1.0 / ell - ell * omega_sq);
}

// 1D Laplacian eigenfunction: phi_j(x) = 1/sqrt(L) * sin(pi*j*(x+L)/(2L))
// For x in [-L, L], j = 1, 2, ...
inline double phi_1d(double x, int j, double L) {
    double norm = 1.0 / std::sqrt(L);
    return norm * std::sin(M_PI * j * (x + L) / (2.0 * L));
}

// 1D eigenvalue: lambda_j = (pi*j / (2*L))^2
inline double lambda_1d(int j, double L) {
    double tmp = M_PI * j / (2.0 * L);
    return tmp * tmp;
}

// Setup HSGP for 2D coordinates
// coords: flattened [x1, y1, x2, y2, ...] (length 2*n_obs)
// m: basis functions per dimension
// c: boundary factor (L = c * max_range)
inline void setup_hsgp_2d(
    const std::vector<double>& coords,
    int n_obs,
    int m,
    double c,
    bool shared,
    HSGPData& data
) {
    data.n_obs = n_obs;
    data.n_dim = 2;
    data.m_per_dim = m;
    data.m_total = m * m;
    data.shared = shared;

    // Find coordinate ranges
    double x_min = coords[0], x_max = coords[0];
    double y_min = coords[1], y_max = coords[1];
    for (int i = 1; i < n_obs; i++) {
        double x = coords[2*i];
        double y = coords[2*i + 1];
        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
    }

    double x_range = x_max - x_min;
    double y_range = y_max - y_min;
    double x_center = (x_max + x_min) / 2.0;
    double y_center = (y_max + y_min) / 2.0;

    // Boundary factors
    data.L1 = c * x_range / 2.0;
    data.L2 = c * y_range / 2.0;

    // Ensure minimum boundary
    if (data.L1 < 0.1) data.L1 = 0.1;
    if (data.L2 < 0.1) data.L2 = 0.1;

    // Scale coordinates to [-L, L]
    data.coords_scaled.resize(2 * n_obs);
    for (int i = 0; i < n_obs; i++) {
        data.coords_scaled[2*i] = coords[2*i] - x_center;
        data.coords_scaled[2*i + 1] = coords[2*i + 1] - y_center;
    }

    // Compute eigenvalues for 2D: lambda_{j1,j2} = lambda_j1 + lambda_j2
    data.eigenvalues.resize(data.m_total);
    for (int j1 = 1; j1 <= m; j1++) {
        for (int j2 = 1; j2 <= m; j2++) {
            int idx = (j1 - 1) * m + (j2 - 1);
            data.eigenvalues[idx] = lambda_1d(j1, data.L1) + lambda_1d(j2, data.L2);
        }
    }

    // Compute basis matrix: phi[i, j] = phi_{j1}(x_i) * phi_{j2}(y_i)
    data.phi_flat.resize(n_obs * data.m_total);
    for (int i = 0; i < n_obs; i++) {
        double x = data.coords_scaled[2*i];
        double y = data.coords_scaled[2*i + 1];

        for (int j1 = 1; j1 <= m; j1++) {
            double phi_x = phi_1d(x, j1, data.L1);
            for (int j2 = 1; j2 <= m; j2++) {
                double phi_y = phi_1d(y, j2, data.L2);
                int j_idx = (j1 - 1) * m + (j2 - 1);
                data.phi_flat[i * data.m_total + j_idx] = phi_x * phi_y;
            }
        }
    }
}

// Evaluate HSGP spatial effects: f = Phi * sqrt(S) * beta
inline void hsgp_evaluate(
    const std::vector<double>& beta,
    double sigma2,
    double lengthscale,
    const HSGPData& data,
    std::vector<double>& f
) {
    f.assign(data.n_obs, 0.0);

    // Precompute sqrt(S) for each basis
    std::vector<double> sqrt_S(data.m_total);
    for (int j = 0; j < data.m_total; j++) {
        double S = spectral_density_se(data.eigenvalues[j], sigma2, lengthscale);
        sqrt_S[j] = std::sqrt(S);
    }

    // f_i = sum_j phi[i,j] * sqrt(S[j]) * beta[j]
    for (int i = 0; i < data.n_obs; i++) {
        double fi = 0.0;
        for (int j = 0; j < data.m_total; j++) {
            fi += data.phi_flat[i * data.m_total + j] * sqrt_S[j] * beta[j];
        }
        f[i] = fi;
    }
}

// Log prior on beta: N(0, I)
inline double hsgp_log_prior_beta(const std::vector<double>& beta) {
    double log_prior = 0.0;
    for (size_t j = 0; j < beta.size(); j++) {
        log_prior += -0.5 * beta[j] * beta[j];
    }
    return log_prior;
}

// Gradient structure for HSGP
struct HSGPGradients {
    std::vector<double> grad_beta;  // Gradient w.r.t. beta
    double grad_log_sigma2;         // Gradient w.r.t. log(sigma2)
    double grad_log_lengthscale;    // Gradient w.r.t. log(lengthscale)
};

// Compute HSGP gradients analytically
// grad_f: gradient of log-likelihood w.r.t. f (computed from likelihood)
inline void hsgp_compute_gradients(
    const std::vector<double>& beta,
    double sigma2,
    double lengthscale,
    const HSGPData& data,
    const std::vector<double>& grad_f,  // d(log_lik)/d(f_i)
    HSGPGradients& grads
) {
    int m_total = data.m_total;
    grads.grad_beta.assign(m_total, 0.0);
    grads.grad_log_sigma2 = 0.0;
    grads.grad_log_lengthscale = 0.0;

    // Precompute sqrt(S) and derivatives
    std::vector<double> sqrt_S(m_total);
    std::vector<double> dsqrtS_dsigma2(m_total);
    std::vector<double> dsqrtS_dlengthscale(m_total);

    for (int j = 0; j < m_total; j++) {
        double omega_sq = data.eigenvalues[j];
        double S = spectral_density_se(omega_sq, sigma2, lengthscale);
        sqrt_S[j] = std::sqrt(S);

        // d(sqrt(S))/d(sigma2) = 0.5 / sqrt(S) * dS/d(sigma2) = 0.5 / sqrt(S) * S/sigma2
        //                      = 0.5 * sqrt(S) / sigma2
        dsqrtS_dsigma2[j] = 0.5 * sqrt_S[j] / sigma2;

        // d(sqrt(S))/d(ell) = 0.5 / sqrt(S) * dS/d(ell)
        double dS_dell = dS_dlengthscale(omega_sq, sigma2, lengthscale);
        dsqrtS_dlengthscale[j] = 0.5 * dS_dell / sqrt_S[j];
    }

    // Gradient w.r.t. beta: sum_i grad_f[i] * phi[i,j] * sqrt(S[j])
    // Plus prior: -beta[j]
    for (int j = 0; j < m_total; j++) {
        double grad_j = 0.0;
        for (int i = 0; i < data.n_obs; i++) {
            grad_j += grad_f[i] * data.phi_flat[i * m_total + j] * sqrt_S[j];
        }
        grads.grad_beta[j] = grad_j - beta[j];  // Prior contribution
    }

    // Gradient w.r.t. log(sigma2):
    // d(log_lik)/d(log_sigma2) = d(log_lik)/d(sigma2) * sigma2
    // d(log_lik)/d(sigma2) = sum_i grad_f[i] * d(f_i)/d(sigma2)
    //                      = sum_i grad_f[i] * sum_j phi[i,j] * d(sqrt(S[j]))/d(sigma2) * beta[j]
    double grad_sigma2 = 0.0;
    for (int i = 0; i < data.n_obs; i++) {
        double df_dsigma2 = 0.0;
        for (int j = 0; j < m_total; j++) {
            df_dsigma2 += data.phi_flat[i * m_total + j] * dsqrtS_dsigma2[j] * beta[j];
        }
        grad_sigma2 += grad_f[i] * df_dsigma2;
    }
    grads.grad_log_sigma2 = grad_sigma2 * sigma2;  // Chain rule

    // Add prior contribution for sigma2: PC prior
    // log p(sigma) = log(rate) - rate*sigma - log(2*sigma)
    // d/d(sigma2) = d/d(sigma) * d(sigma)/d(sigma2) = (-rate - 1/sigma) / (2*sigma)
    double sigma = std::sqrt(sigma2);
    double rate_sigma = 4.6;  // P(sigma > 1) = 0.01
    double grad_prior_sigma2 = (-rate_sigma - 1.0/sigma) / (2.0 * sigma);
    grads.grad_log_sigma2 += grad_prior_sigma2 * sigma2;

    // Gradient w.r.t. log(lengthscale):
    double grad_ell = 0.0;
    for (int i = 0; i < data.n_obs; i++) {
        double df_dell = 0.0;
        for (int j = 0; j < m_total; j++) {
            df_dell += data.phi_flat[i * m_total + j] * dsqrtS_dlengthscale[j] * beta[j];
        }
        grad_ell += grad_f[i] * df_dell;
    }
    grads.grad_log_lengthscale = grad_ell * lengthscale;  // Chain rule

    // Add prior contribution for lengthscale: LogNormal(0, 1)
    // log p(ell) = -0.5 * log(ell)^2 - log(ell) (the -log(ell) is Jacobian)
    // d/d(log_ell) = -log(ell) - 1
    double log_ell = std::log(lengthscale);
    grads.grad_log_lengthscale += -log_ell - 1.0;
}

} // namespace ratiod_hsgp

#endif // RATIOD_HMC_HSGP_H
