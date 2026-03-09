// hmc_icar_collapsed.h
// Collapsed/marginalized ICAR and BYM2 spatial effects via inner Laplace optimization
//
// Instead of sampling S (ICAR) or 2S (BYM2) spatial effects alongside hyperparameters,
// we marginalize them out by finding phi* = argmax [log p(y|phi,theta_outer) + log p(phi|tau)]
// at each HMC gradient evaluation.
//
// ICAR: reduces S+1 params (log_tau + S phi) to just 1 (log_tau)
// BYM2: reduces 2S+2 params (log_sigma, logit_rho, S phi, S theta) to 2 (log_sigma, logit_rho)
//
// The collapsed log-posterior is:
//   log p(theta|y) ~ log p(y|phi*,theta) + log p(phi*|tau) + log p(theta)
//                    - 0.5 * log det(W + tau*Q)  [Laplace correction]
//
// Key advantage over collapsed GP: Q is FIXED (adjacency-based), doesn't depend on
// hyperparameters. Only tau*Q changes. This makes numerical Laplace gradient cheaper.

#ifndef RATIOD_HMC_ICAR_COLLAPSED_H
#define RATIOD_HMC_ICAR_COLLAPSED_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppEigen.h>

// NOTE: Must be included AFTER hmc_sampler.h (which defines ModelData/ModelType).
using ratiod_hmc::ModelData;
using ratiod_hmc::ModelType;
using ratiod_hmc::SpatialType;

// =========================================================================
// Workspace for collapsed ICAR/BYM2 computations
// =========================================================================

struct CollapsedICARWorkspace {
    int S = 0;                          // Number of spatial units
    bool is_bym2 = false;               // BYM2 mode (2S inner variables)
    int inner_dim = 0;                  // S for ICAR, 2S for BYM2

    // Mode variables
    std::vector<double> phi_star;       // Structured spatial mode (S)
    std::vector<double> theta_star;     // Unstructured mode (S, BYM2 only)

    // Data-level Hessian diagonal (per spatial unit)
    std::vector<double> W_data;         // sum_i(-d²LL/deta²) at unit s (length S)

    // Newton workspace
    std::vector<double> grad;           // gradient (inner_dim)
    std::vector<double> hess_diag;      // diagonal part of Hessian (inner_dim)
    std::vector<double> cg_r, cg_p, cg_Ap;  // CG workspace (inner_dim)

    // Laplace correction
    double laplace_log_det = 0.0;       // -0.5 * log det(H)

    bool mode_found = false;

    void init(int n_units, bool bym2) {
        if (n_units != S || bym2 != is_bym2) {
            mode_found = false;
        }
        S = n_units;
        is_bym2 = bym2;
        inner_dim = bym2 ? 2 * S : S;

        phi_star.assign(S, 0.0);
        W_data.assign(S, 0.0);
        grad.assign(inner_dim, 0.0);
        hess_diag.assign(inner_dim, 0.0);
        cg_r.assign(inner_dim, 0.0);
        cg_p.assign(inner_dim, 0.0);
        cg_Ap.assign(inner_dim, 0.0);

        if (bym2) {
            theta_star.assign(S, 0.0);
        } else {
            theta_star.clear();
        }
    }
};

// =========================================================================
// ICAR precision matrix operations (Q = D - W, adjacency-based)
// =========================================================================

// Compute Q*v where Q[i,i] = n_neighbors[i], Q[i,j] = -1 if j~i
// Uses CSR adjacency from ModelData
inline void icar_precision_matvec(
    const double* v,
    double* result,
    int S,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors
) {
    for (int i = 0; i < S; i++) {
        result[i] = n_neighbors[i] * v[i];
        for (int k = adj_row_ptr[i]; k < adj_row_ptr[i + 1]; k++) {
            result[i] -= v[adj_col_idx[k]];
        }
    }
}

// =========================================================================
// (W + tau*Q + lambda*I) matvec for ICAR CG solver
// W = diag(W_data), Q = ICAR precision, lambda = sum-to-zero penalty
// =========================================================================

// ICAR mode: (W + tau*Q + lambda*I) v  +  lambda_s2z * sum(v) * ones
// The sum-to-zero penalty: -0.5 * lambda_s2z * (sum phi)^2
//   → adds lambda_s2z * 11^T to the Hessian
// For efficiency: compute Q*v, then add diagonal + dense rank-1 terms
inline void icar_hessian_matvec(
    const double* v,
    double* result,
    const CollapsedICARWorkspace& ws,
    double tau,
    const ModelData& data,
    double lambda_s2z = 0.001  // sum-to-zero penalty strength
) {
    int S = ws.S;

    // Q * v
    icar_precision_matvec(v, result, S, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);

    // Scale by tau and add W diagonal + sum-to-zero rank-1 term
    double sum_v = 0.0;
    for (int i = 0; i < S; i++) sum_v += v[i];

    for (int i = 0; i < S; i++) {
        result[i] = ws.W_data[i] * v[i] + tau * result[i] + lambda_s2z * sum_v;
    }
}

// BYM2 mode: full 2S Hessian matvec
// Variables: [phi_0..phi_{S-1}, theta_0..theta_{S-1}]
// Block structure:
//   phi-phi:     a²*W_data + tau_icar*Q + lambda_s2z*11^T
//   theta-theta: c²*W_data + I
//   phi-theta:   a*c*W_data (diagonal)
// where a = sigma_s * scale, c = sigma_u
inline void bym2_hessian_matvec(
    const double* v,           // length 2S: [v_phi; v_theta]
    double* result,            // length 2S: [result_phi; result_theta]
    const CollapsedICARWorkspace& ws,
    double a,                  // sigma_s * scale = sigma_total * sqrt(rho) * scale
    double c,                  // sigma_u = sigma_total * sqrt(1-rho)
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    const double* v_phi = v;
    const double* v_theta = v + S;
    double* r_phi = result;
    double* r_theta = result + S;

    // Q * v_phi (ICAR structure)
    std::vector<double> Qv(S);
    icar_precision_matvec(v_phi, Qv.data(), S, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);

    double sum_vphi = 0.0;
    for (int i = 0; i < S; i++) sum_vphi += v_phi[i];

    double a2 = a * a;
    double c2 = c * c;
    double ac = a * c;

    for (int i = 0; i < S; i++) {
        double Wd = ws.W_data[i];
        r_phi[i] = a2 * Wd * v_phi[i] + Qv[i] + lambda_s2z * sum_vphi
                   + ac * Wd * v_theta[i];
        r_theta[i] = ac * Wd * v_phi[i] + (c2 * Wd + 1.0) * v_theta[i];
    }
}

// =========================================================================
// CG solver for collapsed ICAR/BYM2
// =========================================================================

// Generic CG: solves H*x = b where H is applied via function pointer
// For ICAR: H = (W + tau*Q + lambda_s2z*11^T)
// For BYM2: H = full 2S block Hessian
inline int icar_cg_solve(
    double* x,           // Solution (in/out, warm-started)
    const double* b,     // RHS
    CollapsedICARWorkspace& ws,
    double tau,          // ICAR precision (or 1.0 for BYM2 since Q has no tau)
    double a,            // BYM2: sigma_s * scale (unused for ICAR)
    double cc,           // BYM2: sigma_u (unused for ICAR)
    const ModelData& data,
    int max_iter = 100,
    double tol = 1e-8
) {
    int N = ws.inner_dim;

    // r = b - H*x
    if (ws.is_bym2) {
        bym2_hessian_matvec(x, ws.cg_Ap.data(), ws, a, cc, data);
    } else {
        icar_hessian_matvec(x, ws.cg_Ap.data(), ws, tau, data);
    }
    for (int i = 0; i < N; i++) ws.cg_r[i] = b[i] - ws.cg_Ap[i];

    std::memcpy(ws.cg_p.data(), ws.cg_r.data(), N * sizeof(double));

    double rr = 0.0;
    for (int i = 0; i < N; i++) rr += ws.cg_r[i] * ws.cg_r[i];

    double b_norm = 0.0;
    for (int i = 0; i < N; i++) b_norm += b[i] * b[i];
    if (b_norm < 1e-30) return 0;

    for (int iter = 0; iter < max_iter; iter++) {
        if (rr / b_norm < tol * tol) return iter;

        if (ws.is_bym2) {
            bym2_hessian_matvec(ws.cg_p.data(), ws.cg_Ap.data(), ws, a, cc, data);
        } else {
            icar_hessian_matvec(ws.cg_p.data(), ws.cg_Ap.data(), ws, tau, data);
        }

        double pAp = 0.0;
        for (int i = 0; i < N; i++) pAp += ws.cg_p[i] * ws.cg_Ap[i];
        if (pAp < 1e-30) return iter;

        double alpha_cg = rr / pAp;
        for (int i = 0; i < N; i++) {
            x[i] += alpha_cg * ws.cg_p[i];
            ws.cg_r[i] -= alpha_cg * ws.cg_Ap[i];
        }

        double rr_new = 0.0;
        for (int i = 0; i < N; i++) rr_new += ws.cg_r[i] * ws.cg_r[i];

        double beta_cg = rr_new / rr;
        for (int i = 0; i < N; i++) {
            ws.cg_p[i] = ws.cg_r[i] + beta_cg * ws.cg_p[i];
        }
        rr = rr_new;
    }
    return max_iter;
}

// =========================================================================
// Laplace log-det correction via sparse Cholesky
// =========================================================================

// ICAR: build (W + tau*Q + lambda*I) and compute -0.5 * log det
inline double compute_laplace_log_det_icar(
    const CollapsedICARWorkspace& ws,
    double tau,
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(S + 2 * data.adj_col_idx.size());

    // Diagonal: W_data[i] + tau * n_neighbors[i] + lambda_s2z
    for (int i = 0; i < S; i++) {
        triplets.push_back(T(i, i, ws.W_data[i] + tau * data.n_neighbors[i] + lambda_s2z));
    }

    // Off-diagonal: -tau * adj(i,j) (from ICAR Q = D - W)
    for (int i = 0; i < S; i++) {
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            int j = data.adj_col_idx[k];
            triplets.push_back(T(i, j, -tau));
        }
    }

    Eigen::SparseMatrix<double> H(S, S);
    H.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(H);
    if (llt.info() != Eigen::Success) {
        // Add jitter and retry
        for (int i = 0; i < S; i++) {
            triplets.push_back(T(i, i, 1e-6));
        }
        H.setFromTriplets(triplets.begin(), triplets.end());
        llt.compute(H);
        if (llt.info() != Eigen::Success) return 0.0;
    }

    Eigen::SparseMatrix<double> L_sparse = llt.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < S; i++) {
        log_det += std::log(L_sparse.coeff(i, i));
    }
    log_det *= 2.0;

    return -0.5 * log_det;
}

// BYM2: build 2S × 2S block Hessian and compute -0.5 * log det
// [[a²*W + Q + lambda_s2z*I,  a*c*W   ],
//  [a*c*W,                   c²*W + I  ]]
inline double compute_laplace_log_det_bym2(
    const CollapsedICARWorkspace& ws,
    double a, double cc,
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    int dim = 2 * S;

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(dim + 4 * S + 2 * data.adj_col_idx.size());

    double a2 = a * a;
    double c2 = cc * cc;
    double ac = a * cc;

    // phi-phi block (top-left S×S): a²*W + Q + lambda_s2z*I
    for (int i = 0; i < S; i++) {
        triplets.push_back(T(i, i, a2 * ws.W_data[i] + data.n_neighbors[i] + lambda_s2z));
    }
    for (int i = 0; i < S; i++) {
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            int j = data.adj_col_idx[k];
            triplets.push_back(T(i, j, -1.0));  // -Q off-diagonal (no tau for BYM2)
        }
    }

    // theta-theta block (bottom-right S×S): c²*W + I
    for (int i = 0; i < S; i++) {
        triplets.push_back(T(S + i, S + i, c2 * ws.W_data[i] + 1.0));
    }

    // phi-theta cross blocks (diagonal): a*c*W
    for (int i = 0; i < S; i++) {
        if (std::abs(ac * ws.W_data[i]) > 1e-15) {
            triplets.push_back(T(i, S + i, ac * ws.W_data[i]));
            triplets.push_back(T(S + i, i, ac * ws.W_data[i]));
        }
    }

    Eigen::SparseMatrix<double> H(dim, dim);
    H.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(H);
    if (llt.info() != Eigen::Success) {
        for (int i = 0; i < dim; i++) {
            triplets.push_back(T(i, i, 1e-6));
        }
        H.setFromTriplets(triplets.begin(), triplets.end());
        llt.compute(H);
        if (llt.info() != Eigen::Success) return 0.0;
    }

    Eigen::SparseMatrix<double> L_sparse = llt.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < dim; i++) {
        log_det += std::log(L_sparse.coeff(i, i));
    }
    log_det *= 2.0;

    return -0.5 * log_det;
}

// =========================================================================
// Per-spatial-unit likelihood (aggregated over observations at each unit)
// =========================================================================

struct UnitLikResult {
    double ll;          // log-likelihood contribution
    double grad;        // d(ll)/d(spatial_effect)
    double neg_hess;    // -d²(ll)/d(spatial_effect)²
};

// Compute data log-likelihood, gradient, and Hessian at spatial unit s,
// where spatial_eff[s] enters eta for all observations at unit s.
// RE effects are included via re_vals (pre-computed actual RE values).
inline UnitLikResult compute_unit_lik(
    int s,                      // spatial unit index (0-based)
    double spatial_eff,         // spatial effect at this unit
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const double* re_vals,      // pre-computed RE values (actual, not z), length n_re_groups or NULL
    const ModelData& data,
    bool is_binomial
) {
    UnitLikResult res = {0.0, 0.0, 0.0};
    int N = data.N;

    for (int i = 0; i < N; i++) {
        if (data.spatial_group[i] - 1 != s) continue;  // spatial_group is 1-based

        // Compute eta
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.p_num; p++)
            eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.p_denom; p++)
                eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
        }

        // Add spatial effect
        eta_num_i += spatial_eff;
        if (!is_binomial) eta_denom_i += spatial_eff;  // shared by default

        // Add RE
        if (re_vals != nullptr && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            double re_val = re_vals[data.re_group[i] - 1];
            eta_num_i += re_val;
            if (!is_binomial) eta_denom_i += re_val;
        }

        // Per-family likelihood, gradient, Hessian
        double mu_num = std::exp(std::min(eta_num_i, 20.0));
        int y_num = data.y_num[i];

        switch (data.model_type) {
            case ModelType::POISSON_GAMMA: {
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    double y_denom = data.y_denom_cont[i];
                    double shape = phi_denom;
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(std::max(y_denom, 1e-10))
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += shape * (y_denom / mu_denom - 1.0);
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    int y_denom = (int)data.y_denom[i];
                    double r_denom = phi_denom;
                    res.ll += std::lgamma(y_denom + r_denom) - std::lgamma(r_denom) - std::lgamma(y_denom + 1)
                              + y_denom * eta_denom_i - (y_denom + r_denom) * std::log(mu_denom + r_denom)
                              + r_denom * std::log(r_denom);
                    double resid_denom = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                    res.grad += resid_denom;
                    res.neg_hess += mu_denom * r_denom * (y_denom + r_denom) / ((mu_denom + r_denom) * (mu_denom + r_denom));
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));
                // Gamma denominator
                {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    double y_denom = data.y_denom_cont[i];
                    double shape = phi_denom;
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(std::max(y_denom, 1e-10))
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += shape * (y_denom / mu_denom - 1.0);
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                res.ll += y_num * eta_num_i - n_trials * std::log(1.0 + std::exp(eta_num_i));
                res.grad += y_num - n_trials * p_i;
                res.neg_hess += n_trials * p_i * (1.0 - p_i);
                break;
            }
            default:
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                break;
        }
    }
    return res;
}

// =========================================================================
// Newton-Raphson for finding phi* (ICAR inner optimization)
// =========================================================================

// Find phi* = argmax [LL(y|phi,outer) + (-0.5*tau*phi'Q*phi) + (-0.5*lambda*(sum phi)^2)]
// Returns data log-likelihood + ICAR prior at phi* (not including outer param priors)
inline double collapsed_icar_find_mode(
    const double* beta_num, const double* beta_denom,
    double tau,
    double phi_num, double phi_denom,
    const double* re_vals,      // pre-computed RE values or NULL
    const ModelData& data,
    CollapsedICARWorkspace& ws,
    int max_newton = 20,
    double newton_tol = 1e-6
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    ws.init(S, false);

    // Warm-start from previous mode
    if (!ws.mode_found) {
        std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
    } else {
        for (int i = 0; i < S; i++) {
            if (std::isnan(ws.phi_star[i]) || std::isinf(ws.phi_star[i])) {
                std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
                break;
            }
        }
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(S, 0.0);

    for (int newton_iter = 0; newton_iter < max_newton; newton_iter++) {
        // Compute per-unit data likelihood, gradient, and Hessian
        std::fill(ws.grad.begin(), ws.grad.end(), 0.0);
        std::fill(ws.W_data.begin(), ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            UnitLikResult lr = compute_unit_lik(s, ws.phi_star[s],
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial);
            ws.grad[s] = lr.grad;
            ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        // Add ICAR prior gradient: -tau * Q * phi
        icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += ws.phi_star[i];
        for (int i = 0; i < S; i++) {
            ws.grad[i] -= tau * Qphi[i] + 0.001 * sum_phi;  // ICAR + sum-to-zero
        }

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < S; i++) grad_norm += ws.grad[i] * ws.grad[i];
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < newton_tol) break;

        // Solve (W + tau*Q + lambda_s2z*11^T) delta = grad via CG
        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), ws.grad.data(), ws, tau, 0.0, 0.0, data, 100, 1e-8);

        for (int i = 0; i < S; i++) ws.phi_star[i] += delta[i];
    }

    ws.mode_found = true;

    // Compute log-posterior at phi*
    double data_ll = 0.0;
    for (int s = 0; s < S; s++) {
        UnitLikResult lr = compute_unit_lik(s, ws.phi_star[s],
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial);
        data_ll += lr.ll;
        ws.W_data[s] = std::max(lr.neg_hess, 1e-8);  // Update for Laplace
    }

    // ICAR prior: -0.5 * tau * phi' Q phi + 0.5*(S-1)*log(tau) - 0.5*0.001*(sum phi)^2
    icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                          data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    double phiQphi = 0.0;
    for (int i = 0; i < S; i++) phiQphi += ws.phi_star[i] * Qphi[i];
    double sum_phi = 0.0;
    for (int i = 0; i < S; i++) sum_phi += ws.phi_star[i];
    double icar_prior = -0.5 * tau * phiQphi + 0.5 * (S - 1) * std::log(tau)
                        - 0.5 * 0.001 * sum_phi * sum_phi;

    // Laplace correction
    ws.laplace_log_det = compute_laplace_log_det_icar(ws, tau, data);

    return data_ll + icar_prior;
}

// =========================================================================
// BYM2 mode: Newton for (phi*, theta*)
// =========================================================================

inline double collapsed_bym2_find_mode(
    const double* beta_num, const double* beta_denom,
    double sigma_total, double rho, double scale_factor,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    CollapsedICARWorkspace& ws,
    int max_newton = 20,
    double newton_tol = 1e-6
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    double sigma_s = sigma_total * std::sqrt(rho);
    double sigma_u = sigma_total * std::sqrt(1.0 - rho);
    double a = sigma_s * scale_factor;  // coefficient for phi
    double c = sigma_u;                  // coefficient for theta

    ws.init(S, true);

    // Warm-start
    if (!ws.mode_found) {
        std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
        std::fill(ws.theta_star.begin(), ws.theta_star.end(), 0.0);
    } else {
        bool has_nan = false;
        for (int i = 0; i < S; i++) {
            if (std::isnan(ws.phi_star[i]) || std::isinf(ws.phi_star[i]) ||
                std::isnan(ws.theta_star[i]) || std::isinf(ws.theta_star[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
            std::fill(ws.theta_star.begin(), ws.theta_star.end(), 0.0);
        }
    }

    // Combined inner variable: [phi; theta]
    std::vector<double> inner(2 * S, 0.0);
    for (int i = 0; i < S; i++) {
        inner[i] = ws.phi_star[i];
        inner[S + i] = ws.theta_star[i];
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(2 * S, 0.0);

    for (int newton_iter = 0; newton_iter < max_newton; newton_iter++) {
        // Compute spatial effect: b_s = a*phi_s + c*theta_s
        std::fill(ws.grad.begin(), ws.grad.end(), 0.0);
        std::fill(ws.W_data.begin(), ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            double b_s = a * inner[s] + c * inner[S + s];
            UnitLikResult lr = compute_unit_lik(s, b_s,
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial);
            // Data gradients w.r.t. phi and theta (chain rule through b_s)
            ws.grad[s] = lr.grad * a;          // dLL/dphi_s = dLL/db * a
            ws.grad[S + s] = lr.grad * c;      // dLL/dtheta_s = dLL/db * c
            ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        // Add prior gradients
        // phi: ICAR prior -0.5*phi'Q*phi → gradient = -Q*phi
        icar_precision_matvec(inner.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += inner[i];
        for (int i = 0; i < S; i++) {
            ws.grad[i] -= Qphi[i] + 0.001 * sum_phi;  // ICAR + sum-to-zero
        }
        // theta: IID N(0,1) → gradient = -theta
        for (int i = 0; i < S; i++) {
            ws.grad[S + i] -= inner[S + i];
        }

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < 2 * S; i++) grad_norm += ws.grad[i] * ws.grad[i];
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < newton_tol) break;

        // CG solve for 2S system
        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), ws.grad.data(), ws, 1.0, a, c, data, 100, 1e-8);

        for (int i = 0; i < 2 * S; i++) inner[i] += delta[i];
    }

    // Store back
    for (int i = 0; i < S; i++) {
        ws.phi_star[i] = inner[i];
        ws.theta_star[i] = inner[S + i];
    }
    ws.mode_found = true;

    // Compute log-posterior at mode
    double data_ll = 0.0;
    for (int s = 0; s < S; s++) {
        double b_s = a * ws.phi_star[s] + c * ws.theta_star[s];
        UnitLikResult lr = compute_unit_lik(s, b_s,
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial);
        data_ll += lr.ll;
        ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
    }

    // ICAR prior on phi: -0.5 * phi' Q phi - 0.5*0.001*(sum phi)^2
    icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                          data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    double phiQphi = 0.0;
    double sum_phi = 0.0;
    for (int i = 0; i < S; i++) {
        phiQphi += ws.phi_star[i] * Qphi[i];
        sum_phi += ws.phi_star[i];
    }
    double phi_prior = -0.5 * phiQphi - 0.5 * 0.001 * sum_phi * sum_phi;

    // IID prior on theta: -0.5 * sum(theta^2)
    double theta_prior = 0.0;
    for (int i = 0; i < S; i++) {
        theta_prior -= 0.5 * ws.theta_star[i] * ws.theta_star[i];
    }

    // Laplace correction
    ws.laplace_log_det = compute_laplace_log_det_bym2(ws, a, c, data);

    return data_ll + phi_prior + theta_prior;
}

// =========================================================================
// Residual computation at mode (for scattering to outer param gradients)
// =========================================================================

// Compute per-observation residuals dLL/deta at the spatial mode
inline void collapsed_icar_compute_residuals(
    const CollapsedICARWorkspace& ws,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const double* re_vals,
    double a_bym2, double c_bym2,  // BYM2 scaling factors (a=0, c=0 for ICAR)
    const ModelData& data,
    double* resid_num,  // length N
    double* resid_denom // length N
) {
    int N = data.N;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;  // 0-based

        // Spatial effect at this unit
        double spatial_eff;
        if (ws.is_bym2) {
            spatial_eff = a_bym2 * ws.phi_star[s] + c_bym2 * ws.theta_star[s];
        } else {
            spatial_eff = ws.phi_star[s];
        }

        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.p_num; p++)
            eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.p_denom; p++)
                eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
        }

        eta_num_i += spatial_eff;
        if (!is_binomial) eta_denom_i += spatial_eff;

        if (re_vals != nullptr && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            eta_num_i += re_vals[data.re_group[i] - 1];
            if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
        }

        double mu_num = std::exp(std::min(eta_num_i, 20.0));

        switch (data.model_type) {
            case ModelType::POISSON_GAMMA: {
                resid_num[i] = data.y_num[i] - mu_num;
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    resid_denom[i] = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
                } else {
                    resid_denom[i] = 0.0;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                resid_num[i] = data.y_num[i] - mu_num * (data.y_num[i] + r_num) / (mu_num + r_num);
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    int y_denom = (int)data.y_denom[i];
                    double r_denom = phi_denom;
                    resid_denom[i] = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                } else {
                    resid_denom[i] = 0.0;
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_num = phi_num;
                resid_num[i] = data.y_num[i] - mu_num * (data.y_num[i] + r_num) / (mu_num + r_num);
                double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                double shape = phi_denom;
                resid_denom[i] = shape * (data.y_denom_cont[i] / mu_denom - 1.0);
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                resid_num[i] = data.y_num[i] - n_trials * p_i;
                resid_denom[i] = 0.0;
                break;
            }
            default:
                resid_num[i] = data.y_num[i] - mu_num;
                resid_denom[i] = 0.0;
                break;
        }
    }
}

// =========================================================================
// Full Laplace log-det with Newton re-solve (for numerical gradient)
// =========================================================================

// Compute Laplace log-det for given params, with warm-started Newton from phi*
// Used for central-difference numerical gradient of the Laplace correction.
inline double laplace_log_det_icar_full(
    const double* beta_num, const double* beta_denom,
    double tau,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    const std::vector<double>& warm_phi,        // warm start
    const std::vector<double>& warm_theta = {}  // for BYM2
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    // Temporary workspace with warm start
    CollapsedICARWorkspace temp_ws;
    temp_ws.init(S, !warm_theta.empty());
    temp_ws.phi_star = warm_phi;
    if (!warm_theta.empty()) temp_ws.theta_star = warm_theta;
    temp_ws.mode_found = true;

    // Short Newton (warm-started, 5 iters max)
    if (temp_ws.is_bym2) {
        // For BYM2 we need sigma_total, rho, scale from the caller
        // This function is ICAR-only; BYM2 uses laplace_log_det_bym2_full
        return 0.0;
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(S, 0.0);

    for (int iter = 0; iter < 5; iter++) {
        std::fill(temp_ws.grad.begin(), temp_ws.grad.end(), 0.0);
        std::fill(temp_ws.W_data.begin(), temp_ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            UnitLikResult lr = compute_unit_lik(s, temp_ws.phi_star[s],
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial);
            temp_ws.grad[s] = lr.grad;
            temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        icar_precision_matvec(temp_ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += temp_ws.phi_star[i];
        for (int i = 0; i < S; i++) {
            temp_ws.grad[i] -= tau * Qphi[i] + 0.001 * sum_phi;
        }

        double grad_norm = 0.0;
        for (int i = 0; i < S; i++) grad_norm += temp_ws.grad[i] * temp_ws.grad[i];
        if (std::sqrt(grad_norm) < 1e-6) break;

        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), temp_ws.grad.data(), temp_ws, tau, 0.0, 0.0, data, 50, 1e-8);
        for (int i = 0; i < S; i++) temp_ws.phi_star[i] += delta[i];
    }

    // Final Hessian at new mode
    for (int s = 0; s < S; s++) {
        UnitLikResult lr = compute_unit_lik(s, temp_ws.phi_star[s],
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial);
        temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det_icar(temp_ws, tau, data);
}

// BYM2 version of full Laplace log-det with Newton
inline double laplace_log_det_bym2_full(
    const double* beta_num, const double* beta_denom,
    double sigma_total, double rho, double scale_factor,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    const std::vector<double>& warm_phi,
    const std::vector<double>& warm_theta
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    double sigma_s = sigma_total * std::sqrt(rho);
    double sigma_u = sigma_total * std::sqrt(1.0 - rho);
    double a = sigma_s * scale_factor;
    double c = sigma_u;

    CollapsedICARWorkspace temp_ws;
    temp_ws.init(S, true);
    temp_ws.phi_star = warm_phi;
    temp_ws.theta_star = warm_theta;
    temp_ws.mode_found = true;

    // Combined inner variable
    std::vector<double> inner(2 * S);
    for (int i = 0; i < S; i++) {
        inner[i] = temp_ws.phi_star[i];
        inner[S + i] = temp_ws.theta_star[i];
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(2 * S, 0.0);

    for (int iter = 0; iter < 5; iter++) {
        std::fill(temp_ws.grad.begin(), temp_ws.grad.end(), 0.0);
        std::fill(temp_ws.W_data.begin(), temp_ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            double b_s = a * inner[s] + c * inner[S + s];
            UnitLikResult lr = compute_unit_lik(s, b_s,
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial);
            temp_ws.grad[s] = lr.grad * a;
            temp_ws.grad[S + s] = lr.grad * c;
            temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        icar_precision_matvec(inner.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += inner[i];
        for (int i = 0; i < S; i++) {
            temp_ws.grad[i] -= Qphi[i] + 0.001 * sum_phi;
            temp_ws.grad[S + i] -= inner[S + i];
        }

        double grad_norm = 0.0;
        for (int i = 0; i < 2 * S; i++) grad_norm += temp_ws.grad[i] * temp_ws.grad[i];
        if (std::sqrt(grad_norm) < 1e-6) break;

        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), temp_ws.grad.data(), temp_ws, 1.0, a, c, data, 50, 1e-8);
        for (int i = 0; i < 2 * S; i++) inner[i] += delta[i];
    }

    // Update W_data at final mode
    for (int s = 0; s < S; s++) {
        double b_s = a * inner[s] + c * inner[S + s];
        UnitLikResult lr = compute_unit_lik(s, b_s,
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial);
        temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det_bym2(temp_ws, a, c, data);
}

#endif // RATIOD_HMC_ICAR_COLLAPSED_H
