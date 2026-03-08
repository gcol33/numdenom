// hmc_gp_collapsed.h
// Collapsed/marginalized GP spatial effects via inner Laplace optimization
//
// Instead of sampling N_gp spatial effects w alongside hyperparameters,
// we marginalize w out by finding w* = argmax_w [log p(y|w,theta) + log p_NNGP(w|sigma2,phi)]
// at each HMC gradient evaluation. This reduces dimensionality from
// N_gp + 2 (hyperparams) to just 2 hyperparams.
//
// The collapsed log-posterior is:
//   log p(theta|y) ~ log p(y|w*,theta) + log p_NNGP(w*|sigma2,phi) + log p(theta)
//                    - 0.5 * log det(W + Q)  [optional log-det correction]
//
// where W = diag(neg_hessian_data) and Q = NNGP precision matrix.
//
// The gradient dL/dtheta uses the implicit function theorem:
// at the mode w*, dL/dw = 0, so dw*/dtheta terms vanish and
// dL/dtheta = partial_L/partial_theta |_{w=w*}

#ifndef RATIOD_HMC_GP_COLLAPSED_H
#define RATIOD_HMC_GP_COLLAPSED_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppEigen.h>
// NOTE: This header must be included AFTER hmc_sampler.h (which defines ModelData/ModelType)
// and hmc_gp.h (which defines GPData in namespace ratiod_gp).
// It is included from hmc_sampler.cpp in the correct order.

using ratiod_gp::GPData;
using ratiod_svc::compute_cov;
using ratiod_hmc::ModelData;
using ratiod_hmc::ModelType;

// =========================================================================
// NNGP precision matrix operations (Q = (I-B)^T D^{-1} (I-B))
// =========================================================================

// Workspace for collapsed GP computations
struct CollapsedGPWorkspace {
    // NNGP structure coefficients
    std::vector<double> B_flat;     // NNGP regression coefficients (flattened)
    std::vector<int> n_nb;          // Number of neighbors per location
    std::vector<int> nb_idx_flat;   // Neighbor indices (flattened, 0-based)
    std::vector<double> d_cond;     // Conditional variances
    int nn;                         // Max neighbors
    int N_gp;                       // Number of unique locations

    // Transpose neighbor structure (for Q^T v efficiently)
    // For each location i, which locations j have i as their neighbor?
    std::vector<int> rev_ptr;       // CSR-style: rev_ptr[i] .. rev_ptr[i+1] are locations j
    std::vector<int> rev_col;       // The j values
    std::vector<int> rev_nb_pos;    // Position of i in j's neighbor list (for B lookup)

    // Newton workspace
    std::vector<double> w_star;     // Current mode estimate
    std::vector<double> grad_w;     // Gradient w.r.t. w
    std::vector<double> hess_diag;  // Diagonal of data neg-Hessian (W)
    std::vector<double> newton_rhs; // RHS for Newton solve
    std::vector<double> cg_r, cg_p, cg_Ap;  // CG workspace

    // Laplace correction
    double laplace_log_det = 0.0;  // -0.5 * log det(W + Q)

    bool structure_built = false;
    bool mode_found = false;

    void init(int n_gp, int max_nn) {
        // If N_gp changed (different model or different data), reset mode
        if (n_gp != N_gp) {
            mode_found = false;
        }
        N_gp = n_gp;
        nn = max_nn;
        B_flat.assign(n_gp * max_nn, 0.0);
        n_nb.assign(n_gp, 0);
        nb_idx_flat.assign(n_gp * max_nn, -1);
        d_cond.assign(n_gp, 1.0);
        w_star.assign(n_gp, 0.0);
        grad_w.assign(n_gp, 0.0);
        hess_diag.assign(n_gp, 0.0);
        newton_rhs.assign(n_gp, 0.0);
        cg_r.assign(n_gp, 0.0);
        cg_p.assign(n_gp, 0.0);
        cg_Ap.assign(n_gp, 0.0);
    }
};

// Build the NNGP coefficients B and conditional variances D for given (sigma2, phi)
// B[i,j] = c_i^T C_i^{-1} e_j  (regression of w_i on neighbors)
// d[i] = sigma2 - c_i^T C_i^{-1} c_i  (conditional variance)
inline void build_nngp_B_D(
    double sigma2, double phi,
    const GPData& gp_data,
    CollapsedGPWorkspace& ws
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    ws.init(N, nn);

    // Work arrays
    std::vector<double> c_vec(nn), C_mat(nn * nn), L(nn * nn);
    std::vector<double> alpha(nn);

    // First observation: marginal N(0, sigma2)
    int first_idx = gp_data.nn_order[0];
    ws.n_nb[first_idx] = 0;
    ws.d_cond[first_idx] = sigma2;

    for (int i = 1; i < N; i++) {
        int obs_idx = gp_data.nn_order[i];

        // Count neighbors
        int n_nb_i = 0;
        for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb_i++;
        ws.n_nb[obs_idx] = n_nb_i;

        if (n_nb_i == 0) {
            ws.d_cond[obs_idx] = sigma2;
            continue;
        }

        // Build c_vec (cross-covariance with neighbors)
        for (int j = 0; j < n_nb_i; j++) {
            double d = gp_data.nn_dist[i * nn + j];
            c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
        }

        // Build C_mat (neighbor-neighbor covariance) and get neighbor indices
        bool ok = true;
        for (int j1 = 0; j1 < n_nb_i && ok; j1++) {
            int raw1 = gp_data.nn_idx[i * nn + j1];
            if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
            int idx1 = gp_data.nn_order[raw1 - 1];
            ws.nb_idx_flat[obs_idx * nn + j1] = idx1;

            for (int j2 = 0; j2 < n_nb_i; j2++) {
                if (j1 == j2) {
                    C_mat[j1 * n_nb_i + j2] = sigma2;
                } else {
                    int raw2 = gp_data.nn_idx[i * nn + j2];
                    if (raw2 - 1 < 0 || raw2 - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
                    int idx2 = gp_data.nn_order[raw2 - 1];
                    double dx = gp_data.coords[idx1 * 2] - gp_data.coords[idx2 * 2];
                    double dy = gp_data.coords[idx1 * 2 + 1] - gp_data.coords[idx2 * 2 + 1];
                    C_mat[j1 * n_nb_i + j2] = compute_cov(std::sqrt(dx*dx + dy*dy),
                                                           sigma2, phi, gp_data.cov_type);
                }
            }
        }
        if (!ok) {
            ws.d_cond[obs_idx] = sigma2;
            continue;
        }

        // Cholesky: C = LL'
        std::fill(L.begin(), L.begin() + n_nb_i * n_nb_i, 0.0);
        for (int j = 0; j < n_nb_i; j++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) s += L[j * n_nb_i + k] * L[j * n_nb_i + k];
            double diag = C_mat[j * n_nb_i + j] - s;
            L[j * n_nb_i + j] = (diag > 0) ? std::sqrt(diag) : 1e-5;
            for (int k = j + 1; k < n_nb_i; k++) {
                double t = 0.0;
                for (int m = 0; m < j; m++) t += L[k * n_nb_i + m] * L[j * n_nb_i + m];
                L[k * n_nb_i + j] = (C_mat[k * n_nb_i + j] - t) / L[j * n_nb_i + j];
            }
        }

        // Solve L*y = c, L'*alpha = y => alpha = C^{-1}c
        for (int j = 0; j < n_nb_i; j++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) s += L[j * n_nb_i + k] * alpha[k];
            alpha[j] = (c_vec[j] - s) / L[j * n_nb_i + j];
        }
        for (int j = n_nb_i - 1; j >= 0; j--) {
            double s = 0.0;
            for (int k = j + 1; k < n_nb_i; k++) s += L[k * n_nb_i + j] * alpha[k];
            alpha[j] = (alpha[j] - s) / L[j * n_nb_i + j];
        }

        // Store B coefficients and conditional variance
        double c_alpha = 0.0;
        for (int j = 0; j < n_nb_i; j++) {
            ws.B_flat[obs_idx * nn + j] = alpha[j];
            c_alpha += c_vec[j] * alpha[j];
        }
        ws.d_cond[obs_idx] = std::max(sigma2 - c_alpha, 1e-10);
    }

    // Build reverse (transpose) neighbor structure
    // For Q^T v computation: need to know which locations j have i as neighbor
    std::vector<std::vector<std::pair<int,int>>> rev_lists(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < ws.n_nb[i]; j++) {
            int nb = ws.nb_idx_flat[i * nn + j];
            if (nb >= 0 && nb < N) {
                rev_lists[nb].push_back({i, j});
            }
        }
    }

    // Flatten into CSR format
    ws.rev_ptr.resize(N + 1, 0);
    int total_rev = 0;
    for (int i = 0; i < N; i++) {
        ws.rev_ptr[i] = total_rev;
        total_rev += (int)rev_lists[i].size();
    }
    ws.rev_ptr[N] = total_rev;
    ws.rev_col.resize(total_rev);
    ws.rev_nb_pos.resize(total_rev);
    for (int i = 0; i < N; i++) {
        int off = ws.rev_ptr[i];
        for (int k = 0; k < (int)rev_lists[i].size(); k++) {
            ws.rev_col[off + k] = rev_lists[i][k].first;
            ws.rev_nb_pos[off + k] = rev_lists[i][k].second;
        }
    }
    ws.structure_built = true;
}

// Compute Qv = (I-B)^T D^{-1} (I-B) v  (NNGP precision times vector)
// O(N * nn) per call
inline void nngp_precision_matvec(
    const double* v,
    double* result,
    const CollapsedGPWorkspace& ws
) {
    int N = ws.N_gp;
    int nn = ws.nn;

    // Step 1: r = (I-B) v   [forward pass]
    // (I-B) is lower triangular in NNGP ordering
    // r[i] = v[i] - sum_j B[i,j] * v[nb_j]
    std::vector<double> r(N);
    for (int i = 0; i < N; i++) {
        double ri = v[i];
        for (int j = 0; j < ws.n_nb[i]; j++) {
            int nb = ws.nb_idx_flat[i * nn + j];
            if (nb >= 0) ri -= ws.B_flat[i * nn + j] * v[nb];
        }
        r[i] = ri;
    }

    // Step 2: s = D^{-1} r
    std::vector<double> s(N);
    for (int i = 0; i < N; i++) {
        s[i] = r[i] / ws.d_cond[i];
    }

    // Step 3: result = (I-B)^T s   [backward pass using reverse structure]
    // result[i] = s[i] - sum_{j where i is neighbor of j} B[j, pos] * s[j]
    for (int i = 0; i < N; i++) {
        double res_i = s[i];
        for (int k = ws.rev_ptr[i]; k < ws.rev_ptr[i + 1]; k++) {
            int j = ws.rev_col[k];
            int pos = ws.rev_nb_pos[k];
            res_i -= ws.B_flat[j * nn + pos] * s[j];
        }
        result[i] = res_i;
    }
}

// Compute (W + Q) v  where W = diag(hess_diag) and Q = NNGP precision
inline void wplusq_matvec(
    const double* v,
    double* result,
    const double* hess_diag,
    const CollapsedGPWorkspace& ws
) {
    // Qv
    nngp_precision_matvec(v, result, ws);
    // Add Wv
    for (int i = 0; i < ws.N_gp; i++) {
        result[i] += hess_diag[i] * v[i];
    }
}

// Solve (W + Q) x = b using Conjugate Gradient
// Returns number of CG iterations
inline int cg_solve(
    double* x,           // Solution (in/out, initial guess)
    const double* b,     // RHS
    const double* hess_diag,
    CollapsedGPWorkspace& ws,
    int max_iter = 100,
    double tol = 1e-8
) {
    int N = ws.N_gp;

    // r = b - (W+Q)x
    wplusq_matvec(x, ws.cg_Ap.data(), hess_diag, ws);
    for (int i = 0; i < N; i++) ws.cg_r[i] = b[i] - ws.cg_Ap[i];

    // p = r
    std::memcpy(ws.cg_p.data(), ws.cg_r.data(), N * sizeof(double));

    double rr = 0.0;
    for (int i = 0; i < N; i++) rr += ws.cg_r[i] * ws.cg_r[i];

    double b_norm = 0.0;
    for (int i = 0; i < N; i++) b_norm += b[i] * b[i];
    if (b_norm < 1e-30) return 0;

    for (int iter = 0; iter < max_iter; iter++) {
        if (rr / b_norm < tol * tol) return iter;

        // Ap = (W+Q) p
        wplusq_matvec(ws.cg_p.data(), ws.cg_Ap.data(), hess_diag, ws);

        double pAp = 0.0;
        for (int i = 0; i < N; i++) pAp += ws.cg_p[i] * ws.cg_Ap[i];
        if (pAp < 1e-30) return iter;

        double alpha = rr / pAp;

        for (int i = 0; i < N; i++) {
            x[i] += alpha * ws.cg_p[i];
            ws.cg_r[i] -= alpha * ws.cg_Ap[i];
        }

        double rr_new = 0.0;
        for (int i = 0; i < N; i++) rr_new += ws.cg_r[i] * ws.cg_r[i];

        double beta = rr_new / rr;
        for (int i = 0; i < N; i++) {
            ws.cg_p[i] = ws.cg_r[i] + beta * ws.cg_p[i];
        }
        rr = rr_new;
    }
    return max_iter;
}

// =========================================================================
// Laplace log-det correction via sparse Cholesky
// =========================================================================

// Build W+Q as Eigen sparse matrix and compute log det via Cholesky.
// W = diag(hess_diag), Q = (I-B)^T D^{-1} (I-B)
// Returns -0.5 * log det(W + Q) (the Laplace correction term)
inline double compute_laplace_log_det(CollapsedGPWorkspace& ws) {
    int N = ws.N_gp;
    int nn = ws.nn;

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(N * (2 * nn + 1));

    // Build Q + W as sparse matrix via outer products of (I-B) rows
    for (int k = 0; k < N; k++) {
        double inv_dk = 1.0 / ws.d_cond[k];
        int n_nb_k = ws.n_nb[k];
        int n_entries = 1 + n_nb_k;
        int cols[32];
        double vals[32];
        cols[0] = k;
        vals[0] = 1.0;
        for (int j = 0; j < n_nb_k; j++) {
            cols[1 + j] = ws.nb_idx_flat[k * nn + j];
            vals[1 + j] = -ws.B_flat[k * nn + j];
        }
        for (int a = 0; a < n_entries; a++) {
            for (int b = a; b < n_entries; b++) {
                double v = inv_dk * vals[a] * vals[b];
                triplets.push_back(T(cols[a], cols[b], v));
                if (a != b) {
                    triplets.push_back(T(cols[b], cols[a], v));
                }
            }
        }
    }

    // Add W diagonal
    for (int i = 0; i < N; i++) {
        triplets.push_back(T(i, i, ws.hess_diag[i]));
    }

    // Build sparse matrix and compute Cholesky (all local, no thread_local Eigen)
    Eigen::SparseMatrix<double> WpQ(N, N);
    WpQ.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(WpQ);
    if (llt.info() != Eigen::Success) {
        // Add jitter and retry
        for (int i = 0; i < N; i++) {
            triplets.push_back(T(i, i, 1e-6));
        }
        WpQ.setFromTriplets(triplets.begin(), triplets.end());
        llt.compute(WpQ);
        if (llt.info() != Eigen::Success) {
            ws.laplace_log_det = 0.0;
            return 0.0;
        }
    }

    // log det(W+Q) = 2 * sum(log(diag(L)))
    Eigen::SparseMatrix<double> L_sparse = llt.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < N; i++) {
        log_det += std::log(L_sparse.coeff(i, i));
    }
    log_det *= 2.0;

    ws.laplace_log_det = -0.5 * log_det;
    return ws.laplace_log_det;
}

// =========================================================================
// Data likelihood helpers (per-location, aggregated over observations)
// =========================================================================

// Compute data log-likelihood gradient and negative Hessian w.r.t. w[loc]
// Aggregates over all observations at location loc
struct LocLikResult {
    double ll;          // log-likelihood contribution
    double grad;        // d(ll)/d(w[loc])
    double neg_hess;    // -d²(ll)/d(w[loc])²
};

inline LocLikResult compute_loc_lik(
    int loc,
    const double* w,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const ModelData& data,
    bool is_binomial
) {
    LocLikResult res = {0.0, 0.0, 0.0};
    int N = data.N;

    // DEBUG: per-obs NaN tracking
    static thread_local int loc_lik_debug = 0;

    for (int i = 0; i < N; i++) {
        if (data.gp_data.obs_to_loc[i] != loc) continue;

        // Compute eta
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.p_num; p++)
            eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.p_denom; p++)
                eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
        }

        // RE effects are in the outer HMC params, not in the collapsed GP
        // They are added to eta via the outer gradient function

        eta_num_i += w[loc];
        if (!is_binomial && data.gp_data.shared) eta_denom_i += w[loc];

        // Per-family likelihood, gradient, Hessian
        double mu_num = std::exp(eta_num_i);
        int y_num = data.y_num[i];

        // DEBUG: print first obs details if NaN
        if (loc_lik_debug < 3 && N >= 100) {
            double y_dc = (data.y_denom_cont.size() > (size_t)i) ? data.y_denom_cont[i] : -999.0;
            int y_d = (data.y_denom.size() > (size_t)i) ? data.y_denom[i] : -999;
            Rprintf("[LOC_LIK] i=%d loc=%d eta_num=%.4f eta_den=%.4f mu_num=%.4f y_num=%d "
                    "y_denom=%d y_denom_cont=%.6f phi_num=%.4f phi_den=%.4f shared=%d model=%d\n",
                    i, loc, eta_num_i, eta_denom_i, mu_num, y_num,
                    y_d, y_dc, phi_num, phi_denom,
                    (int)data.gp_data.shared, (int)data.model_type);
            // Print beta and X values
            Rprintf("  p_num=%d p_denom=%d X_flat_size=%d w[%d]=%.6f\n",
                    data.p_num, data.p_denom, (int)data.X_num_flat.size(), loc, w[loc]);
            // Recompute eta from scratch
            double eta_check = 0.0;
            for (int p = 0; p < std::min(data.p_num, 4); p++) {
                int idx = i * data.p_num + p;
                double x_val = (idx < (int)data.X_num_flat.size()) ? data.X_num_flat[idx] : -9999.0;
                eta_check += x_val * beta_num[p];
                Rprintf("  beta_num[%d]=%.6f X_num[%d,%d]=%.6f partial_eta=%.6f\n",
                        p, beta_num[p], i, p, x_val, eta_check);
            }
            eta_check += w[loc];
            Rprintf("  eta_check=%.6f (should match eta_num=%.6f)\n", eta_check, eta_num_i);
            loc_lik_debug++;
        }

        switch (data.model_type) {
            case ModelType::POISSON_GAMMA: {
                // Numerator: Poisson
                res.ll += y_num * eta_num_i - mu_num;
                double resid_num = y_num - mu_num;
                res.grad += resid_num;
                res.neg_hess += mu_num;

                // Denominator: Gamma (log link, continuous y_denom)
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    double y_denom = data.y_denom_cont[i];
                    double shape = phi_denom;
                    double resid_denom = shape * (y_denom / mu_denom - 1.0);
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(y_denom)
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += resid_denom;
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                // Numerator: NegBin
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));

                // Denominator: NegBin (shared GP)
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
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
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                res.ll += y_num * eta_num_i - n_trials * std::log(1.0 + std::exp(eta_num_i));
                res.grad += y_num - n_trials * p_i;
                res.neg_hess += n_trials * p_i * (1.0 - p_i);
                break;
            }
            default:
                // Fallback: Poisson-like
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                break;
        }
    }
    return res;
}

// Compute Laplace log-det for given params by finding mode and computing log det(W+Q).
// Used for numerical differentiation of the Laplace correction w.r.t. all params.
// warm_w: initial guess for Newton (warm start from nearby params)
// If base_ws is non-null and rebuild_nngp is false, reuses existing NNGP structure
// (valid when only non-GP params like beta/phi change, not sigma2/phi_gp).
inline double laplace_log_det_full(
    const double* beta_num, const double* beta_denom,
    double sigma2, double phi,
    double phi_num, double phi_denom,
    const ModelData& data,
    const std::vector<double>& warm_w,  // warm start for Newton
    const CollapsedGPWorkspace* base_ws = nullptr,  // existing NNGP to reuse
    bool rebuild_nngp = true
) {
    CollapsedGPWorkspace temp_ws;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    if (!rebuild_nngp && base_ws != nullptr) {
        // Reuse existing NNGP structure (B, D, reverse index)
        temp_ws.N_gp = base_ws->N_gp;
        temp_ws.nn = base_ws->nn;
        temp_ws.B_flat = base_ws->B_flat;
        temp_ws.n_nb = base_ws->n_nb;
        temp_ws.nb_idx_flat = base_ws->nb_idx_flat;
        temp_ws.d_cond = base_ws->d_cond;
        temp_ws.rev_ptr = base_ws->rev_ptr;
        temp_ws.rev_col = base_ws->rev_col;
        temp_ws.rev_nb_pos = base_ws->rev_nb_pos;
        temp_ws.structure_built = true;
        // Allocate Newton workspace
        int n = temp_ws.N_gp;
        temp_ws.w_star.resize(n);
        temp_ws.grad_w.resize(n);
        temp_ws.hess_diag.resize(n);
        temp_ws.newton_rhs.resize(n);
        temp_ws.cg_r.resize(n);
        temp_ws.cg_p.resize(n);
        temp_ws.cg_Ap.resize(n);
    } else {
        build_nngp_B_D(sigma2, phi, data.gp_data, temp_ws);
    }

    // Warm-start from provided w
    temp_ws.w_star = warm_w;
    temp_ws.mode_found = true;

    // Run Newton (should converge in 1-2 iterations from warm start)
    int N_gp = temp_ws.N_gp;
    std::vector<double> Qw(N_gp);
    std::vector<double> delta_w(N_gp, 0.0);

    for (int newton_iter = 0; newton_iter < 5; newton_iter++) {
        std::fill(temp_ws.grad_w.begin(), temp_ws.grad_w.end(), 0.0);
        std::fill(temp_ws.hess_diag.begin(), temp_ws.hess_diag.end(), 0.0);

        for (int loc = 0; loc < N_gp; loc++) {
            LocLikResult lr = compute_loc_lik(loc, temp_ws.w_star.data(),
                                               beta_num, beta_denom,
                                               phi_num, phi_denom, data, is_binomial);
            temp_ws.grad_w[loc] = lr.grad;
            temp_ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);
        }

        nngp_precision_matvec(temp_ws.w_star.data(), Qw.data(), temp_ws);
        double grad_norm = 0.0;
        for (int i = 0; i < N_gp; i++) {
            temp_ws.grad_w[i] -= Qw[i];
            grad_norm += temp_ws.grad_w[i] * temp_ws.grad_w[i];
        }
        if (std::sqrt(grad_norm) < 1e-6) break;

        std::fill(delta_w.begin(), delta_w.end(), 0.0);
        cg_solve(delta_w.data(), temp_ws.grad_w.data(), temp_ws.hess_diag.data(), temp_ws, 50, 1e-8);
        for (int i = 0; i < N_gp; i++) temp_ws.w_star[i] += delta_w[i];
    }

    // Final hess_diag at the new mode
    std::fill(temp_ws.hess_diag.begin(), temp_ws.hess_diag.end(), 0.0);
    for (int loc = 0; loc < N_gp; loc++) {
        LocLikResult lr = compute_loc_lik(loc, temp_ws.w_star.data(),
                                           beta_num, beta_denom,
                                           phi_num, phi_denom, data, is_binomial);
        temp_ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det(temp_ws);
}

// =========================================================================
// Newton-Raphson for finding w* (inner Laplace optimization)
// =========================================================================

// Find w* = argmax_w [ log p(y|beta,w) + log p_NNGP(w|sigma2,phi) ]
// using Newton-Raphson with CG inner solve
//
// Returns the log-posterior at w* (data + prior, no hyperparameter priors)
inline double collapsed_gp_find_mode(
    const double* beta_num, const double* beta_denom,
    double sigma2, double phi,
    double phi_num, double phi_denom,
    const ModelData& data,
    CollapsedGPWorkspace& ws,
    int max_newton = 20,
    double newton_tol = 1e-6
) {
    int N_gp = data.gp_data.n_obs;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    // Build NNGP structure for current (sigma2, phi)
    build_nngp_B_D(sigma2, phi, data.gp_data, ws);

    // Initialize w to previous mode or zero
    if (!ws.mode_found) {
        std::fill(ws.w_star.begin(), ws.w_star.end(), 0.0);
    } else {
        // Warm-start from previous w_star, but guard against NaN
        bool has_nan = false;
        for (int i = 0; i < N_gp; i++) {
            if (std::isnan(ws.w_star[i]) || std::isinf(ws.w_star[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            std::fill(ws.w_star.begin(), ws.w_star.end(), 0.0);
        }
    }

    std::vector<double> Qw(N_gp);
    std::vector<double> delta_w(N_gp, 0.0);

    // DEBUG: Newton iteration tracking
    static thread_local int newton_debug_count = 0;
    bool newton_debug = (newton_debug_count < 3 && N_gp >= 20);

    for (int newton_iter = 0; newton_iter < max_newton; newton_iter++) {
        // Compute data gradient and Hessian per location
        std::fill(ws.grad_w.begin(), ws.grad_w.end(), 0.0);
        std::fill(ws.hess_diag.begin(), ws.hess_diag.end(), 0.0);

        double data_ll = 0.0;
        int obs_count = 0;
        for (int loc = 0; loc < N_gp; loc++) {
            LocLikResult lr = compute_loc_lik(loc, ws.w_star.data(),
                                               beta_num, beta_denom,
                                               phi_num, phi_denom,
                                               data, is_binomial);
            ws.grad_w[loc] = lr.grad;
            ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);  // Ensure positive
            data_ll += lr.ll;
            // Count obs matching this location
            if (newton_debug && newton_iter == 0) {
                int cnt = 0;
                for (int i = 0; i < data.N; i++)
                    if (data.gp_data.obs_to_loc[i] == loc) cnt++;
                obs_count += cnt;
                if (std::isnan(lr.ll) || std::isnan(lr.grad) || cnt == 0) {
                    Rprintf("  [NEWTON] loc=%d cnt=%d ll=%.4f grad=%.4f hess=%.8f\n",
                            loc, cnt, lr.ll, lr.grad, lr.neg_hess);
                }
            }
        }
        if (newton_debug && newton_iter == 0) {
            // Print obs_to_loc range
            int otl_min = data.N, otl_max = -1;
            for (int i = 0; i < data.N; i++) {
                otl_min = std::min(otl_min, data.gp_data.obs_to_loc[i]);
                otl_max = std::max(otl_max, data.gp_data.obs_to_loc[i]);
            }
            Rprintf("  obs_to_loc range=[%d,%d] N_gp=%d\n", otl_min, otl_max, N_gp);
        }

        // Add NNGP prior gradient: d/dw log p_NNGP(w|sigma2,phi) = -Q w
        nngp_precision_matvec(ws.w_star.data(), Qw.data(), ws);
        for (int i = 0; i < N_gp; i++) {
            ws.grad_w[i] -= Qw[i];  // gradient = data_grad - Q*w
        }

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < N_gp; i++) grad_norm += ws.grad_w[i] * ws.grad_w[i];
        grad_norm = std::sqrt(grad_norm);

        if (newton_debug) {
            double w_min = 1e30, w_max = -1e30;
            double dw_min = 1e30, dw_max = -1e30;
            for (int i = 0; i < N_gp; i++) {
                w_min = std::min(w_min, ws.w_star[i]);
                w_max = std::max(w_max, ws.w_star[i]);
            }
            Rprintf("[NEWTON iter=%d] grad_norm=%.6e data_ll=%.4f w=[%.4f,%.4f]",
                    newton_iter, grad_norm, data_ll, w_min, w_max);
            if (newton_iter == 0)
                Rprintf(" obs_count=%d N_gp=%d N=%d", obs_count, N_gp, data.N);
            Rprintf("\n");
        }

        if (grad_norm < newton_tol) {
            break;
        }

        // Solve (W + Q) delta_w = grad_w using CG
        std::fill(delta_w.begin(), delta_w.end(), 0.0);
        int cg_iters = cg_solve(delta_w.data(), ws.grad_w.data(), ws.hess_diag.data(), ws, 100, 1e-8);

        if (newton_debug) {
            double dw_min = 1e30, dw_max = -1e30;
            for (int i = 0; i < N_gp; i++) {
                dw_min = std::min(dw_min, delta_w[i]);
                dw_max = std::max(dw_max, delta_w[i]);
            }
            Rprintf("  CG iters=%d delta_w=[%.4f,%.4f]\n", cg_iters, dw_min, dw_max);
        }

        // Newton update: w <- w + delta_w
        for (int i = 0; i < N_gp; i++) {
            ws.w_star[i] += delta_w[i];
        }
    }
    if (newton_debug) newton_debug_count++;

    ws.mode_found = true;

    // Compute log-posterior at w*
    double data_ll = 0.0;
    for (int loc = 0; loc < N_gp; loc++) {
        LocLikResult lr = compute_loc_lik(loc, ws.w_star.data(),
                                           beta_num, beta_denom,
                                           phi_num, phi_denom,
                                           data, is_binomial);
        data_ll += lr.ll;
    }

    // NNGP prior: -0.5 * w^T Q w - 0.5 * log det(Q) + 0.5 * N * log(2pi)
    // For the collapsed log-post we need: data_ll + nngp_prior
    nngp_precision_matvec(ws.w_star.data(), Qw.data(), ws);
    double wQw = 0.0;
    for (int i = 0; i < N_gp; i++) wQw += ws.w_star[i] * Qw[i];

    // log det(Q) = -2 * sum log(d_i) (from Q = (I-B)^T D^{-1} (I-B), det Q = 1/prod(d_i))
    double log_det_Q = 0.0;
    for (int i = 0; i < N_gp; i++) log_det_Q -= std::log(ws.d_cond[i]);

    double nngp_prior = -0.5 * wQw + 0.5 * log_det_Q;

    // Laplace correction: -0.5 * log det(W + Q) via sparse Cholesky
    compute_laplace_log_det(ws);

    return data_ll + nngp_prior;
}

// =========================================================================
// Collapsed GP gradient computation
// =========================================================================

// Compute residuals (dL/deta) at w* for scattering to beta gradients
inline void collapsed_gp_compute_residuals(
    const double* w_star,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const ModelData& data,
    double* resid_num,  // length N (observations, not locations)
    double* resid_denom // length N
) {
    int N = data.N;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    for (int i = 0; i < N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];

        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.p_num; p++)
            eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.p_denom; p++)
                eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
        }

        eta_num_i += w_star[loc_i];
        if (!is_binomial && data.gp_data.shared) eta_denom_i += w_star[loc_i];

        double mu_num = std::exp(eta_num_i);

        switch (data.model_type) {
            case ModelType::POISSON_GAMMA: {
                resid_num[i] = data.y_num[i] - mu_num;
                if (!is_binomial && data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    resid_denom[i] = phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0);
                } else {
                    resid_denom[i] = 0.0;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                resid_num[i] = data.y_num[i] - mu_num * (data.y_num[i] + r_num) / (mu_num + r_num);
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    int y_denom = (int)data.y_denom[i];
                    double r_denom = phi_denom;
                    resid_denom[i] = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                } else {
                    resid_denom[i] = 0.0;
                }
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

// end collapsed GP functions

#endif // RATIOD_HMC_GP_COLLAPSED_H
