// hmc_re.h
// Random-effect log prior and effective effects, one templated implementation
// serving both the analytic log posterior and the autodiff one.
//
// Covers the whole surface the parameter layout allocates for: one or more
// crossed terms, each intercept-only or carrying slopes, correlated or not, in
// the centred or the non-centred parameterization. The effective effect the
// linear predictor sees is not the parameter itself under the non-centred and
// correlated parameterizations, so it is returned alongside the prior.
//
// Include AFTER hmc_sampler.h, which defines ModelData and ParamLayout.

#ifndef RATIOD_HMC_RE_H
#define RATIOD_HMC_RE_H

#include <cmath>
#include <vector>
#include "autodiff_utils.h"

namespace ratiod_re {

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using namespace ratiod::math;

// tanh via the logistic, so the autodiff types need no tanh primitive:
// tanh(x) = 2 * sigmoid(2x) - 1.
template <typename T>
inline T re_tanh(const T& x) {
    return T(2.0) * inv_logit(T(2.0) * x) - T(1.0);
}

// Effective random effects, laid out per term as [group * n_coefs + coef].
template <typename T>
using ReEffects = std::vector<std::vector<T> >;

// Number of coefficients this term carries (1 = intercept only).
inline int re_term_n_coefs(const ParamLayout& layout, int t) {
    return layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
}

// Index of the log_sigma parameter for coefficient c of term t.
inline int re_log_sigma_idx(const ParamLayout& layout, int n_terms, int t, int c) {
    if (layout.has_re_slopes) return layout.log_sigma_re_slopes[t][c];
    if (n_terms > 1) return layout.log_sigma_re_multi[t];
    return layout.log_sigma_re_idx;
}

// Log prior on every random-effect term, filling `re_eff` with the effects the
// linear predictor sees. Sets `ok` false when the Cholesky factor of a
// correlated term leaves no positive diagonal, which the caller reports as
// -infinity.
template <typename T>
T re_log_prior(const std::vector<T>& params,
               const ModelData& data,
               const ParamLayout& layout,
               ReEffects<T>& re_eff,
               bool& ok) {
    ok = true;
    T log_prior = T(0.0);
    if (!layout.has_re) {
        re_eff.clear();
        return log_prior;
    }

    const int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
    re_eff.assign(n_terms, std::vector<T>());

    for (int t = 0; t < n_terms; t++) {
        const int n_groups = (data.n_re_terms > 0) ? data.re_n_groups_multi[t]
                                                   : data.n_re_groups;
        const int n_coefs = re_term_n_coefs(layout, t);
        const bool correlated = layout.has_re_slopes
                                && layout.re_correlated_multi[t] && n_coefs > 1;

        // Half-Cauchy(0, scale) on each standard deviation, with the Jacobian of
        // the log transform.
        std::vector<T> sigmas(n_coefs);
        for (int c = 0; c < n_coefs; c++) {
            const T log_sigma = params[re_log_sigma_idx(layout, n_terms, t, c)];
            sigmas[c] = safe_exp(log_sigma);
            log_prior = log_prior + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);
        }

        // Correlated slopes: Sigma = diag(sigma) L L' diag(sigma), with L lower
        // triangular of unit row norm. Off-diagonals come through tanh so the
        // sampler sees an unconstrained space with no boundary to explode at.
        std::vector<T> L_flat;
        if (correlated) {
            const int chol_start = layout.chol_re_start_multi[t];
            L_flat.assign(static_cast<size_t>(n_coefs) * n_coefs, T(0.0));
            int chol_idx = 0;
            for (int r = 0; r < n_coefs; r++) {
                T row_sum_sq = T(0.0);
                for (int c = 0; c < r; c++) {
                    const T l_rc = re_tanh(params[chol_start + chol_idx]);
                    L_flat[r * n_coefs + c] = l_rc;
                    row_sum_sq = row_sum_sq + l_rc * l_rc;
                    // log|d(tanh)/d(raw)| = log(1 - tanh^2)
                    log_prior = log_prior
                        + safe_log(safe_max(T(1.0) - l_rc * l_rc, T(1e-300)));
                    chol_idx++;
                }
                const T diag_sq = T(1.0) - row_sum_sq;
                if (get_value(diag_sq) < 1e-10) {
                    ok = false;
                    return log_prior;
                }
                L_flat[r * n_coefs + r] = safe_sqrt(diag_sq);
            }

            // LKJ(eta) on the implied correlation matrix, then the Jacobian from
            // the Cholesky factor to that correlation matrix.
            const double eta_lkj = 2.0;
            for (int k = 0; k < n_coefs; k++) {
                log_prior = log_prior
                    + T((eta_lkj - 1.0 + (n_coefs - k - 1) / 2.0) * 2.0)
                    * safe_log(L_flat[k * n_coefs + k]);
            }
            for (int k = 1; k < n_coefs; k++) {
                log_prior = log_prior
                    + T(n_coefs - k) * safe_log(L_flat[k * n_coefs + k]);
            }
        }

        const int re_start = (n_terms > 1 || layout.has_re_slopes)
            ? layout.re_start_multi[t] : layout.re_start;
        re_eff[t].assign(static_cast<size_t>(n_groups) * n_coefs, T(0.0));

        if (correlated) {
            // Non-centred: params hold z ~ N(0, I) and re = diag(sigma) L z. The
            // |det(diag(sigma) L)| of the change of variables cancels the
            // |Sigma|^{-1/2} of the density exactly, so no determinant appears.
            for (int g = 0; g < n_groups; g++) {
                for (int c = 0; c < n_coefs; c++) {
                    T Lz = T(0.0);
                    for (int k = 0; k <= c; k++) {
                        Lz = Lz + L_flat[c * n_coefs + k]
                                * params[re_start + g * n_coefs + k];
                    }
                    re_eff[t][g * n_coefs + c] = sigmas[c] * Lz;
                    const T z_gc = params[re_start + g * n_coefs + c];
                    log_prior = log_prior - T(0.5) * z_gc * z_gc;
                }
            }
        } else if (data.re_parameterization == 1) {
            for (int g = 0; g < n_groups; g++) {
                for (int c = 0; c < n_coefs; c++) {
                    const T z_gc = params[re_start + g * n_coefs + c];
                    re_eff[t][g * n_coefs + c] = sigmas[c] * z_gc;
                    log_prior = log_prior - T(0.5) * z_gc * z_gc;
                }
            }
        } else {
            for (int g = 0; g < n_groups; g++) {
                for (int c = 0; c < n_coefs; c++) {
                    const T re_val = params[re_start + g * n_coefs + c];
                    re_eff[t][g * n_coefs + c] = re_val;
                    const T tau_re = T(1.0) / (sigmas[c] * sigmas[c] + T(1e-10));
                    log_prior = log_prior - T(0.5) * tau_re * re_val * re_val
                                          + T(0.5) * safe_log(tau_re);
                }
            }
            log_prior = log_prior
                - T(0.5 * n_groups * n_coefs * std::log(2.0 * M_PI));
        }
    }

    return log_prior;
}

// Random-effect contribution to one observation's linear predictor.
template <typename T>
T re_eta(int i, const ModelData& data, const ParamLayout& layout,
         const ReEffects<T>& re_eff) {
    if (re_eff.empty()) return T(0.0);
    const int n_terms = static_cast<int>(re_eff.size());
    T contrib = T(0.0);

    if (layout.has_re_slopes) {
        for (int t = 0; t < n_terms; t++) {
            const int group_idx = data.re_group_multi_flat[i * n_terms + t];
            if (group_idx <= 0) continue;
            const int g = group_idx - 1;
            const int n_coefs = layout.re_n_coefs_multi[t];
            contrib = contrib + re_eff[t][g * n_coefs];
            const int n_slopes = n_coefs - 1;
            if (n_slopes > 0 && !data.re_slope_matrices[t].empty()) {
                for (int s = 0; s < n_slopes; s++) {
                    contrib = contrib + re_eff[t][g * n_coefs + 1 + s]
                        * T(data.re_slope_matrices[t][i * n_slopes + s]);
                }
            }
        }
    } else if (n_terms > 1) {
        for (int t = 0; t < n_terms; t++) {
            const int group_idx = data.re_group_multi_flat[i * n_terms + t];
            if (group_idx > 0) contrib = contrib + re_eff[t][group_idx - 1];
        }
    } else if (!data.re_group.empty() && data.re_group[i] > 0) {
        contrib = contrib + re_eff[0][data.re_group[i] - 1];
    }

    return contrib;
}

}  // namespace ratiod_re

#endif  // RATIOD_HMC_RE_H
