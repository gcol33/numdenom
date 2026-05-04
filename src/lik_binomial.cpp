// lik_binomial.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Per-observation model
// ---------------------
//   y_i | n_i ~ Binomial(n_i, p_i),         i = 1, ..., N
//   eta_i = x_i^T beta + offset_i           (offset_i = 0 in B1a)
//   p_i   = sigmoid(eta_i) = 1 / (1 + exp(-eta_i))   (logit link)
//
//   y_i = success count (numerator), integer in [0, n_i]
//   n_i = trial count (denominator), fixed integer not modelled
//
// Per-observation log-likelihood
// ------------------------------
//   log L_i(beta) = y_i * log(p_i) + (n_i - y_i) * log(1 - p_i) + lchoose(n_i, y_i)
//
// Numerically stable evaluation (used here):
//   log(p_i)     = eta_i - softplus(eta_i)   where softplus(x) = log(1 + exp(x))
//   log(1 - p_i) =       - softplus(eta_i)
//
// Closed-form per-obs gradient (recorded for B2; AD provides it in B1a):
//   d log L_i / d eta_i = y_i - n_i * p_i
//
// ZI / OI: ignored in B1a. Plain binomial registers only `ll_double`,
// `ll_arena`, `ll_fwd`. No `FullGradFn`, no `ResidualFn`, no `EtaWeightsFn`.
// =====================================================================

#include <type_traits>
#include <tulpa/likelihood.h>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>
#include <tulpa/autodiff_arena.h>
#include <tulpa/autodiff_fwd.h>

#include "lik_specs/ratio_config.h"
#include "lik_specs/lik_helpers.h"

namespace tulpaRatio {

// ============================================================================
// Templated per-observation log-likelihood for plain (non-ZI) binomial.
// Signature matches tulpa::LikelihoodFn<T>.
// ============================================================================
template <typename T>
static T binomial_log_likelihood(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data
) {
    // Anchor the arena so any temporaries created below allocate against the
    // arena that produced eta[0]. Mirrors the convention used in tulpaOcc.
    if constexpr (std::is_same_v<T, tulpa::arena::Var>) {
        tulpa::arena::current_arena() = eta[0].arena_;
    }

    const auto* resp = static_cast<const lik::BinomialResponseData*>(model_data);
    const int y_i = resp->y[i];
    const int n_i = resp->n[i];

    return lik::binom_logit_log_pmf(y_i, n_i, eta[0]);
}

// ============================================================================
// Public builder: assemble a LikelihoodSpec for the binomial-ratio family.
// Currently only the logit link is supported; new links should branch here.
// ============================================================================
tulpa::LikelihoodSpec build_binomial_spec(const RatioConfig& cfg) {
    tulpa::LikelihoodSpec spec;
    spec.name         = "tulpaRatio_binomial";
    spec.n_processes  = 1;
    spec.n_extra_params = 0;

    if (cfg.num_link != "logit") {
        // B1a registers only the logit link. Other links are deferred.
        spec.name = "tulpaRatio_binomial_unsupported_link";
        return spec;  // dispatcher will raise — see lik_dispatch.cpp
    }

    spec.ll_double = binomial_log_likelihood<double>;
    spec.ll_arena  = binomial_log_likelihood<tulpa::arena::Var>;
    spec.ll_fwd    = binomial_log_likelihood<::fwd::Dual>;

    // No FullGradFn, ResidualFn, EtaWeightsFn, extend_layout, or extra_prior.
    // Autodiff path only — A_r / A / N all work.
    return spec;
}

}  // namespace tulpaRatio
