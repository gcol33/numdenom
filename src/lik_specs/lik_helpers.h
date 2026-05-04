// lik_helpers.h
// Templated numerical helpers shared across tulpaRatio LikelihoodSpec
// implementations. Single source of truth for the link inverses, log-link-prob
// helpers, and integer-response payload accessor.
//
// All functions are `static inline` so each TU that includes this header sees
// its own copy with no ODR risk; the compiler inlines them at use sites.
//
// Templated on T so the same code path works for double (evaluation),
// tulpa::arena::Var (reverse AD), and fwd::Dual (forward AD).

#ifndef TULPARATIO_LIK_HELPERS_H
#define TULPARATIO_LIK_HELPERS_H

#include <cmath>
#include <vector>
#include <tulpa/autodiff_arena.h>
#include <tulpa/autodiff_fwd.h>
#include <tulpa/portable_math.h>

namespace tulpaRatio {
namespace lik {

// ----- safe_log1pexp ---------------------------------------------------------
// log(1 + exp(x)) without overflow; mirrors tulpaOcc::safe_log1pexp.
// ADL on std::exp / std::log1p covers double; tulpa::arena::Var and fwd::Dual
// supply their own exp / log1p in their respective namespaces.
template <typename T>
static inline T safe_log1pexp(const T& x) {
    using std::exp;
    using std::log1p;
    if (x > 35.0) return x;
    if (x < -10.0) return exp(x);
    return log1p(exp(x));
}

// ----- log_inv_logit / log1m_inv_logit --------------------------------------
// Numerically stable forms of log(sigmoid(x)) and log(1 - sigmoid(x)).
//   log(sigmoid(x))   = x - log(1 + exp(x))
//   log(1 - sigmoid(x)) = -log(1 + exp(x))
template <typename T>
static inline T log_inv_logit(const T& x) {
    return x - safe_log1pexp(x);
}

template <typename T>
static inline T log1m_inv_logit(const T& x) {
    return T(0.0) - safe_log1pexp(x);
}

// ----- BinomialResponseData --------------------------------------------------
// Payload that the R-side bridge attaches to ModelData::model_response_data
// for binomial / beta_binomial families. Holds the success counts and trial
// counts as plain int vectors so the templated likelihood reads them directly.
struct BinomialResponseData {
    std::vector<int> y;  // numerator successes, length N
    std::vector<int> n;  // trial counts (denominator), length N
};

// ----- log P(Y = y | n, p) under the logit link -----------------------------
// Returns the binomial log-pmf, evaluated through log_inv_logit/log1m_inv_logit
// so the eta -> log p transformation stays numerically stable. lchoose is the
// y- and n-dependent normalizing constant; it is a parameter-independent term
// but kept for absolute log-likelihood parity with the legacy implementation.
template <typename T>
static inline T binom_logit_log_pmf(int y, int n, const T& eta) {
    T log_p   = log_inv_logit(eta);
    T log_1mp = log1m_inv_logit(eta);
    T lchoose = T(tulpa::math::portable_lchoose(n, y));
    return T(double(y)) * log_p + T(double(n - y)) * log_1mp + lchoose;
}

}  // namespace lik
}  // namespace tulpaRatio

#endif  // TULPARATIO_LIK_HELPERS_H
