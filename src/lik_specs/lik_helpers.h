// lik_helpers.h
// Templated numerical helpers and per-family response payloads shared across
// tulpaRatio LikelihoodSpec implementations. Single source of truth for the
// link inverses, log-pmf / log-pdf kernels, and integer/continuous response
// data structs.
//
// All functions are `static inline` so each TU that includes this header sees
// its own copy with no ODR risk; the compiler inlines them at use sites.
//
// Templated on T so the same code path works for double (evaluation),
// tulpa::arena::Var (reverse AD), and fwd::Dual (forward AD).
//
// Numerical conventions
// ---------------------
// * exp / log / log1p / lgamma resolve via ADL on std:: for double, and via
//   the corresponding overload in `tulpa::arena` / `fwd` namespaces for the
//   AD types (provided by tulpa's autodiff_arena.h / autodiff_fwd.h).
// * Constant terms that depend only on integer counts (lgamma(y+1), lchoose,
//   log(y) for continuous y) are kept in the log-pmf for absolute parity with
//   the legacy log_post_impl.h code; the autodiff path treats them as zeros
//   in the derivative.

#ifndef TULPARATIO_LIK_HELPERS_H
#define TULPARATIO_LIK_HELPERS_H

#include <cmath>
#include <vector>
#include <tulpa/autodiff_arena.h>
#include <tulpa/autodiff_fwd.h>
#include <tulpa/portable_math.h>

namespace tulpaRatio {
namespace lik {

// ============================================================================
// Numerically stable scalar helpers
// ============================================================================

// log(1 + exp(x)) without overflow. Mirrors tulpaOcc::safe_log1pexp.
template <typename T>
static inline T safe_log1pexp(const T& x) {
    using std::exp;
    using std::log1p;
    if (x > 35.0) return x;
    if (x < -10.0) return exp(x);
    return log1p(exp(x));
}

// log(sigmoid(x)) and log(1 - sigmoid(x)).
template <typename T>
static inline T log_inv_logit(const T& x) {
    return x - safe_log1pexp(x);
}
template <typename T>
static inline T log1m_inv_logit(const T& x) {
    return T(0.0) - safe_log1pexp(x);
}

// ----- log_lgamma ---------------------------------------------------------
// Wrapper around lgamma that resolves to std::lgamma for double and the
// AD-namespace lgamma for tulpa::arena::Var / fwd::Dual via ADL.
template <typename T>
static inline T log_lgamma(const T& x) {
    using std::lgamma;
    return lgamma(x);
}

// ============================================================================
// Per-family response payload structs (Pattern 2: per-family POD)
// ----------------------------------------------------------------------------
// One struct per family, each holding only the fields that family needs. The
// bridge (tulpa_bridge.cpp) constructs the right struct based on cfg.family,
// stashes a pointer in ModelData::model_response_data, and the templated
// likelihood callback reinterprets it. No tagged union, no junk fields.
//
// Continuous responses keep their own double vectors so families that mix
// integer / continuous (poisson_gamma, negbin_gamma) read each field cleanly.
// ============================================================================

struct BinomialResponseData {
    std::vector<int> y;  // numerator successes, length N
    std::vector<int> n;  // trial counts (denominator), length N
};

struct BetaBinomialResponseData {
    std::vector<int> y;
    std::vector<int> n;
};

struct PoissonGammaResponseData {
    std::vector<int>    y_num;        // count numerator
    std::vector<double> y_denom_cont; // continuous denominator (Gamma response)
};

struct NegbinGammaResponseData {
    std::vector<int>    y_num;        // count numerator (NegBin)
    std::vector<double> y_denom_cont; // continuous denominator (Gamma)
};

struct NegbinNegbinResponseData {
    std::vector<int> y_num;
    std::vector<int> y_denom;
};

struct GammaGammaResponseData {
    std::vector<double> y_num_cont;
    std::vector<double> y_denom_cont;
};

struct LognormalResponseData {
    std::vector<double> y_num_cont;
    std::vector<double> y_denom_cont;
};

// ============================================================================
// Log-pmf / log-pdf kernels shared across families
// ----------------------------------------------------------------------------
// Each family file calls these so the per-observation algebra appears in only
// one place. The mean parameterizations match the legacy autodiff_utils.h
// kernels; we restate them as comments for the math header inside each lik_*
// builder.
// ============================================================================

// ----- Binomial logit ------------------------------------------------------
// log P(y | n, p), p = sigmoid(eta).  Used for binomial.
template <typename T>
static inline T binom_logit_log_pmf(int y, int n, const T& eta) {
    T log_p   = log_inv_logit(eta);
    T log_1mp = log1m_inv_logit(eta);
    T lchoose = T(tulpa::math::portable_lchoose(n, y));
    return T(double(y)) * log_p + T(double(n - y)) * log_1mp + lchoose;
}

// ----- Beta-Binomial -------------------------------------------------------
// y ~ BetaBinomial(n, alpha, beta) with mean-precision parameterization:
//   p = sigmoid(eta), phi = precision.  alpha = p*phi, beta = (1-p)*phi.
// log P(y) = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+alpha+beta)
//           - lgamma(alpha) - lgamma(beta) + lgamma(alpha+beta)
//           + lchoose(n, y).
template <typename T>
static inline T beta_binomial_log_pmf(int y, int n, const T& eta, const T& phi) {
    using std::exp;
    // Use stable inv_logit via log_inv_logit for consistency, then exp once.
    T log_p   = log_inv_logit(eta);
    T log_1mp = log1m_inv_logit(eta);
    T p       = exp(log_p);
    T one_mp  = exp(log_1mp);

    T alpha = p      * phi;
    T beta  = one_mp * phi;

    T ll = log_lgamma(T(double(y)) + alpha)
         + log_lgamma(T(double(n - y)) + beta)
         - log_lgamma(T(double(n)) + alpha + beta);
    ll = ll - log_lgamma(alpha) - log_lgamma(beta) + log_lgamma(alpha + beta);
    ll = ll + T(tulpa::math::portable_lchoose(n, y));
    return ll;
}

// ----- Poisson with offset --------------------------------------------------
// Per-observation Poisson log-pmf with mean parameterization mu = exposure *
// exp(eta). Constant lgamma(y+1) kept for absolute parity.
//   log P(y | mu) = y*log(mu) - mu - lgamma(y+1).
// In poisson_gamma the legacy code parameterizes the *numerator* as
//   mu_num = exp(eta_num)        (no exposure multiplied through eta),
// so we expose two flavours: with-exposure and without.
template <typename T>
static inline T poisson_log_pmf(int y, const T& mu) {
    using std::log;
    T y_T = T(double(y));
    return y_T * log(mu) - mu - T(std::lgamma(double(y) + 1.0));
}

// ----- NegBin (mean / overdispersion) --------------------------------------
// y ~ NegBin(mu, phi) via Gamma-Poisson mixture:
//   log P(y) = lgamma(y + phi) - lgamma(y+1) - lgamma(phi)
//            + phi*log(phi/(mu+phi)) + y*log(mu/(mu+phi)).
template <typename T>
static inline T negbin_log_pmf(int y, const T& mu, const T& phi) {
    using std::log;
    T y_T = T(double(y));
    T ll  = log_lgamma(y_T + phi) - T(std::lgamma(double(y) + 1.0)) - log_lgamma(phi);
    T mp  = mu + phi;
    ll = ll + phi * log(phi / mp);
    ll = ll + y_T * log(mu / mp);
    return ll;
}

// ----- Gamma (shape / mean) ------------------------------------------------
// y ~ Gamma(shape, rate=shape/mu) (mean parameterization).
//   log p(y) = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape).
// log(y) is constant w.r.t. parameters but kept for absolute parity.
template <typename T>
static inline T gamma_log_pdf(double y, const T& shape, const T& mu) {
    using std::log;
    T rate = shape / mu;
    return shape * log(rate)
         + (shape - T(1.0)) * T(std::log(y))
         - rate * y
         - log_lgamma(shape);
}

// ----- Lognormal -----------------------------------------------------------
// log(y) ~ Normal(mu_log, sigma).  mu_log is the linear predictor on the
// log scale, sigma is the std-dev (always positive — passed in as exp(log_sig)).
//   log p(y) = -log(y) - log(sigma) - 0.5*((log(y)-mu)/sigma)^2  (drops the
//             constant -0.5*log(2*pi) for parity with the legacy kernel).
template <typename T>
static inline T lognormal_log_pdf(double y, const T& mu, const T& sigma) {
    using std::log;
    double log_y = std::log(y);
    T z = (T(log_y) - mu) / sigma;
    return T(-log_y) - log(sigma) - T(0.5) * z * z;
}

// ============================================================================
// Priors on extra parameters (added inside the per-obs likelihood at i==0)
// ----------------------------------------------------------------------------
// The B1b dispatcher does NOT register an `extra_prior` callback because that
// forces tulpa's gradient dispatcher onto numerical gradients (see
// hmc_gradient_dispatch.h:36). Instead, the per-obs templated likelihood adds
// the prior contribution exactly once at i==0 — autodiff differentiates
// through it cleanly. Single source of truth shared across every family that
// has extra parameters.
// ============================================================================

// Gamma prior on a phi parameter, with the parameterization used in
// log_post_impl.h:
//   p(log_phi)  ∝  (shape-1)*log_phi - rate*phi + log_phi   (Jacobian-adjusted)
template <typename T>
static inline T log_prior_gamma_log(const T& log_phi, double shape, double rate) {
    using std::exp;
    T phi = exp(log_phi);
    return T(shape - 1.0) * log_phi - T(rate) * phi + log_phi;
}

// Half-Cauchy prior on a sigma parameter on log-scale: matches
// log_prior_half_cauchy in autodiff_utils.h:
//   p(log_sigma) ∝ -log(1 + (sigma/scale)^2) + log_sigma
template <typename T>
static inline T log_prior_half_cauchy_log(const T& log_sigma, double scale) {
    using std::exp;
    using std::log;
    T sigma = exp(log_sigma);
    T ratio = sigma / T(scale);
    return T(0.0) - log(T(1.0) + ratio * ratio) + log_sigma;
}

}  // namespace lik
}  // namespace tulpaRatio

#endif  // TULPARATIO_LIK_HELPERS_H
