// autodiff_utils.h
// Templated math functions that work with both double and ad::Var
// Enables single implementation of compute_log_post for both evaluation and gradient

#ifndef RATIOD_AUTODIFF_UTILS_H
#define RATIOD_AUTODIFF_UTILS_H

#include <cmath>
#include <vector>
#include <Rcpp.h>
#include "autodiff.h"

namespace ratiod {
namespace math {

// ============================================================================
// Type traits to detect ad::Var
// ============================================================================

template<typename T>
struct is_ad_var : std::false_type {};

template<>
struct is_ad_var<ad::Var> : std::true_type {};

// ============================================================================
// Basic math functions - dispatch to std:: or ad::
// ============================================================================

// exp
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
safe_exp(T x) {
    constexpr double EXP_MAX = 700.0;
    constexpr double EXP_MIN = -700.0;
    if (x > EXP_MAX) x = EXP_MAX;
    if (x < EXP_MIN) x = EXP_MIN;
    return std::exp(x);
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_exp(const T& x) {
    // ad::exp already handles the tape recording
    return ad::exp(x);
}

// log
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
safe_log(T x) {
    if (x <= 0.0) return -1e10;
    return std::log(x);
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_log(const T& x) {
    return ad::log(x);
}

// sqrt
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
safe_sqrt(T x) {
    if (x < 0.0) return 0.0;
    return std::sqrt(x);
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
safe_sqrt(const T& x) {
    return ad::sqrt(x);
}

// lgamma (log of gamma function)
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
lgamma_fn(T x) {
    return R::lgammafn(x);
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
lgamma_fn(const T& x) {
    return ad::lgamma(x);
}

// digamma (derivative of lgamma)
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
digamma_fn(T x) {
    return R::digamma(x);
}

// For ad::Var, digamma is handled internally by lgamma's backward pass

// inv_logit (logistic function)
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
inv_logit(T x) {
    if (x > 0) {
        double exp_neg_x = std::exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
inv_logit(const T& x) {
    return ad::inv_logit(x);
}

// log1p
template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, T>::type
log1p_fn(T x) {
    return std::log1p(x);
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, T>::type
log1p_fn(const T& x) {
    return ad::log1p(x);
}

// ============================================================================
// Value extraction (for getting double from T)
// ============================================================================

template<typename T>
inline typename std::enable_if<!is_ad_var<T>::value, double>::type
get_value(const T& x) {
    return x;
}

template<typename T>
inline typename std::enable_if<is_ad_var<T>::value, double>::type
get_value(const T& x) {
    return x.val();
}

// ============================================================================
// Dot product that works with both types
// ============================================================================

template<typename T>
inline T dot_product(const std::vector<T>& x, const std::vector<T>& y) {
    T sum = T(0.0);
    for (size_t i = 0; i < x.size(); i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// Dot product with double design matrix and T coefficients
template<typename T>
inline T dot_product_mixed(const double* x, const std::vector<T>& y, int n) {
    T sum = T(0.0);
    for (int i = 0; i < n; i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// Simpler: dot product where x is double array, y is T vector starting at offset
template<typename T>
inline T dot_product_mixed(const double* x, const T* y, int n) {
    T sum = T(0.0);
    for (int i = 0; i < n; i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// ============================================================================
// Likelihood functions (templated)
// ============================================================================

template<typename T>
inline T log_lik_poisson(int y, const T& mu) {
    // y * log(mu) - mu - lgamma(y+1)
    // Note: lgamma(y+1) is constant w.r.t. parameters, but we include it for correctness
    return y * safe_log(mu) - mu - lgamma_fn(T(y + 1.0));
}

template<typename T>
inline T log_lik_gamma(double y, const T& shape, const T& mu) {
    // Gamma(y | shape, rate) where rate = shape/mu (mean parameterization)
    // = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape)
    T rate = shape / mu;
    return shape * safe_log(rate) + (shape - 1.0) * std::log(y) - rate * y - lgamma_fn(shape);
}

template<typename T>
inline T log_lik_negbin(int y, const T& mu, const T& phi) {
    // Negative binomial with mean mu and overdispersion phi
    // Using the Gamma-Poisson mixture parameterization
    T log_lik = lgamma_fn(T(y) + phi) - lgamma_fn(T(y + 1.0)) - lgamma_fn(phi);
    log_lik = log_lik + phi * safe_log(phi / (mu + phi));
    log_lik = log_lik + y * safe_log(mu / (mu + phi));
    return log_lik;
}

template<typename T>
inline T log_lik_binomial(int y, int n, const T& p) {
    // y * log(p) + (n-y) * log(1-p) + lchoose(n, y)
    // lchoose is constant w.r.t. p
    T log_lik = y * safe_log(p) + (n - y) * safe_log(T(1.0) - p);
    log_lik = log_lik + R::lchoose(n, y);  // constant
    return log_lik;
}

// ============================================================================
// Prior distributions (templated)
// ============================================================================

// Normal prior: -0.5 * tau * x^2 (ignoring constants)
template<typename T>
inline T log_prior_normal(const T& x, double tau) {
    return T(-0.5) * tau * x * x;
}

// Half-Cauchy prior on sigma (log scale): -log(1 + (sigma/scale)^2) + log(sigma)
template<typename T>
inline T log_prior_half_cauchy(const T& log_sigma, double scale) {
    T sigma = safe_exp(log_sigma);
    T ratio = sigma / scale;
    return -safe_log(T(1.0) + ratio * ratio) + log_sigma;
}

// Gamma prior on phi (log scale): (shape-1)*log(phi) - rate*phi + log(phi)
template<typename T>
inline T log_prior_gamma(const T& log_phi, double shape, double rate) {
    T phi = safe_exp(log_phi);
    return (shape - 1.0) * log_phi - rate * phi + log_phi;
}

}  // namespace math
}  // namespace ratiod

#endif  // RATIOD_AUTODIFF_UTILS_H
