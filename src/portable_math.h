// portable_math.h
// Thread-safe C++ math functions that replace R::lgammafn, R::digamma, etc.
// No R dependency in hot paths - avoids R's C stack checking and is OpenMP-safe.

#ifndef RATIOD_PORTABLE_MATH_H
#define RATIOD_PORTABLE_MATH_H

#include <cmath>
#include <limits>

namespace ratiod {
namespace math {

// Portable digamma using asymptotic expansion + recurrence
// For x > 0 only (sufficient for all our use cases)
inline double portable_digamma(double x) {
    if (x <= 0.0) return -1e10;  // guard (should not happen in practice)
    double result = 0.0;
    // Use recurrence to shift x >= 7 for good asymptotic convergence
    while (x < 7.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion: digamma(x) ~ log(x) - 1/(2x) - sum B_{2k}/(2k*x^{2k})
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    // Bernoulli coefficients: B_2/(2*x^2), B_4/(4*x^4), ...
    result += std::log(x) - 0.5 * inv_x
              - inv_x2 * (1.0/12.0
              - inv_x2 * (1.0/120.0
              - inv_x2 * (1.0/252.0
              - inv_x2 * (1.0/240.0
              - inv_x2 * (5.0/660.0
              - inv_x2 * (691.0/32760.0
              - inv_x2 * (1.0/12.0)))))));
    return result;
}

// Portable trigamma using asymptotic expansion + recurrence
inline double portable_trigamma(double x) {
    if (x <= 0.0) return 1e10;  // guard
    double result = 0.0;
    // Use recurrence to shift x >= 7
    while (x < 7.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic expansion: trigamma(x) ~ 1/x + 1/(2x^2) + sum B_{2k}/(x^{2k+1})
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += inv_x + 0.5 * inv_x2
              + inv_x2 * inv_x * (1.0/6.0
              - inv_x2 * (1.0/30.0
              - inv_x2 * (1.0/42.0
              - inv_x2 * (1.0/30.0
              - inv_x2 * (5.0/66.0)))));
    return result;
}

// Portable lchoose: log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
inline double portable_lchoose(int n, int k) {
    if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

}  // namespace math
}  // namespace ratiod

#endif  // RATIOD_PORTABLE_MATH_H
