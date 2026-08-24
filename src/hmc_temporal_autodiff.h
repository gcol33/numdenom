// hmc_temporal_autodiff.h
// Templated temporal functions for autodiff support
// Works with both double (for evaluation) and ad::Var (for gradients)

#ifndef RATIOD_HMC_TEMPORAL_AUTODIFF_H
#define RATIOD_HMC_TEMPORAL_AUTODIFF_H

#include <vector>
#include <cmath>
#include "hmc_temporal.h"
#include "autodiff_utils.h"

namespace ratiod_temporal {

using namespace ratiod::math;

// =============================================================================
// Templated RW1 quadratic form
// =============================================================================

// The pointer form is the implementation: callers that hold a strided slice of a
// flat coefficient vector (a spatiotemporal location's series, one TVC term's
// trajectory) can pass an offset instead of copying into a vector first.
template<typename T>
T rw1_quadratic_form_t(
    const T* phi,
    int T_len,
    bool cyclic
) {
    T quad = T(0.0);

    // Sum of squared differences
    for (int t = 0; t < T_len - 1; t++) {
        T diff = phi[t + 1] - phi[t];
        quad = quad + diff * diff;
    }

    // Cyclic: add edge from T to 1
    if (cyclic && T_len > 0) {
        T diff = phi[0] - phi[T_len - 1];
        quad = quad + diff * diff;
    }

    return quad;
}

// =============================================================================
// Templated RW2 quadratic form
// =============================================================================

template<typename T>
T rw1_quadratic_form_t(const std::vector<T>& phi, int T_len, bool cyclic) {
    return rw1_quadratic_form_t(phi.data(), T_len, cyclic);
}

template<typename T>
T rw2_quadratic_form_t(
    const T* phi,
    int T_len,
    bool cyclic
) {
    if (T_len < 3) return T(0.0);

    T quad = T(0.0);

    // Sum of squared second differences
    for (int t = 0; t < T_len - 2; t++) {
        T diff2 = phi[t] - T(2.0) * phi[t + 1] + phi[t + 2];
        quad = quad + diff2 * diff2;
    }

    // Cyclic: wrap around
    if (cyclic && T_len >= 3) {
        // t = T-2
        T diff2_1 = phi[T_len - 2] - T(2.0) * phi[T_len - 1] + phi[0];
        quad = quad + diff2_1 * diff2_1;
        // t = T-1
        T diff2_2 = phi[T_len - 1] - T(2.0) * phi[0] + phi[1];
        quad = quad + diff2_2 * diff2_2;
    }

    return quad;
}

// =============================================================================
// Templated AR1 log-density
// =============================================================================

template<typename T>
T rw2_quadratic_form_t(const std::vector<T>& phi, int T_len, bool cyclic) {
    return rw2_quadratic_form_t(phi.data(), T_len, cyclic);
}

template<typename T>
T ar1_log_density_t(
    const T* phi,
    int T_len,
    const T& rho,
    const T& tau  // precision = 1/sigma^2
) {
    return ratiod_ar1::ar1_log_density(phi, T_len, rho, tau);
}

template<typename T>
T ar1_log_density_t(const std::vector<T>& phi, int T_len, const T& rho, const T& tau) {
    return ar1_log_density_t(phi.data(), T_len, rho, tau);
}

// =============================================================================
// Templated temporal log-prior
// =============================================================================

template<typename T>
T temporal_log_prior_t(
    const std::vector<T>& phi,
    TemporalType type,
    const T& tau,     // precision for RW1/RW2, or conditional precision for AR1
    const T& rho,     // AR1 autocorrelation (ignored for RW)
    bool cyclic
) {
    int T_len = phi.size();
    T log_prior = T(0.0);

    if (type == TemporalType::RW1) {
        // RW1: p(phi|tau) propto tau^{(T-1)/2} exp(-0.5 * tau * phi' Q phi)
        T quad = rw1_quadratic_form_t(phi, T_len, cyclic);
        int rank = tulpa::rw1_rank(T_len, cyclic);
        log_prior = log_prior + T(0.5 * rank) * safe_log(tau);
        log_prior = log_prior - T(0.5) * tau * quad;

    } else if (type == TemporalType::RW2) {
        // RW2: p(phi|tau) propto tau^{(T-2)/2} exp(-0.5 * tau * phi' Q phi)
        T quad = rw2_quadratic_form_t(phi, T_len, cyclic);
        int rank = tulpa::rw2_rank(T_len, cyclic);
        log_prior = log_prior + T(0.5 * rank) * safe_log(tau);
        log_prior = log_prior - T(0.5) * tau * quad;

    } else if (type == TemporalType::AR1) {
        // AR1: proper prior
        log_prior = log_prior + ar1_log_density_t(phi, T_len, rho, tau);

    } else if (type == TemporalType::IID) {
        // IID N(0, 1/tau): sum of independent normal log-densities
        T sigma2 = T(1.0) / tau;
        T log_norm = T(-0.5) * safe_log(T(2.0 * M_PI) * sigma2);
        for (int t = 0; t < T_len; t++) {
            log_prior = log_prior + log_norm - T(0.5) * phi[t] * phi[t] / sigma2;
        }
    }

    return log_prior;
}

}  // namespace ratiod_temporal

#endif  // RATIOD_HMC_TEMPORAL_AUTODIFF_H
