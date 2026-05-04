// ratio_config.h
// POD struct describing a tulpaRatio LikelihoodSpec request.
//
// Decouples the R-side family parsing from the C++ spec builder: R fills in the
// fields, the dispatcher in lik_dispatch.cpp reads them. Add new families by
// appending fields here and a registry entry in build_ratio_likelihood_spec.

#ifndef TULPARATIO_RATIO_CONFIG_H
#define TULPARATIO_RATIO_CONFIG_H

#include <string>

namespace tulpaRatio {

struct RatioConfig {
    // Family name. B1b registers all 7 ratio families:
    //   "binomial", "poisson_gamma", "negbin_gamma", "negbin_negbin",
    //   "gamma_gamma", "lognormal", "beta_binomial".
    std::string family;

    // Zero / one-inflation variant: "none" only for B1b (ZI deferred to B1c).
    // Future values mirror the legacy `model_type_str` + ZIType combinations.
    std::string zi = "none";

    // Numerator link function: "logit" for binomial families, "log" for
    // count / continuous families. Other links (probit, cloglog) deferred.
    std::string num_link = "logit";

    // Denominator link function. Unused for binomial / beta_binomial / beta
    // (denominator is the fixed trial count). Used for two-process families.
    std::string denom_link = "log";

    // ----- Hyperpriors for extra parameters --------------------------------
    // Gamma(shape, rate) prior on phi-scale parameters (overdispersion,
    // gamma shape, beta-binomial precision). Mirrors data.phi_prior_shape /
    // data.phi_prior_rate from the legacy backend so parity holds without
    // any additional R-side wiring.
    double phi_prior_shape = 1.0;
    double phi_prior_rate  = 0.01;

    // Half-Cauchy(0, sigma_re_scale) prior on lognormal sigma parameters
    // (log scale). Mirrors data.sigma_re_scale from the legacy backend.
    double sigma_prior_scale = 1.0;
};

}  // namespace tulpaRatio

#endif  // TULPARATIO_RATIO_CONFIG_H
