// ratio_config.h
// POD struct describing a tulpaRatio LikelihoodSpec request.
//
// Decouples the R-side family parsing from the C++ spec builder: R fills in the
// fields, the dispatcher in lik_dispatch.cpp reads them. Add new families by
// appending fields here and a branch in build_ratio_likelihood_spec.

#ifndef TULPARATIO_RATIO_CONFIG_H
#define TULPARATIO_RATIO_CONFIG_H

#include <string>

namespace tulpaRatio {

struct RatioConfig {
    // Family name: "binomial" only for B1a; "negbin_negbin", "poisson_gamma",
    // "negbin_gamma", "gamma_gamma", "lognormal", "beta_binomial" deferred.
    std::string family;

    // Zero / one-inflation variant: "none" only for B1a. Future values mirror
    // the legacy `model_type_str` + ZIType combinations
    // ("zi_binomial", "oi_binomial", "zoib", "hurdle_binomial").
    std::string zi = "none";

    // Numerator link function: "logit" only for B1a. Future: "probit", "cloglog".
    std::string num_link = "logit";

    // Denominator link function: unused for binomial (denom is fixed trial count).
    // Reserved so non-binomial families can populate it consistently.
    std::string denom_link = "log";
};

}  // namespace tulpaRatio

#endif  // TULPARATIO_RATIO_CONFIG_H
