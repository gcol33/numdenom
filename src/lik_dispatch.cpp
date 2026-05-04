// lik_dispatch.cpp
// Table-driven family dispatcher for tulpaRatio LikelihoodSpec.
//
// B1a registers only the binomial family. Other families fall through to a
// clear runtime error so the R-side feature flag can keep them on the legacy
// engine while this PoC is in place.

#include <Rcpp.h>
#include <string>
#include <tulpa/likelihood.h>

#include "lik_specs/ratio_config.h"

namespace tulpaRatio {

// Forward declarations of per-family builders.
tulpa::LikelihoodSpec build_binomial_spec(const RatioConfig& cfg);

tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg) {
    if (cfg.zi != "none") {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: zi='%s' not implemented "
                   "in B1a (binomial-only PoC). Disable tulpaRatio.use_specs to "
                   "fall back to the legacy backend.",
                   cfg.zi.c_str());
    }

    if (cfg.family == "binomial") {
        tulpa::LikelihoodSpec spec = build_binomial_spec(cfg);
        if (spec.ll_double == nullptr) {
            Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: binomial family with "
                       "num_link='%s' is not registered. Only 'logit' is supported "
                       "in B1a.", cfg.num_link.c_str());
        }
        return spec;
    }

    Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: family='%s' not implemented "
               "in B1a. Only 'binomial' is registered. Disable "
               "tulpaRatio.use_specs to fall back to the legacy backend.",
               cfg.family.c_str());
}

}  // namespace tulpaRatio
