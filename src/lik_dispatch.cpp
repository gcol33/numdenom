// lik_dispatch.cpp
// Table-driven family dispatcher for tulpaRatio LikelihoodSpec.
//
// B1b registers all 7 ratio families. ZI variants (zi != "none") still error
// here — those land in B1c. Adding a new family is two lines: one forward
// declaration plus one entry in `kFamilyTable`.

#include <Rcpp.h>
#include <string>
#include <unordered_map>
#include <tulpa/likelihood.h>

#include "lik_specs/ratio_config.h"

namespace tulpaRatio {

// Forward declarations of per-family builders.
tulpa::LikelihoodSpec build_binomial_spec     (const RatioConfig& cfg);
tulpa::LikelihoodSpec build_poisson_gamma_spec(const RatioConfig& cfg);
tulpa::LikelihoodSpec build_negbin_gamma_spec (const RatioConfig& cfg);
tulpa::LikelihoodSpec build_negbin_negbin_spec(const RatioConfig& cfg);
tulpa::LikelihoodSpec build_gamma_gamma_spec  (const RatioConfig& cfg);
tulpa::LikelihoodSpec build_lognormal_spec    (const RatioConfig& cfg);
tulpa::LikelihoodSpec build_beta_binomial_spec(const RatioConfig& cfg);

namespace {

using FamilyBuilder = tulpa::LikelihoodSpec(*)(const RatioConfig&);

const std::unordered_map<std::string, FamilyBuilder>& family_table() {
    static const std::unordered_map<std::string, FamilyBuilder> kFamilyTable = {
        {"binomial",      &build_binomial_spec},
        {"poisson_gamma", &build_poisson_gamma_spec},
        {"negbin_gamma",  &build_negbin_gamma_spec},
        {"negbin_negbin", &build_negbin_negbin_spec},
        {"gamma_gamma",   &build_gamma_gamma_spec},
        {"lognormal",     &build_lognormal_spec},
        {"beta_binomial", &build_beta_binomial_spec},
    };
    return kFamilyTable;
}

}  // namespace

tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg) {
    if (cfg.zi != "none") {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: zi='%s' not implemented "
                   "in B1b (ZI lands in B1c). Disable tulpaRatio.use_specs to "
                   "fall back to the legacy backend.",
                   cfg.zi.c_str());
    }

    const auto& table = family_table();
    auto it = table.find(cfg.family);
    if (it == table.end()) {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: family='%s' not "
                   "registered. Disable tulpaRatio.use_specs to fall back to "
                   "the legacy backend.",
                   cfg.family.c_str());
    }

    tulpa::LikelihoodSpec spec = it->second(cfg);
    if (spec.ll_double == nullptr) {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: family='%s' with "
                   "num_link='%s' / denom_link='%s' has no registered "
                   "implementation in B1b.",
                   cfg.family.c_str(), cfg.num_link.c_str(), cfg.denom_link.c_str());
    }
    return spec;
}

}  // namespace tulpaRatio
