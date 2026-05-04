// lik_dispatch.cpp
// Table-driven family dispatcher for tulpaRatio LikelihoodSpec.
//
// B1b registered the 7 base ratio families. B1c adds ZI / hurdle / OI / ZOIB
// variants — the family builder itself stays single-source, branching on
// cfg.zi inside the templated likelihood callback. We only validate here that
// the requested cfg.zi is compatible with cfg.family (so the templated
// switch never falls into the default branch silently).

#include <Rcpp.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// Per-family compatibility table for cfg.zi. "none" is always valid. Adding a
// new family or ZI variant: one line here + matching switch case in the
// family's templated likelihood.
const std::unordered_map<std::string, std::unordered_set<std::string>>&
zi_compat_table() {
    static const std::unordered_map<std::string, std::unordered_set<std::string>>
        kZiCompat = {
            {"binomial",      {"none", "zi_binomial", "hurdle_binomial",
                               "oi_binomial", "zoib"}},
            {"poisson_gamma", {"none", "zi_poisson", "hurdle_poisson"}},
            {"negbin_gamma",  {"none", "zi_negbin", "hurdle_negbin"}},
            {"negbin_negbin", {"none", "zi_negbin", "hurdle_negbin"}},
            {"gamma_gamma",   {"none"}},
            {"lognormal",     {"none"}},
            {"beta_binomial", {"none"}},
        };
    return kZiCompat;
}

}  // namespace

tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg) {
    const auto& table = family_table();
    auto it = table.find(cfg.family);
    if (it == table.end()) {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: family='%s' not "
                   "registered. Disable tulpaRatio.use_specs to fall back to "
                   "the legacy backend.",
                   cfg.family.c_str());
    }

    const auto& compat = zi_compat_table();
    auto cit = compat.find(cfg.family);
    if (cit == compat.end() || cit->second.find(cfg.zi) == cit->second.end()) {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: family='%s' does not "
                   "support zi='%s' on the spec path. Disable "
                   "tulpaRatio.use_specs to fall back to the legacy backend.",
                   cfg.family.c_str(), cfg.zi.c_str());
    }

    tulpa::LikelihoodSpec spec = it->second(cfg);
    if (spec.ll_double == nullptr) {
        Rcpp::stop("tulpaRatio LikelihoodSpec dispatcher: family='%s' with "
                   "num_link='%s' / denom_link='%s' has no registered "
                   "implementation.",
                   cfg.family.c_str(), cfg.num_link.c_str(), cfg.denom_link.c_str());
    }
    return spec;
}

}  // namespace tulpaRatio
