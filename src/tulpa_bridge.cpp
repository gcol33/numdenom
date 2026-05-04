#include <Rcpp.h>
#include <vector>
#include <string>
#include <tulpa/nuts_api.h>
#include <tulpa/pg_api.h>
#include <tulpa/likelihood.h>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>

#include "lik_specs/ratio_config.h"
#include "lik_specs/lik_helpers.h"

namespace tulpaRatio {
tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg);
}

namespace {

Rcpp::NumericMatrix row_major_matrix(const double* values, int nrow, int ncol) {
  Rcpp::NumericMatrix out(nrow, ncol);
  if (values == nullptr) {
    return out;
  }
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      out(i, j) = values[i * ncol + j];
    }
  }
  return out;
}

Rcpp::NumericVector buffer_vector(const double* values, int n) {
  Rcpp::NumericVector out(n);
  if (values == nullptr) {
    return out;
  }
  for (int i = 0; i < n; ++i) {
    out[i] = values[i];
  }
  return out;
}

template <typename T>
Rcpp::IntegerVector buffer_int_vector(const T* values, int n) {
  Rcpp::IntegerVector out(n);
  if (values == nullptr) {
    return out;
  }
  for (int i = 0; i < n; ++i) {
    out[i] = static_cast<int>(values[i]);
  }
  return out;
}

} // namespace

// [[Rcpp::export]]
int cpp_tulpa_abi_version() {
  tulpa::check_abi_version();
  auto fn = reinterpret_cast<tulpa::GetABIVersionFn>(
    R_GetCCallable("tulpa", "tulpa_get_abi_version")
  );
  return fn();
}

// [[Rcpp::export]]
Rcpp::List cpp_tulpa_pg_binomial_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_scale = 2.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  tulpa::PGShimResult result{};
  result.beta = nullptr;
  result.re = nullptr;
  result.sigma_re = nullptr;
  result.eta = nullptr;
  result.r_disp = nullptr;
  result.spatial = nullptr;
  result.tau_spatial = nullptr;

  tulpa::get_pg_binomial_fn()(
    INTEGER(y),
    INTEGER(n),
    REAL(X),
    y.size(),
    X.ncol(),
    INTEGER(group),
    n_groups,
    n_iter,
    n_warmup,
    thin,
    prior_beta_sd,
    prior_sigma_scale,
    store_eta ? 1 : 0,
    verbose ? 1 : 0,
    n_threads,
    &result
  );

  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("beta") = row_major_matrix(result.beta, result.n_save, result.n_beta),
    Rcpp::Named("re") = row_major_matrix(result.re, result.n_save, result.n_re),
    Rcpp::Named("sigma_re") = buffer_vector(result.sigma_re, result.n_save)
  );

  if (store_eta) {
    out["eta"] = row_major_matrix(result.eta, result.n_save, result.n_obs);
  }

  result.free_buffers();
  return out;
}

// ============================================================================
// LikelihoodSpec-driven NUTS entry point (B1a binomial PoC).
//
// Builds a single-process ModelData populated only with X_num, attaches a
// BinomialResponseData payload, asks the family dispatcher for a
// LikelihoodSpec, then routes through tulpa::get_nuts_fn().
//
// Returns a list shaped like the legacy `cpp_hmc_fit` output (samples,
// log_prob, accept_prob, n_leapfrog, treedepth, divergent, epsilon, sampler)
// for a single chain so backend_hmc.R's existing converter can consume it.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpaRatio_run_nuts_specs(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericMatrix X_num,
    Rcpp::List cfg_list,
    Rcpp::NumericVector init,
    int n_iter,
    int n_warmup,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 1,
    bool verbose = false,
    std::string gradient_mode = "A_r",
    double sigma_beta = 2.5
) {
  const int N = X_num.nrow();
  const int p = X_num.ncol();

  if (y_num.size() != N || y_denom.size() != N) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: y_num/y_denom length must equal nrow(X_num).");
  }
  if (init.size() != p) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: init length (%d) must equal ncol(X_num) (%d) "
               "for B1a binomial scope.", (int)init.size(), p);
  }

  // ---- Build RatioConfig from R-side list ------------------------------------
  tulpaRatio::RatioConfig cfg;
  cfg.family    = Rcpp::as<std::string>(cfg_list["family"]);
  if (cfg_list.containsElementNamed("zi"))         cfg.zi         = Rcpp::as<std::string>(cfg_list["zi"]);
  if (cfg_list.containsElementNamed("num_link"))   cfg.num_link   = Rcpp::as<std::string>(cfg_list["num_link"]);
  if (cfg_list.containsElementNamed("denom_link")) cfg.denom_link = Rcpp::as<std::string>(cfg_list["denom_link"]);

  // ---- Response payload (binomial) -------------------------------------------
  tulpaRatio::lik::BinomialResponseData resp;
  resp.y.assign(N, 0);
  resp.n.assign(N, 0);
  for (int i = 0; i < N; ++i) {
    resp.y[i] = y_num[i];
    resp.n[i] = y_denom[i];
  }

  // ---- Build LikelihoodSpec via dispatcher -----------------------------------
  tulpa::LikelihoodSpec spec = tulpaRatio::build_ratio_likelihood_spec(cfg);

  // ---- ModelData (single process, no spatial/temporal/RE/ZI) -----------------
  tulpa::ModelData data;
  data.N = N;
  data.n_processes = spec.n_processes;
  data.sigma_beta = sigma_beta;
  data.model_response_data = &resp;
  data.likelihood_spec = &spec;

  data.processes.resize(spec.n_processes);
  // X_num is column-major in Rcpp, but ProcessData expects row-major X_flat[N*p].
  data.processes[0].p = p;
  data.processes[0].X_flat.assign(N * p, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < p; ++j) {
      data.processes[0].X_flat[i * p + j] = X_num(i, j);
    }
  }
  // No offset.
  data.sharing.init(spec.n_processes);

  // No ZI/OI in B1a.
  data.zi_type   = tulpa::ZIType::NONE;
  data.p_zi      = 0;
  data.p_oi      = 0;
  data.zi_prior_sd = 1.0;
  data.oi_prior_sd = 1.0;

  // ---- Param layout (engine derives from data + spec) ------------------------
  tulpa::ParamLayout layout = tulpa::compute_layout(data);
  if (layout.total_params != p) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: layout.total_params (%d) != p (%d). "
               "B1a expects a binomial intercept-only-or-fixed-only model with "
               "no extra latent structure.",
               layout.total_params, p);
  }

  // ---- Pin gradient mode -----------------------------------------------------
  int grad_rc = tulpa::set_gradient_mode_str(gradient_mode.c_str());
  if (grad_rc != 0) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: unrecognised gradient_mode '%s'.",
               gradient_mode.c_str());
  }

  // ---- Run NUTS via the registered tulpa entry point ------------------------
  std::vector<double> init_vec(init.begin(), init.end());
  tulpa::NUTSResult result{};
  tulpa::get_nuts_fn()(
    &data,
    &layout,
    init_vec.data(),
    layout.total_params,
    n_iter,
    n_warmup,
    max_treedepth,
    adapt_delta,
    static_cast<unsigned int>(seed),
    verbose ? 1 : 0,
    &result
  );

  const int n_save = result.n_sample;
  const int n_par  = result.n_params;

  Rcpp::NumericMatrix samples = row_major_matrix(result.samples, n_save, n_par);
  Rcpp::NumericVector log_prob    = buffer_vector(result.log_prob,    n_save);
  Rcpp::NumericVector accept_prob = buffer_vector(result.accept_prob, n_save);
  Rcpp::IntegerVector divergent   = buffer_int_vector(result.divergent, n_save);
  Rcpp::IntegerVector treedepth   = buffer_int_vector(result.treedepth, n_save);
  // The legacy converter also reads `n_leapfrog`. NUTSResult doesn't expose it
  // separately; report `2^treedepth - 1` as a faithful proxy (the treedepth is
  // the per-iteration step count exponent in the doubling scheme).
  Rcpp::IntegerVector n_leapfrog(n_save);
  for (int s = 0; s < n_save; ++s) {
    int td = treedepth[s];
    n_leapfrog[s] = (td > 0) ? ((1 << td) - 1) : 1;
  }

  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("samples")     = samples,
    Rcpp::Named("log_prob")    = log_prob,
    Rcpp::Named("accept_prob") = accept_prob,
    Rcpp::Named("n_leapfrog")  = n_leapfrog,
    Rcpp::Named("treedepth")   = treedepth,
    Rcpp::Named("divergent")   = divergent,
    Rcpp::Named("epsilon")     = result.epsilon,
    Rcpp::Named("sampler")     = std::string(result.sampler[0] ? result.sampler : "NUTS")
  );

  result.free_buffers();
  return out;
}
