#include <Rcpp.h>
#include <functional>
#include <memory>
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

// Copy a column-major Rcpp NumericMatrix into a row-major flat double vector
// with shape [N x p], matching tulpa::ProcessData::X_flat layout.
std::vector<double> rmajor_from_rmatrix(const Rcpp::NumericMatrix& X) {
  const int N = X.nrow();
  const int p = X.ncol();
  std::vector<double> out(static_cast<size_t>(N) * p, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < p; ++j) {
      out[i * p + j] = X(i, j);
    }
  }
  return out;
}

// ----- Per-family payload builder ------------------------------------------
// Each family has its own POD struct. We build the right one based on
// cfg.family, returning a `void*` plus a `unique_ptr<void, ...>` for cleanup.
// Centralising the construction here keeps the family files pure (no Rcpp
// in lik_*.cpp) and the bridge a single switch — no per-family allocation
// boilerplate sprinkled around.
struct ResponsePayload {
  void* ptr = nullptr;
  std::function<void(void*)> deleter;
  ~ResponsePayload() { if (ptr) deleter(ptr); }
  ResponsePayload() = default;
  ResponsePayload(const ResponsePayload&) = delete;
  ResponsePayload& operator=(const ResponsePayload&) = delete;
  ResponsePayload(ResponsePayload&& o) noexcept
      : ptr(o.ptr), deleter(std::move(o.deleter)) { o.ptr = nullptr; }
};

template <typename Payload>
ResponsePayload make_payload(Payload* obj) {
  ResponsePayload r;
  r.ptr = obj;
  r.deleter = [](void* p) { delete static_cast<Payload*>(p); };
  return r;
}

ResponsePayload build_response_payload(
    const std::string& family,
    const Rcpp::IntegerVector& y_num_int,
    const Rcpp::IntegerVector& y_denom_int,
    const Rcpp::NumericVector& y_num_cont,
    const Rcpp::NumericVector& y_denom_cont,
    int N
) {
  using namespace tulpaRatio::lik;

  if (family == "binomial") {
    auto* p = new BinomialResponseData;
    p->y.assign(N, 0); p->n.assign(N, 0);
    for (int i = 0; i < N; ++i) { p->y[i] = y_num_int[i]; p->n[i] = y_denom_int[i]; }
    return make_payload(p);
  }
  if (family == "beta_binomial") {
    auto* p = new BetaBinomialResponseData;
    p->y.assign(N, 0); p->n.assign(N, 0);
    for (int i = 0; i < N; ++i) { p->y[i] = y_num_int[i]; p->n[i] = y_denom_int[i]; }
    return make_payload(p);
  }
  if (family == "poisson_gamma") {
    auto* p = new PoissonGammaResponseData;
    p->y_num.assign(N, 0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num[i]        = y_num_int[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }
  if (family == "negbin_gamma") {
    auto* p = new NegbinGammaResponseData;
    p->y_num.assign(N, 0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num[i]        = y_num_int[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }
  if (family == "negbin_negbin") {
    auto* p = new NegbinNegbinResponseData;
    p->y_num.assign(N, 0); p->y_denom.assign(N, 0);
    for (int i = 0; i < N; ++i) { p->y_num[i] = y_num_int[i]; p->y_denom[i] = y_denom_int[i]; }
    return make_payload(p);
  }
  if (family == "gamma_gamma") {
    auto* p = new GammaGammaResponseData;
    p->y_num_cont.assign(N, 0.0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num_cont[i]   = y_num_cont[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }
  if (family == "lognormal") {
    auto* p = new LognormalResponseData;
    p->y_num_cont.assign(N, 0.0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num_cont[i]   = y_num_cont[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }

  Rcpp::stop("build_response_payload: unknown family '%s'", family.c_str());
  return ResponsePayload();  // unreachable
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
// LikelihoodSpec-driven NUTS entry point (B1b: 7 ratio families).
//
// Builds a 1- or 2-process ModelData populated only with X_num (and X_denom
// when the family has two processes), attaches a per-family response payload,
// asks the family dispatcher for a LikelihoodSpec, then routes through
// tulpa::get_nuts_fn().
//
// Returns a list shaped like the legacy `cpp_hmc_fit` output (samples,
// log_prob, accept_prob, n_leapfrog, treedepth, divergent, epsilon, sampler)
// for a single chain so backend_hmc.R's existing converter can consume it.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpaRatio_run_nuts_specs(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::List cfg_list,
    Rcpp::NumericVector init,
    int n_iter,
    int n_warmup,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 1,
    bool verbose = false,
    std::string gradient_mode = "A_r",
    double sigma_beta = 2.5,
    double phi_prior_shape = 1.0,
    double phi_prior_rate  = 0.01,
    double sigma_prior_scale = 1.0
) {
  const int N      = X_num.nrow();
  const int p_num   = X_num.ncol();
  const int p_denom = X_denom.ncol();

  // ---- Build RatioConfig from R-side list (with hyperprior overrides) -----
  tulpaRatio::RatioConfig cfg;
  cfg.family    = Rcpp::as<std::string>(cfg_list["family"]);
  if (cfg_list.containsElementNamed("zi"))         cfg.zi         = Rcpp::as<std::string>(cfg_list["zi"]);
  if (cfg_list.containsElementNamed("num_link"))   cfg.num_link   = Rcpp::as<std::string>(cfg_list["num_link"]);
  if (cfg_list.containsElementNamed("denom_link")) cfg.denom_link = Rcpp::as<std::string>(cfg_list["denom_link"]);
  cfg.phi_prior_shape   = phi_prior_shape;
  cfg.phi_prior_rate    = phi_prior_rate;
  cfg.sigma_prior_scale = sigma_prior_scale;

  // ---- Validate response shape vs. family expectations -------------------
  const bool has_y_int  = (y_num.size()  == N) && (y_denom.size()  == N);
  const bool has_y_cont = (y_num_cont.size() == N) || (y_denom_cont.size() == N);
  if (!has_y_int && !has_y_cont) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: response vectors all empty.");
  }

  // ---- Build LikelihoodSpec & response payload via dispatcher ------------
  tulpa::LikelihoodSpec spec = tulpaRatio::build_ratio_likelihood_spec(cfg);
  ResponsePayload response = build_response_payload(
      cfg.family, y_num, y_denom, y_num_cont, y_denom_cont, N);

  // ---- ModelData (1 or 2 processes, no spatial/temporal/RE/ZI) ----------
  tulpa::ModelData data;
  data.N = N;
  data.n_processes = spec.n_processes;
  data.sigma_beta  = sigma_beta;
  data.model_response_data = response.ptr;
  data.likelihood_spec     = &spec;

  data.processes.resize(spec.n_processes);
  data.processes[0].p      = p_num;
  data.processes[0].X_flat = rmajor_from_rmatrix(X_num);
  if (spec.n_processes >= 2) {
    if (X_denom.nrow() != N) {
      Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_denom must have N rows for "
                 "a 2-process family ('%s').", cfg.family.c_str());
    }
    data.processes[1].p      = p_denom;
    data.processes[1].X_flat = rmajor_from_rmatrix(X_denom);
  }
  data.sharing.init(spec.n_processes);

  // No ZI/OI in B1b.
  data.zi_type     = tulpa::ZIType::NONE;
  data.p_zi        = 0;
  data.p_oi        = 0;
  data.zi_prior_sd = 1.0;
  data.oi_prior_sd = 1.0;

  // ---- Param layout (engine derives from data + spec) -------------------
  tulpa::ParamLayout layout = tulpa::compute_layout(data);

  int expected_total = p_num + (spec.n_processes >= 2 ? p_denom : 0) + spec.n_extra_params;
  if (layout.total_params != expected_total) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: layout.total_params (%d) != "
               "expected (%d) for family='%s' [p_num=%d, p_denom=%d, n_extra=%d]. "
               "B1b expects no extra latent structure.",
               layout.total_params, expected_total, cfg.family.c_str(),
               p_num, (spec.n_processes >= 2 ? p_denom : 0), spec.n_extra_params);
  }
  if (init.size() != expected_total) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: init length (%d) != expected (%d) "
               "for family='%s'.",
               (int)init.size(), expected_total, cfg.family.c_str());
  }

  // ---- Pin gradient mode -------------------------------------------------
  int grad_rc = tulpa::set_gradient_mode_str(gradient_mode.c_str());
  if (grad_rc != 0) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: unrecognised gradient_mode '%s'.",
               gradient_mode.c_str());
  }

  // ---- Run NUTS via the registered tulpa entry point ---------------------
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
