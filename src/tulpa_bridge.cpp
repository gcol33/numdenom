#include <Rcpp.h>
#include <tulpa/nuts_api.h>
#include <tulpa/pg_api.h>

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
