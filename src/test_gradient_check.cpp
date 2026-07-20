// Finite-difference verification for the analytic gradient.
//
// The log posterior and the analytic gradient are written out separately, in
// several copies, for each field type. When a prior term is added to one and
// not the other, NUTS samples a density whose gradient points elsewhere; the
// chain still runs and still returns numbers, so nothing downstream fails.
// These helpers build a small model, evaluate both, and compare.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "hmc_sampler.h"
#include "hmc_temporal.h"

using namespace ratiod_hmc;

namespace {

// Rook-and-diagonal adjacency on the smallest square grid holding n units,
// emitted as CSR. Matches the structure the benchmark builds in R.
void build_grid_adjacency(int n,
                          std::vector<int>& row_ptr,
                          std::vector<int>& col_idx,
                          std::vector<int>& n_neighbors) {
  const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
  std::vector<int> lon(n), lat(n);
  for (int i = 0; i < n; i++) {
    lon[i] = i % side;
    lat[i] = i / side;
  }
  row_ptr.assign(n + 1, 0);
  col_idx.clear();
  n_neighbors.assign(n, 0);
  for (int i = 0; i < n; i++) {
    row_ptr[i] = static_cast<int>(col_idx.size());
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      const int dx = lon[i] - lon[j];
      const int dy = lat[i] - lat[j];
      if (dx * dx + dy * dy <= 2) {
        col_idx.push_back(j);
        n_neighbors[i]++;
      }
    }
  }
  row_ptr[n] = static_cast<int>(col_idx.size());
}

// Synthetic model with an intercept, one covariate, and whichever structured
// field is under test. `family` selects binomial (no denominator predictors)
// or poisson_gamma (a continuous denominator with its own predictors), which
// is what exercises the paths that treat the two linear predictors
// differently, `temporal_shared` among them.
ModelData make_model(const std::string& field, int n_obs, int n_units,
                     int n_times, unsigned int seed,
                     const std::string& family = "binomial",
                     bool temporal_shared = true) {
  ModelData data;
  std::mt19937 rng(seed);
  std::normal_distribution<double> rnorm(0.0, 1.0);
  std::uniform_int_distribution<int> rtrials(10, 50);

  const bool is_binomial = (family == "binomial");

  data.N = n_obs;
  data.model_type = is_binomial ? ModelType::BINOMIAL : ModelType::POISSON_GAMMA;
  data.p_num = 2;
  data.p_denom = is_binomial ? 0 : 2;
  data.n_threads = 1;
  data.n_cores = 1;

  data.sigma_beta = 10.0;
  data.sigma_re_scale = 1.0;
  data.phi_prior_shape = 1.0;
  data.phi_prior_rate = 0.01;
  data.tau_spatial_shape = 1.0;
  data.tau_spatial_rate = 0.01;
  data.tau_temporal_shape = 1.0;
  data.tau_temporal_rate = 0.01;

  data.X_num_flat.resize(static_cast<size_t>(n_obs) * 2);
  data.y_num.resize(n_obs);
  // y_denom is sized for every family: the continuous-denominator families
  // carry a placeholder of 1, matching what the backend passes down, and the
  // lgamma/lchoose precompute reads it unconditionally.
  data.y_denom.assign(n_obs, 1);
  if (!is_binomial) {
    data.X_denom_flat.resize(static_cast<size_t>(n_obs) * 2);
    data.y_denom_cont.resize(n_obs);
  }
  for (int i = 0; i < n_obs; i++) {
    const double x = rnorm(rng);
    data.X_num_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
    data.X_num_flat[static_cast<size_t>(i) * 2 + 1] = x;
    if (is_binomial) {
      const int trials = rtrials(rng);
      const double p = 1.0 / (1.0 + std::exp(-(0.5 + 0.3 * x)));
      std::binomial_distribution<int> rbinom(trials, p);
      data.y_denom[i] = trials;
      data.y_num[i] = rbinom(rng);
    } else {
      const double z = rnorm(rng);
      data.X_denom_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.X_denom_flat[static_cast<size_t>(i) * 2 + 1] = z;
      std::poisson_distribution<int> rpois(std::exp(0.5 + 0.3 * x));
      data.y_num[i] = rpois(rng);
      std::gamma_distribution<double> rgamma(2.0, std::exp(0.4 + 0.2 * z) / 2.0);
      data.y_denom_cont[i] = std::max(1e-3, rgamma(rng));
    }
  }

  // No random effects: the constrained field is what is under test.
  data.n_re_terms = 0;
  data.n_re_groups = 0;
  data.total_re_groups = 0;
  data.re_parameterization = 0;

  const bool want_icar = (field == "icar" || field == "icar_rw1");
  const bool want_bym2 = (field == "bym2");
  const bool want_rw1  = (field == "rw1"  || field == "icar_rw1");
  const bool want_rw2  = (field == "rw2");

  if (want_icar || want_bym2) {
    data.spatial_type = want_bym2 ? SpatialType::BYM2 : SpatialType::ICAR;
    data.n_spatial_units = n_units;
    build_grid_adjacency(n_units, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    data.spatial_group.resize(n_obs);
    for (int i = 0; i < n_obs; i++) data.spatial_group[i] = (i % n_units) + 1;
    data.bym2_scale_factor = 1.0;
    data.icar_collapsed = false;
    data.bym2_collapsed = false;
  } else {
    data.spatial_type = SpatialType::NONE;
    data.n_spatial_units = 0;
    data.bym2_scale_factor = 1.0;
  }

  if (want_rw1 || want_rw2) {
    data.temporal_type = want_rw2 ? TemporalType::RW2 : TemporalType::RW1;
    data.n_times = n_times;
    data.n_temporal_groups = 1;
    data.n_temporal_params = n_times;
    data.temporal_cyclic = false;
    data.temporal_shared = temporal_shared;
    data.temporal_time_idx.resize(n_obs);
    data.temporal_group_idx.assign(n_obs, 1);
    for (int i = 0; i < n_obs; i++) data.temporal_time_idx[i] = (i % n_times) + 1;
  } else {
    data.temporal_type = TemporalType::NONE;
    data.n_times = 0;
    data.n_temporal_groups = 0;
    data.n_temporal_params = 0;
  }

  data.zi_type = ZIType::NONE;
  data.p_zi = 0;
  data.zi_prior_sd = 1.0;
  data.p_oi = 0;
  data.oi_prior_sd = 1.0;

  return data;
}

}  // namespace

// Returns the analytic gradient alongside a central-difference gradient of the
// log posterior, both evaluated at the same random point.
// [[Rcpp::export]]
Rcpp::List cpp_gradient_check(std::string field,
                              int n_obs = 400,
                              int n_units = 25,
                              int n_times = 10,
                              double eps = 1e-5,
                              unsigned int seed = 42,
                              std::string mode = "handcoded",
                              bool precenter = false,
                              std::string family = "binomial",
                              bool temporal_shared = true) {
  ModelData data = make_model(field, n_obs, n_units, n_times, seed,
                              family, temporal_shared);
  ParamLayout layout = compute_param_layout(data);
  const int n_params = get_n_params(data);

  GradientMode gm = GradientMode::HANDCODED;
  if (mode == "arena")    gm = GradientMode::AUTODIFF_ARENA;
  if (mode == "forward")  gm = GradientMode::AUTODIFF_FORWARD;
  if (mode == "tape")     gm = GradientMode::AUTODIFF_TAPE;
  set_gradient_mode(gm);

  // A point away from the origin, so that terms which vanish at zero (a
  // sum-to-zero penalty among them) still contribute.
  std::mt19937 rng(seed + 1000u);
  std::normal_distribution<double> rnorm(0.0, 0.35);
  std::vector<double> params(n_params);
  for (int i = 0; i < n_params; i++) params[i] = rnorm(rng);

  if (precenter && layout.spatial_start >= 0) {
    double m = 0.0;
    for (int i = layout.spatial_start; i < layout.spatial_end; i++) m += params[i];
    m /= static_cast<double>(layout.spatial_end - layout.spatial_start);
    for (int i = layout.spatial_start; i < layout.spatial_end; i++) params[i] -= m;
  }

  std::vector<double> grad_analytic(n_params, 0.0);
  double lp_from_mode = 0.0;
  compute_gradient(params, data, layout, grad_analytic, &lp_from_mode);

  std::vector<double> grad_fd(n_params, 0.0);
  for (int i = 0; i < n_params; i++) {
    std::vector<double> up = params, dn = params;
    const double h = eps * std::max(1.0, std::fabs(params[i]));
    up[i] += h;
    dn[i] -= h;
    const double lp_up = compute_log_post(up, data, layout);
    const double lp_dn = compute_log_post(dn, data, layout);
    grad_fd[i] = (lp_up - lp_dn) / (2.0 * h);
  }

  Rcpp::CharacterVector block(n_params, "other");
  for (int i = 0; i < n_params; i++) {
    if (i >= layout.beta_num_start && i < layout.beta_num_end) block[i] = "beta_num";
    if (layout.spatial_start >= 0 && i >= layout.spatial_start && i < layout.spatial_end)
      block[i] = "spatial";
    if (layout.theta_bym2_start >= 0 && i >= layout.theta_bym2_start && i < layout.theta_bym2_end)
      block[i] = "theta_bym2";
    if (layout.temporal_start >= 0 && i >= layout.temporal_start && i < layout.temporal_end)
      block[i] = "temporal";
    if (i == layout.log_tau_spatial_idx) block[i] = "log_tau_spatial";
    if (i == layout.log_tau_temporal_idx) block[i] = "log_tau_temporal";
  }

  return Rcpp::List::create(
    Rcpp::Named("analytic") = Rcpp::wrap(grad_analytic),
    Rcpp::Named("finite_diff") = Rcpp::wrap(grad_fd),
    Rcpp::Named("block") = block,
    Rcpp::Named("n_params") = n_params,
    Rcpp::Named("log_post") = compute_log_post(params, data, layout),
    Rcpp::Named("log_post_mode") = lp_from_mode
  );
}

// Evaluates the log posterior at a shared point so the analytic and autodiff
// implementations, which are written out separately, can be compared directly.
// [[Rcpp::export]]
double cpp_logpost_at(std::string field,
                      int n_obs = 400,
                      int n_units = 25,
                      int n_times = 10,
                      unsigned int seed = 42,
                      std::string family = "binomial",
                      bool temporal_shared = true) {
  ModelData data = make_model(field, n_obs, n_units, n_times, seed,
                              family, temporal_shared);
  ParamLayout layout = compute_param_layout(data);
  const int n_params = get_n_params(data);
  std::mt19937 rng(seed + 1000u);
  std::normal_distribution<double> rnorm(0.0, 0.35);
  std::vector<double> params(n_params);
  for (int i = 0; i < n_params; i++) params[i] = rnorm(rng);
  return compute_log_post(params, data, layout);
}
