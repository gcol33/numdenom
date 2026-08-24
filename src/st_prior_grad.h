// st_prior_grad.h
// The analytic gradient of the spatiotemporal interaction's own prior.
//
// Two hand-coded gradient functions reach a model carrying an interaction --
// compute_gradient_spatiotemporal_handcoded, which specializes it, and
// compute_gradient_composite, the catch-all -- and each used to carry its own
// copy of every interaction type. The copies had already drifted: the composite
// had no Type II RW2 branch and no HSGP-ST-with-AR1 branch, the two disagreed
// with each other and with the density on the rank a non-RW time margin
// contributes, and neither wrote logit_rho_st at all
// (gcol33/tulpaRatio#66). One implementation is what keeps a term added to the
// density from reaching one gradient and not the other.
//
// Everything here is with respect to the RAW parameter block -- z under the
// non-centered Type IV reparameterization, delta otherwise -- which is what the
// sampler moves.

#ifndef RATIOD_ST_PRIOR_GRAD_H
#define RATIOD_ST_PRIOR_GRAD_H

#include <cmath>
#include <limits>
#include <vector>

#include <tulpa/soft_sum_to_zero.h>

#include "ar1_shared.h"
#include "hmc_hsgp.h"
#include "hmc_spatiotemporal.h"
#include "tls_workspace.h"

namespace ratiod_spatiotemporal {

// What the interaction's prior contributes to each hyperparameter, alongside
// the log-prior itself so a fused gradient/density call does not evaluate the
// Kronecker quadratic form twice.
struct StPriorGrad {
  double log_tau = 0.0;
  double logit_rho = 0.0;
  double log_sigma2_hsgp = 0.0;
  double log_lengthscale_hsgp = 0.0;
  // The GP types' two ranges, with respect to the log parameters the sampler
  // moves. Zero for every Knorr-Held type, which carries no range.
  double log_phi_space = 0.0;
  double log_phi_time = 0.0;
  double log_prior = 0.0;
};

// What a GP interaction's prior contributes, before the sum-to-zero margins.
struct StGpGrad {
  double log_prior = 0.0;
  double log_tau = 0.0;
  double log_phi_space = 0.0;
  double log_phi_time = 0.0;
  bool ok = true;
};

// The GP interaction's prior gradient.
//
// The conditional decomposition is the same pass the density makes
// (st_gp_nngp_scan), so the two cannot disagree about what the field's prior
// is; what is written here is the derivative of each location's Gaussian term.
// With e_i = w_i - m_i, m_i = alpha' w_nb and d_i = sigma2 - c' alpha,
//
//   log p = sum_i [ -0.5 log(2 pi d_i) - 0.5 e_i^2 / d_i ]
//   d/dtheta = (-0.5/d_i + 0.5 e_i^2/d_i^2) dd_i + (e_i/d_i) dm_i.
//
// The FIELD enters only through w, so its gradient is the scatter of -e_i/d_i
// onto location i and +alpha_j e_i/d_i onto each of its neighbours: one pass,
// no second factorization, and exactly the (I - A)' D^-1 (I - A) w application
// the Vecchia factorization already holds.
//
// SIGMA2 factors out of the neighbour block (C = sigma2 R), so alpha and m_i do
// not move with it and d_i is proportional to it. That collapses to
// d(log p)/d(log sigma2) = sum_i (-0.5 + 0.5 e_i^2/d_i) with no extra solve,
// and log_tau = -log sigma2 flips the sign.
//
// The two RANGES need dC/dphi. With beta = C^-1 w_nb, one more pair of
// triangular solves against the factor the scan already built,
//
//   dd_i = -2 (dc/dphi)' alpha + alpha' (dC/dphi) alpha
//   dm_i =    (dc/dphi)' beta  - alpha' (dC/dphi) beta.
//
// A location whose conditional variance sat on its floor contributes no dd_i:
// what is reported there is a constant, not a function of the parameters. Its
// dm_i still counts, the conditional mean being untouched by the floor.
//
//   grad_w  ACCUMULATED, length st.n_params
inline StGpGrad st_gp_nngp_grad(const SpatiotemporalData& st, const double* w,
                                double sigma2, double phi_space,
                                double phi_time, double* grad_w) {
  StGpGrad out;
  double d_dphi_space = 0.0;
  double d_dphi_time = 0.0;

  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_wnb);
  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_tmp);
  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_beta);
  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_dc_s);
  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_dc_t);
  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_dC_s);
  RATIOD_TLS_WORKSPACE(std::vector<double>, gp_dC_t);

  const int nn = st.nn;

  out.ok = st_gp_nngp_scan<double>(
      w, sigma2, phi_space, phi_time, st,
      [&](int i, int obs_idx, const int* nb, int n_nb, const double* L,
          const double* /*c_vec*/, const double* alpha, const double& cond_mean,
          const double& cond_var, bool floored) {
        const double e = w[obs_idx] - cond_mean;
        const double r = e / cond_var;
        out.log_prior += -0.5 * std::log(2.0 * M_PI * cond_var) - 0.5 * e * r;
        grad_w[obs_idx] -= r;
        if (!floored) out.log_tau += 0.5 - 0.5 * e * r;

        if (n_nb == 0) return;
        for (int j = 0; j < n_nb; j++) grad_w[nb[j]] += alpha[j] * r;

        const size_t nsq = static_cast<size_t>(n_nb) * n_nb;
        gp_wnb.resize(n_nb);
        gp_tmp.resize(n_nb);
        gp_beta.resize(n_nb);
        gp_dc_s.resize(n_nb);
        gp_dc_t.resize(n_nb);
        gp_dC_s.assign(nsq, 0.0);
        gp_dC_t.assign(nsq, 0.0);

        for (int j = 0; j < n_nb; j++) gp_wnb[j] = w[nb[j]];
        ratiod_cov::nngp_forward_solve(L, n_nb, gp_wnb.data(), gp_tmp.data());
        ratiod_cov::nngp_back_solve(L, n_nb, gp_tmp.data(), gp_beta.data());

        for (int j = 0; j < n_nb; j++) {
          const size_t f = static_cast<size_t>(i) * nn + j;
          double ds, dt;
          st_corr_dphi(st.nn_dist_space[f], st.nn_dist_time[f], phi_space,
                       phi_time, st, &ds, &dt);
          gp_dc_s[j] = sigma2 * ds;
          gp_dc_t[j] = sigma2 * dt;
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
          for (int j2 = j1; j2 < n_nb; j2++) {
            double h12, u12, ds, dt;
            st_pair_separation(st, nb[j1], nb[j2], &h12, &u12);
            st_corr_dphi(h12, u12, phi_space, phi_time, st, &ds, &dt);
            gp_dC_s[static_cast<size_t>(j1) * n_nb + j2] = sigma2 * ds;
            gp_dC_s[static_cast<size_t>(j2) * n_nb + j1] = sigma2 * ds;
            gp_dC_t[static_cast<size_t>(j1) * n_nb + j2] = sigma2 * dt;
            gp_dC_t[static_cast<size_t>(j2) * n_nb + j1] = sigma2 * dt;
          }
        }

        const double w_d = floored ? 0.0 : (-0.5 / cond_var + 0.5 * r * r);
        double dot_c_a_s = 0.0, dot_c_b_s = 0.0;
        double dot_c_a_t = 0.0, dot_c_b_t = 0.0;
        double quad_aa_s = 0.0, quad_ab_s = 0.0;
        double quad_aa_t = 0.0, quad_ab_t = 0.0;
        for (int j1 = 0; j1 < n_nb; j1++) {
          dot_c_a_s += gp_dc_s[j1] * alpha[j1];
          dot_c_b_s += gp_dc_s[j1] * gp_beta[j1];
          dot_c_a_t += gp_dc_t[j1] * alpha[j1];
          dot_c_b_t += gp_dc_t[j1] * gp_beta[j1];
          double row_a_s = 0.0, row_b_s = 0.0, row_a_t = 0.0, row_b_t = 0.0;
          for (int j2 = 0; j2 < n_nb; j2++) {
            const double s = gp_dC_s[static_cast<size_t>(j1) * n_nb + j2];
            const double t = gp_dC_t[static_cast<size_t>(j1) * n_nb + j2];
            row_a_s += s * alpha[j2];
            row_b_s += s * gp_beta[j2];
            row_a_t += t * alpha[j2];
            row_b_t += t * gp_beta[j2];
          }
          quad_aa_s += alpha[j1] * row_a_s;
          quad_ab_s += alpha[j1] * row_b_s;
          quad_aa_t += alpha[j1] * row_a_t;
          quad_ab_t += alpha[j1] * row_b_t;
        }
        d_dphi_space += w_d * (-2.0 * dot_c_a_s + quad_aa_s)
                      + r * (dot_c_b_s - quad_ab_s);
        d_dphi_time += w_d * (-2.0 * dot_c_a_t + quad_aa_t)
                     + r * (dot_c_b_t - quad_ab_t);
      });

  // The sampled parameters are the logs of the two ranges.
  out.log_phi_space = d_dphi_space * phi_space;
  out.log_phi_time = d_dphi_time * phi_time;
  return out;
}

// The interaction's log-prior gradient.
//
//   st              the interaction's structure
//   is_hsgp         whether the spatial margin is a spectral basis
//   hsgp            the basis; read only when is_hsgp
//   z_or_delta      the raw parameter block
//   delta           the reconstructed field (== z_or_delta unless use_nc)
//   grad_delta_lik  d(log-likelihood) / d(delta)
//   tau, rho        the interaction's precision and time-margin correlation
//   phi_space, phi_time  the GP types' two ranges; unread by Knorr-Held I-IV
//   use_nc          the non-centered Type IV reparameterization, delta = z / sqrt(tau)
//   grad_delta      OUT, length st.n_params, ASSIGNED (not accumulated)
inline StPriorGrad st_interaction_prior_grad(
    const SpatiotemporalData& st,
    bool is_hsgp,
    const ratiod_hsgp::HSGPData& hsgp,
    const double* z_or_delta,
    const double* delta,
    const double* grad_delta_lik,
    double tau,
    double rho,
    double sigma2_hsgp,
    double lengthscale_hsgp,
    double phi_space,
    double phi_time,
    bool use_nc,
    double* grad_delta
) {
  StPriorGrad out;

  const int S = st.n_spatial;
  const int T = st.n_times;
  const int ST = st.n_params;
  const double inv_scale = use_nc ? (1.0 / std::sqrt(tau)) : 1.0;

  // -----------------------------------------------------------------------
  // HSGP-ST: one temporal GMRF per basis function, at the precision the
  // spectral density scales. The spatial margin is the basis, so this branch
  // carries its own sum-to-zero constraint (per basis function, over time) and
  // the S x T margins below do not apply to it.
  // -----------------------------------------------------------------------
  if (is_hsgp) {
    const int M = hsgp.m_total;
    const int rank_t = st_time_rank(st.temporal_type, T, st.temporal_cyclic);
    double drho_sum = 0.0;   // d/d(rho) of the summed quadratic forms, weighted by prec_j

    for (int j = 0; j < M; j++) {
      const double omega_sq = hsgp.eigenvalues[j];
      const double S_j = ratiod_hsgp::spectral_density_se(omega_sq, sigma2_hsgp,
                                                          lengthscale_hsgp);
      const double S_j_safe = std::max(S_j, 1e-10);
      const double prec_j = tau / S_j_safe;

      const double* dj = &delta[j * T];
      double* gj = &grad_delta[j * T];
      double qf = 0.0;

      if (st.temporal_type == TemporalType::RW1) {
        for (int t = 0; t < T; t++) {
          double g = 0.0;
          if (t > 0) { g += prec_j * (dj[t-1] - dj[t]); qf += (dj[t] - dj[t-1]) * (dj[t] - dj[t-1]); }
          if (t < T - 1) g += prec_j * (dj[t+1] - dj[t]);
          gj[t] = grad_delta_lik[j * T + t] + g;
        }
      } else if (st.temporal_type == TemporalType::RW2) {
        for (int t = 0; t < T; t++) {
          double g = 0.0;
          if (t >= 2) g -= prec_j * (dj[t-2] - 2.0 * dj[t-1] + dj[t]);
          if (t >= 1 && t < T - 1) g += 2.0 * prec_j * (dj[t-1] - 2.0 * dj[t] + dj[t+1]);
          if (t < T - 2) g -= prec_j * (dj[t] - 2.0 * dj[t+1] + dj[t+2]);
          gj[t] = grad_delta_lik[j * T + t] + g;
        }
        for (int t = 2; t < T; t++) {
          const double d2 = dj[t-2] - 2.0 * dj[t-1] + dj[t];
          qf += d2 * d2;
        }
      } else if (st.temporal_type == TemporalType::AR1 && T >= 2) {
        RATIOD_TLS_WORKSPACE(std::vector<double>, r_dj);
        r_dj.resize(T);
        ratiod_ar1::ar1_precision_apply(dj, T, rho, r_dj.data());
        for (int t = 0; t < T; t++) {
          gj[t] = grad_delta_lik[j * T + t] - prec_j * r_dj[t];
          qf += dj[t] * r_dj[t];
        }
        double interior = 0.0;
        for (int t = 1; t < T - 1; t++) interior += dj[t] * dj[t];
        double cross = 0.0;
        for (int t = 1; t < T; t++) cross += dj[t] * dj[t-1];
        drho_sum += prec_j * ratiod_ar1::ar1_quadratic_form_drho(interior, cross, rho);
      } else {
        for (int t = 0; t < T; t++) gj[t] = grad_delta_lik[j * T + t];
      }

      // Soft sum-to-zero over time, per basis function.
      double sum_j = 0.0;
      for (int t = 0; t < T; t++) sum_j += dj[t];
      const double push_j = tulpa::s2z_precision(T) * sum_j;
      for (int t = 0; t < T; t++) gj[t] -= push_j;

      out.log_tau += 0.5 * rank_t - 0.5 * prec_j * qf;
      out.log_prior += 0.5 * rank_t * std::log(prec_j) - 0.5 * prec_j * qf
                     - 0.5 * tulpa::s2z_precision(T) * sum_j * sum_j;

      // Both spectral hyperparameters enter only through S_j.
      const double dLogPrior_dS = -0.5 * rank_t / S_j_safe
                                + 0.5 * tau * qf / (S_j_safe * S_j_safe);
      const double dS_dsigma2 = S_j_safe / sigma2_hsgp;
      out.log_sigma2_hsgp += dLogPrior_dS * dS_dsigma2 * sigma2_hsgp;
      const double dS_dl = S_j_safe * (1.0 / lengthscale_hsgp
                                       - lengthscale_hsgp * omega_sq);
      out.log_lengthscale_hsgp += dLogPrior_dS * dS_dl * lengthscale_hsgp;
    }

    if (st.temporal_type == TemporalType::AR1 && T >= 2) {
      // Each basis function carries its own log|R(rho)|, so the log-determinant
      // term counts M of them.
      const double omr2 = ratiod_ar1::one_minus_rho2(rho);
      out.log_prior += 0.5 * M * std::log(omr2);
      const double grad_rho = 0.5 * M * ratiod_ar1::dlog_one_minus_rho2_drho(rho)
                            - 0.5 * drho_sum;
      out.logit_rho += grad_rho * ratiod_ar1::drho_dlogit(rho);
    }

    return out;
  }

  // -----------------------------------------------------------------------
  // Knorr-Held Type I-IV
  // -----------------------------------------------------------------------
  double grad_rho = 0.0;

  if (st.type == STType::TYPE_I) {
    // IID: log p = 0.5*n*log(tau) - 0.5*tau*sum(delta^2)
    double qf = 0.0;
    for (int k = 0; k < ST; k++) {
      grad_delta[k] = grad_delta_lik[k] - tau * delta[k];
      qf += delta[k] * delta[k];
    }
    out.log_tau += 0.5 * ST - 0.5 * tau * qf;
    out.log_prior = 0.5 * ST * std::log(tau) - 0.5 * tau * qf;

  } else if (st.type == STType::TYPE_II) {
    // Structured time per spatial unit: the temporal precision applied to
    // delta[s,:] independently for each s.
    double total_qf = 0.0;
    for (int s = 0; s < S; s++) {
      const double* delta_s = &delta[s * T];
      double* grad_s = &grad_delta[s * T];
      if (st.temporal_type == TemporalType::RW1) {
        double qf = 0.0;
        for (int t = 0; t < T; t++) {
          double g = 0.0;
          if (t > 0) { g += tau * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
          if (t < T - 1) g += tau * (delta_s[t+1] - delta_s[t]);
          grad_s[t] = grad_delta_lik[s * T + t] + g;
        }
        total_qf += qf;
      } else if (st.temporal_type == TemporalType::RW2) {
        double qf = 0.0;
        for (int t = 0; t < T; t++) {
          double g = 0.0;
          if (t >= 2) g -= tau * (delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t]);
          if (t >= 1 && t < T - 1) g += 2.0 * tau * (delta_s[t-1] - 2.0*delta_s[t] + delta_s[t+1]);
          if (t < T - 2) g -= tau * (delta_s[t] - 2.0*delta_s[t+1] + delta_s[t+2]);
          grad_s[t] = grad_delta_lik[s * T + t] + g;
        }
        for (int t = 2; t < T; t++) qf += std::pow(delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t], 2);
        total_qf += qf;
      } else if (st.temporal_type == TemporalType::AR1 && T >= 2) {
        RATIOD_TLS_WORKSPACE(std::vector<double>, r_ds);
        r_ds.resize(T);
        ratiod_ar1::ar1_precision_apply(delta_s, T, rho, r_ds.data());
        double qf = 0.0;
        for (int t = 0; t < T; t++) {
          grad_s[t] = grad_delta_lik[s * T + t] - tau * r_ds[t];
          qf += delta_s[t] * r_ds[t];
        }
        total_qf += qf;
        double interior = 0.0;
        for (int t = 1; t < T - 1; t++) interior += delta_s[t] * delta_s[t];
        double cross = 0.0;
        for (int t = 1; t < T; t++) cross += delta_s[t] * delta_s[t-1];
        grad_rho += -0.5 * tau * ratiod_ar1::ar1_quadratic_form_drho(interior, cross, rho);
      } else {
        for (int t = 0; t < T; t++) grad_s[t] = grad_delta_lik[s * T + t];
      }
    }
    const int rank_per_unit = st_time_rank(st.temporal_type, T, st.temporal_cyclic);
    out.log_tau += 0.5 * S * rank_per_unit - 0.5 * tau * total_qf;
    out.log_prior = 0.5 * S * rank_per_unit * std::log(tau) - 0.5 * tau * total_qf;
    if (st.temporal_type == TemporalType::AR1 && T >= 2) {
      // One log|R(rho)| per spatial unit.
      out.log_prior += 0.5 * S * std::log(ratiod_ar1::one_minus_rho2(rho));
      grad_rho += 0.5 * S * ratiod_ar1::dlog_one_minus_rho2_drho(rho);
    }

  } else if (st.type == STType::TYPE_III) {
    // Structured space per time point: ICAR applied to delta[:,t].
    double total_qf = 0.0;
    for (int t = 0; t < T; t++) {
      for (int s = 0; s < S; s++) {
        double icar_grad = 0.0;
        for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
          const int j = st.adj_col_idx[idx] - 1;
          icar_grad += tau * (delta[j * T + t] - delta[s * T + t]);
        }
        grad_delta[s * T + t] = grad_delta_lik[s * T + t] + icar_grad;
      }
      for (int s = 0; s < S; s++) {
        for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
          const int j = st.adj_col_idx[idx] - 1;
          if (j > s) {
            const double diff = delta[s * T + t] - delta[j * T + t];
            total_qf += diff * diff;
          }
        }
      }
    }
    const int rank_spatial = S - 1;
    out.log_tau += 0.5 * T * rank_spatial - 0.5 * tau * total_qf;
    out.log_prior = 0.5 * T * rank_spatial * std::log(tau) - 0.5 * tau * total_qf;

  } else if (st.type == STType::SEPARABLE || st.type == STType::NONSEP_GP) {
    // A GP over the field's own index set, so tau enters as the marginal
    // variance sigma2 = 1/tau rather than through a GMRF stencil, and the two
    // ranges are parameters of the same prior. The Knorr-Held branches read a
    // Kronecker structure this one does not have; what they share is the
    // sum-to-zero block below, which the density applies to every type.
    for (int k = 0; k < ST; k++) grad_delta[k] = grad_delta_lik[k];
    const StGpGrad g =
        st_gp_nngp_grad(st, delta, 1.0 / tau, phi_space, phi_time, grad_delta);
    if (!g.ok) {
      // The structure was not filled, or a neighbour block would not factorize.
      // The density returns -inf at the same state, so the step is rejected;
      // what the field must not carry away from here is a partial prior.
      for (int k = 0; k < ST; k++) grad_delta[k] = grad_delta_lik[k];
      out.log_prior = -std::numeric_limits<double>::infinity();
      return out;
    }
    out.log_prior = g.log_prior;
    out.log_tau += g.log_tau;
    out.log_phi_space = g.log_phi_space;
    out.log_phi_time = g.log_phi_time;

  } else if (st.type == STType::TYPE_IV) {
    // Kronecker: Q_delta = tau * (Q_s (x) Q_t). Under the non-centered
    // reparameterization the stencil acts on z and carries no tau, the whole
    // precision having moved into delta = z / sqrt(tau).
    const double* x = use_nc ? z_or_delta : delta;
    const double qf_scale = use_nc ? 1.0 : tau;

    // Step 1: the time margin, v[s,t] = (Q_t x[s,:])_t.
    RATIOD_TLS_WORKSPACE(std::vector<double>, v);
    v.assign(S * T, 0.0);
    if (st.temporal_type == TemporalType::RW1) {
      for (int s = 0; s < S; s++) {
        for (int t = 0; t < T; t++) {
          double qt_val = 0.0;
          int n_t_neigh = 0;
          if (t > 0) { qt_val -= x[s * T + t - 1]; n_t_neigh++; }
          if (t < T - 1) { qt_val -= x[s * T + t + 1]; n_t_neigh++; }
          qt_val += n_t_neigh * x[s * T + t];
          v[s * T + t] = qt_val;
        }
      }
    } else if (st.temporal_type == TemporalType::RW2) {
      // Q_t[t,:] x is a linear combination of second differences; forming them
      // once per unit avoids recomputing each across the three t it serves.
      for (int s = 0; s < S; s++) {
        const double* d_s = &x[s * T];
        double* v_s = &v[s * T];
        if (T >= 3) {
          const int n_d2 = T - 2;
          RATIOD_TLS_WORKSPACE(std::vector<double>, d2_buf);
          d2_buf.resize(n_d2);
          double* d2 = d2_buf.data();
          for (int k = 0; k < n_d2; k++) {
            d2[k] = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
          }
          v_s[0] = d2[0];
          v_s[1] = -2.0 * d2[0];
          if (n_d2 > 1) v_s[1] += d2[1];
          for (int t = 2; t < T - 2; t++) {
            v_s[t] = d2[t - 2] - 2.0 * d2[t - 1] + d2[t];
          }
          if (T >= 4) {
            v_s[T - 2] = d2[n_d2 - 2] - 2.0 * d2[n_d2 - 1];
          } else {
            v_s[T - 2] = -2.0 * d2[0];
          }
          v_s[T - 1] = d2[n_d2 - 1];
        }
        // T < 3: no second differences exist, v stays zero.
      }
    } else if (st.temporal_type == TemporalType::AR1 && T >= 2) {
      for (int s = 0; s < S; s++) {
        ratiod_ar1::ar1_precision_apply(&x[s * T], T, rho, &v[s * T]);
      }
    }

    // Step 2: the space margin, (Q_s (x) Q_t) x.
    double total_qf = 0.0;
    for (int s = 0; s < S; s++) {
      for (int t = 0; t < T; t++) {
        double qs_v = 0.0;
        for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
          const int j = st.adj_col_idx[idx] - 1;
          qs_v -= v[j * T + t];
        }
        const int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
        qs_v += n_neigh * v[s * T + t];

        grad_delta[s * T + t] = use_nc
            ? (grad_delta_lik[s * T + t] * inv_scale - qs_v)
            : (grad_delta_lik[s * T + t] - tau * qs_v);
        total_qf += x[s * T + t] * qs_v;
      }
    }

    const int rank_space = S - 1;
    const int rank_time = st_time_rank(st.temporal_type, T, st.temporal_cyclic);
    const int total_rank = rank_space * rank_time;

    if (st.temporal_type == TemporalType::AR1 && T >= 2) {
      // The time margin is proper, so its log-determinant moves with rho.
      // Contracting dR/d(rho) against the Gram matrix [x[*,t1]' Q_s x[*,t2]]
      // needs only that matrix's interior diagonal and first off-diagonal,
      // which one extra spatial pass gives without forming it.
      RATIOD_TLS_WORKSPACE(std::vector<double>, u);
      u.assign(S * T, 0.0);
      for (int s = 0; s < S; s++) {
        for (int t = 0; t < T; t++) {
          double qs_x = 0.0;
          for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
            const int j = st.adj_col_idx[idx] - 1;
            qs_x -= x[j * T + t];
          }
          const int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
          qs_x += n_neigh * x[s * T + t];
          u[s * T + t] = qs_x;
        }
      }
      double interior = 0.0;
      for (int t = 1; t < T - 1; t++) {
        for (int s = 0; s < S; s++) interior += x[s * T + t] * u[s * T + t];
      }
      double cross = 0.0;
      for (int t = 1; t < T; t++) {
        for (int s = 0; s < S; s++) cross += x[s * T + t] * u[s * T + t - 1];
      }
      grad_rho += 0.5 * rank_space * ratiod_ar1::dlog_one_minus_rho2_drho(rho)
                - 0.5 * qf_scale * ratiod_ar1::ar1_quadratic_form_drho(interior, cross, rho);
    }

    if (use_nc) {
      // The combined GMRF normalization and non-centering Jacobian is
      // 0.5*(rank - ST)*log(tau); the field's own quadratic form is tau-free.
      double lik_tau_grad = 0.0;
      for (int k = 0; k < ST; k++) lik_tau_grad += grad_delta_lik[k] * delta[k];
      out.log_tau += 0.5 * (total_rank - ST) - 0.5 * lik_tau_grad;
      out.log_prior = -0.5 * total_qf + 0.5 * (total_rank - ST) * std::log(tau);
    } else {
      out.log_tau += 0.5 * total_rank - 0.5 * tau * total_qf;
      out.log_prior = 0.5 * total_rank * std::log(tau) - 0.5 * tau * total_qf;
    }
    if (st.temporal_type == TemporalType::AR1 && T >= 2) {
      out.log_prior += 0.5 * rank_space * std::log(ratiod_ar1::one_minus_rho2(rho));
    }
  }

  // -----------------------------------------------------------------------
  // Soft sum-to-zero on the reconstructed field. Each margin carries its own
  // constant, matching st_sum_to_zero_penalty: a space margin sums S terms at
  // one time, a time margin sums T times at one unit.
  // -----------------------------------------------------------------------
  {
    const double lambda_s = tulpa::s2z_precision(S);
    const double lambda_t = tulpa::s2z_precision(T);
    // Under the non-centered reparameterization the penalty is
    // -0.5*lambda*(inv_scale*sum(z))^2, so d/dz is the centered push scaled
    // once and the precision picks up the rest.
    const double stz_scale = use_nc ? inv_scale : 1.0;

    RATIOD_TLS_WORKSPACE(std::vector<double>, row_sums_buf);
    RATIOD_TLS_WORKSPACE(std::vector<double>, col_sums_buf);
    row_sums_buf.assign(T, 0.0);
    col_sums_buf.assign(S, 0.0);
    for (int s = 0; s < S; s++) {
      for (int t = 0; t < T; t++) {
        const double d = delta[s * T + t];
        row_sums_buf[t] += d;
        col_sums_buf[s] += d;
      }
    }

    for (int s = 0; s < S; s++) {
      for (int t = 0; t < T; t++) {
        grad_delta[s * T + t] -=
            stz_scale * (lambda_s * row_sums_buf[t] + lambda_t * col_sums_buf[s]);
      }
    }
    if (use_nc) {
      double tau_stz_grad = 0.0;
      for (int t = 0; t < T; t++) tau_stz_grad += lambda_s * row_sums_buf[t] * row_sums_buf[t];
      for (int s = 0; s < S; s++) tau_stz_grad += lambda_t * col_sums_buf[s] * col_sums_buf[s];
      tau_stz_grad *= 0.5;
      out.log_tau += tau_stz_grad;
    }

    for (int t = 0; t < T; t++) out.log_prior -= 0.5 * lambda_s * row_sums_buf[t] * row_sums_buf[t];
    for (int s = 0; s < S; s++) out.log_prior -= 0.5 * lambda_t * col_sums_buf[s] * col_sums_buf[s];
  }

  out.logit_rho += grad_rho * ratiod_ar1::drho_dlogit(rho);
  return out;
}

}  // namespace ratiod_spatiotemporal

#endif  // RATIOD_ST_PRIOR_GRAD_H
