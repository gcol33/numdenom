// hmc_temporal_nc.h
// Non-centred coordinate for the intrinsic temporal walks.
//
// An augmented intrinsic precision factors as Q + 11'/n = A'A, where A stacks
// the walk's difference operator on 1'/sqrt(n): the augmentation IS that extra
// row. Sampling z ~ N(0, I) and setting phi = sigma * A^{-1} z therefore gives
// phi exactly the density the centred coordinate carries,
//
//     -0.5 * phi' (Q + 11'/n) phi / sigma2 - 0.5 * rank * log(2*pi*sigma2),
//
// because |A phi|^2 = sigma2 * |z|^2 and the Jacobian sigma^rank cancels the
// normalizer term for term. What changes is that sigma reaches the field
// through the transform rather than through a quadratic form, so the step size
// that holds the field stable no longer has to shrink with it.
//
// A non-cyclic RW2 annihilates a linear ramp as well as the constant, and only
// the constant is augmented (see rw2_log_lik), so A is (n-1) x n and one
// coordinate is left a free direction under the flat prior it already had.
// Q L = 0 makes the quadratic form blind to that direction, so it drops out of
// the identity above rather than complicating it.
//
// A^{-1} is a cumulative sum, once for RW1 and twice for RW2, so both the
// transform and its adjoint are O(n).

#ifndef RATIOD_HMC_TEMPORAL_NC_H
#define RATIOD_HMC_TEMPORAL_NC_H

#include <cmath>
#include <vector>

#include <tulpa/sum_to_zero.h>

namespace ratiod_temporal_nc {

// The linear null direction of a non-cyclic RW2, centred so that it is
// orthogonal to the constant and the sum coordinate below reads only the
// constant.
inline double lin_basis(int t, int n) {
  return static_cast<double>(t) - 0.5 * (static_cast<double>(n) - 1.0);
}

// Whether a walk of this length and order has a non-centred coordinate at all.
// Below it the centred quadratic form is empty and the two coordinates are the
// same block.
inline bool nc_applies(int n, int order, bool cyclic) {
  if (order == 1) return cyclic ? (n >= 3) : (n >= 2);
  return !cyclic && n >= 3;
}

// How many of the n coordinates carry the N(0, I) density. The remainder is
// the RW2's free linear direction, which keeps its flat prior.
inline int nc_normal_dim(int n, int order, bool cyclic) {
  if (!nc_applies(n, order, cyclic)) return 0;
  if (order == 1) return cyclic ? n : n;
  return n - 1;
}

// phi = sigma * A^{-1} z, with the RW2's linear coordinate added unscaled.
//
// Coordinate order, for a block of length n:
//   RW1 non-cyclic: z[0 .. n-2] increments,        z[n-1] sum
//   RW1 cyclic:     z[0 .. n-1] all of B^{-1}'s coordinates
//   RW2 non-cyclic: z[0 .. n-3] second differences, z[n-2] sum,
//                   z[n-1] the free linear direction (flat, unscaled)
template <typename T>
inline void rw_nc_forward(const T* z, int n, int order, bool cyclic,
                          const T& sigma, T* phi) {
  if (!nc_applies(n, order, cyclic)) {
    for (int t = 0; t < n; t++) phi[t] = z[t];
    return;
  }
  const double dn = static_cast<double>(n);
  const double sqrt_n = std::sqrt(dn);

  if (order == 1 && !cyclic) {
    // u_t = sum_{j < t} z_j, then the constant fixed by the sum coordinate.
    T run = T(0.0), sum_u = T(0.0);
    for (int t = 0; t < n; t++) {
      if (t > 0) run = run + z[t - 1];
      phi[t] = run;
      sum_u = sum_u + run;
    }
    const T a = (T(sqrt_n) * z[n - 1] - sum_u) / T(dn);
    for (int t = 0; t < n; t++) phi[t] = sigma * (phi[t] + a);
    return;
  }

  if (order == 1 && cyclic) {
    // B = C + 11'/n maps the constant to itself and acts as the circular
    // difference on the complement, so the constant part of z is the field's
    // mean and the rest integrates.
    T zbar = T(0.0);
    for (int t = 0; t < n; t++) zbar = zbar + z[t];
    zbar = zbar / T(dn);

    T run = T(0.0), sum_w = T(0.0);
    for (int t = 0; t < n; t++) {
      if (t > 0) run = run + (z[t] - zbar);
      phi[t] = run;
      sum_w = sum_w + run;
    }
    const T wbar = sum_w / T(dn);
    for (int t = 0; t < n; t++) phi[t] = sigma * (phi[t] - wbar + zbar);
    return;
  }

  // RW2, non-cyclic: v is the double integral with v_0 = v_1 = 0, the constant
  // comes from the sum coordinate, and the linear direction is carried by
  // z[n-1] at its own scale.
  //
  // Any right inverse of A reproduces the density, since A L = 0 makes the
  // quadratic form blind to what the map puts along L. It is not blind to the
  // CONDITIONING though: the double integral's particular solution carries a
  // large ramp, which z[n-1] can undo, so the two coordinates would describe
  // nearly the same direction of eta and the sampler would have to travel a
  // ridge between them. Projecting the ramp out leaves A v alone and the sum
  // alone (L is orthogonal to the constant) and makes z[n-1] the only
  // coordinate carrying it.
  T sum_v = T(0.0);
  for (int t = 0; t < n; t++) {
    if (t < 2) {
      phi[t] = T(0.0);
    } else {
      phi[t] = T(2.0) * phi[t - 1] - phi[t - 2] + z[t - 2];
    }
    sum_v = sum_v + phi[t];
  }
  T vL = T(0.0);
  double LL = 0.0;
  for (int t = 0; t < n; t++) {
    const double L = lin_basis(t, n);
    vL = vL + phi[t] * T(L);
    LL += L * L;
  }
  const T cL = vL / T(LL);
  for (int t = 0; t < n; t++) phi[t] = phi[t] - cL * T(lin_basis(t, n));

  const T a = (T(sqrt_n) * z[n - 2] - sum_v) / T(dn);
  for (int t = 0; t < n; t++) {
    phi[t] = sigma * (phi[t] + a) + z[n - 1] * T(lin_basis(t, n));
  }
}

// dL/dz and dL/d(log sigma2) from dL/dphi.
//
// `g` is already whatever projection the caller's eta assembly implies, so this
// is the transform's adjoint alone. `phi` is the field the forward pass built,
// which the sigma derivative needs: phi is sigma-linear except in the free
// linear direction, so d(phi)/d(log sigma2) = 0.5 * (phi - linear part).
inline void rw_nc_backward(const double* g, const double* phi, const double* z,
                           int n, int order, bool cyclic, double sigma,
                           double* gz, double* g_log_sigma2) {
  if (!nc_applies(n, order, cyclic)) {
    for (int t = 0; t < n; t++) gz[t] += g[t];
    return;
  }
  const double dn = static_cast<double>(n);
  const double sqrt_n = std::sqrt(dn);

  double gsum = 0.0;
  for (int t = 0; t < n; t++) gsum += g[t];

  if (order == 1 && !cyclic) {
    // S1(j) = sum_{t >= j} g_t, so d(u_t)/d(z_k) = 1[t > k] gives S1(k+1), and
    // the constant's share is the count (n - k - 1) spread over n.
    double tail = 0.0;
    std::vector<double> S1(n + 1, 0.0);
    for (int t = n - 1; t >= 0; t--) { tail += g[t]; S1[t] = tail; }
    for (int k = 0; k <= n - 2; k++) {
      gz[k] += sigma * (S1[k + 1] - (dn - k - 1.0) / dn * gsum);
    }
    gz[n - 1] += sigma * gsum / sqrt_n;
  } else if (order == 1 && cyclic) {
    // Solve B' y = g: the constant part of y is mean(g), and on the complement
    // (C' y)_k = y_k - y_{k+1} integrates backwards.
    const double gbar = gsum / dn;
    std::vector<double> y(n, 0.0);
    double run = 0.0;
    for (int k = n - 2; k >= 0; k--) { run += (g[k] - gbar); y[k] = run; }
    double ybar = 0.0;
    for (int k = 0; k < n; k++) ybar += y[k];
    ybar /= dn;
    for (int k = 0; k < n; k++) gz[k] += sigma * (y[k] - ybar + gbar);
  } else {
    // RW2: d(v_t)/d(z_k) = (t - k - 1) for t >= k + 2, a ramp, so the weighted
    // tail sums S1 and S2 give it in one pass rather than n.
    std::vector<double> S1(n + 2, 0.0), S2(n + 2, 0.0);
    for (int t = n - 1; t >= 0; t--) {
      S1[t] = S1[t + 1] + g[t];
      S2[t] = S2[t + 1] + static_cast<double>(t) * g[t];
    }
    // The ramp's own projection onto L, which the forward pass removed: its
    // inner products need the same two tail sums taken against L.
    std::vector<double> L1(n + 2, 0.0), L2(n + 2, 0.0);
    double LL = 0.0, gL = 0.0;
    for (int t = n - 1; t >= 0; t--) {
      const double L = lin_basis(t, n);
      L1[t] = L1[t + 1] + L;
      L2[t] = L2[t + 1] + static_cast<double>(t) * L;
      LL += L * L;
      gL += g[t] * L;
    }
    for (int k = 0; k <= n - 3; k++) {
      const int j = k + 2;
      const double ramp_g = S2[j] - (k + 1.0) * S1[j];
      // <d(v)/d(z_k), L>, the coefficient the forward pass subtracted
      const double ramp_L = L2[j] - (k + 1.0) * L1[j];
      // sum_t d(v_t)/d(z_k) = M(M+1)/2 with M = n - k - 2, unchanged by the
      // projection because L sums to zero
      const double m = dn - k - 2.0;
      const double ramp_sum = 0.5 * m * (m + 1.0);
      gz[k] += sigma * (ramp_g - ramp_L / LL * gL - ramp_sum / dn * gsum);
    }
    gz[n - 2] += sigma * gsum / sqrt_n;
    double glin = 0.0;
    for (int t = 0; t < n; t++) glin += g[t] * lin_basis(t, n);
    gz[n - 1] += glin;
  }

  if (g_log_sigma2 != nullptr) {
    // phi is sigma-linear apart from the free linear direction, and
    // d(sigma)/d(log sigma2) = 0.5 * sigma.
    double dot = 0.0;
    const bool has_lin = (order == 2 && !cyclic);
    for (int t = 0; t < n; t++) {
      const double scaled =
          has_lin ? (phi[t] - z[n - 1] * lin_basis(t, n)) : phi[t];
      dot += g[t] * scaled;
    }
    *g_log_sigma2 += 0.5 * dot;
  }
}

// =====================================================================
// Several walks under one augmented constant
// =====================================================================
//
// With G temporal groups the prior is blockdiag(Q_g) + (1/N) 11', N = G*T: one
// global constant is augmented, not one per group, so G - 1 group contrasts
// stay improper and the transform has to carry them beside the scaled
// directions. Running the single-group primitive per group would augment each
// group's constant instead, which is a different model.
//
// The primitive already isolates the direction that has to be re-mixed. Its
// coordinate splits into the walk's increments and ONE sum coordinate c, with
//
//     sum_t phi_t = sigma * sqrt(T) * c,
//
// and the walk's quadratic form is blind to c, so a group's increments carry
// |z_inc|^2 whatever c is. Collect the G sum coordinates into a vector c, give
// every group the same primitive sum coordinate cbar = mean(c), and add the
// deviation (c_g - cbar) / sqrt(T) to group g as an unscaled constant. That is
// the orthogonal split of c into the global direction m = sqrt(G) * cbar and
// the G - 1 contrasts, written without materializing a Helmert basis: the
// deviations ARE the contrast coordinates read in the group basis.
//
// The contrasts are orthogonal to 1 and so drop out of the global sum, which
// is sigma * sqrt(T*G) * m. With N = G*T,
//
//     tau * global_aug_quad = tau * (1/N) (global sum)^2 = m^2,
//     tau * sum_g |D_g phi_g|^2 = sum_g |z_g,inc|^2,
//
// and the scaled coordinates number G * rank(Q_g) + 1, the normalizer's rank,
// so the Jacobian cancels as it does for one group. The contrasts are left
// unscaled: they are flat directions, and scaling them would couple the group
// levels to tau where the centred coordinate does not. G = 1 leaves the
// primitive's coordinate untouched.

// The index within a block that carries its sum coordinate, or -1 when the
// coordinate is not a single entry: a cyclic walk inverts B = C + 11'/n, whose
// constant coordinate is the block's mean.
inline int nc_sum_slot(int n, int order, bool cyclic) {
  if (!nc_applies(n, order, cyclic) || cyclic) return -1;
  return (order == 2) ? n - 2 : n - 1;
}

// The block's sum coordinate c, the one direction of z the transform turns
// into the block's total: sum_t phi_t = sigma * sqrt(n) * c.
template <typename T>
inline T nc_sum_coord(const T* z, int n, int order, bool cyclic) {
  const int slot = nc_sum_slot(n, order, cyclic);
  if (slot >= 0) return z[slot];
  T s = T(0.0);
  for (int t = 0; t < n; t++) s = s + z[t];
  return s / T(std::sqrt(static_cast<double>(n)));
}

// `z` with its sum coordinate set to `c` and every other direction untouched.
template <typename T>
inline void nc_with_sum_coord(const T* z, int n, int order, bool cyclic,
                              const T& c, T* out) {
  for (int t = 0; t < n; t++) out[t] = z[t];
  const int slot = nc_sum_slot(n, order, cyclic);
  if (slot >= 0) {
    out[slot] = c;
    return;
  }
  const T shift = (c - nc_sum_coord(z, n, order, cyclic)) /
                  T(std::sqrt(static_cast<double>(n)));
  for (int t = 0; t < n; t++) out[t] = out[t] + shift;
}

// |z|^2 over the block's scaled coordinates other than the sum one: the
// increments the walk's own quadratic form reads.
template <typename T>
inline T nc_increment_quad(const T* z, int n, int order, bool cyclic) {
  const int d = nc_normal_dim(n, order, cyclic);
  if (d == 0) return T(0.0);
  const int slot = nc_sum_slot(n, order, cyclic);
  T quad = T(0.0);
  if (slot >= 0) {
    for (int k = 0; k < d; k++) {
      if (k != slot) quad = quad + z[k] * z[k];
    }
    return quad;
  }
  T mean = T(0.0);
  for (int t = 0; t < n; t++) mean = mean + z[t];
  mean = mean / T(static_cast<double>(n));
  for (int t = 0; t < n; t++) {
    const T dev = z[t] - mean;
    quad = quad + dev * dev;
  }
  return quad;
}

// How many of the G*n coordinates carry the N(0, I) density: every group's
// increments plus the one global direction the augmentation fills.
inline int nc_grouped_normal_dim(int n, int n_groups, int order, bool cyclic) {
  const int d = nc_normal_dim(n, order, cyclic);
  if (d == 0) return 0;
  return n_groups * (d - 1) + 1;
}

// phi = sigma * A^{-1} z over G blocks of length n sharing one augmented
// constant.
template <typename T>
inline void rw_nc_grouped_forward(const T* z, int n, int n_groups, int order,
                                  bool cyclic, const T& sigma, T* phi) {
  if (!nc_applies(n, order, cyclic) || n_groups <= 1) {
    for (int g = 0; g < n_groups; g++) {
      rw_nc_forward(z + g * n, n, order, cyclic, sigma, phi + g * n);
    }
    return;
  }
  std::vector<T> c(n_groups, T(0.0));
  T cbar = T(0.0);
  for (int g = 0; g < n_groups; g++) {
    c[g] = nc_sum_coord(z + g * n, n, order, cyclic);
    cbar = cbar + c[g];
  }
  cbar = cbar / T(static_cast<double>(n_groups));

  const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
  std::vector<T> zg(n, T(0.0));
  for (int g = 0; g < n_groups; g++) {
    nc_with_sum_coord(z + g * n, n, order, cyclic, cbar, zg.data());
    rw_nc_forward(zg.data(), n, order, cyclic, sigma, phi + g * n);
    const T off = (c[g] - cbar) * T(inv_sqrt_n);
    for (int t = 0; t < n; t++) phi[g * n + t] = phi[g * n + t] + off;
  }
}

// log N(z; 0, I) over the coordinates that carry it, for G blocks: every
// group's increments and the global direction m = sqrt(G) * cbar. The G - 1
// contrasts and a non-cyclic RW2's per-group linear direction keep the flat
// prior the centred coordinate gives them.
template <typename T>
inline T rw_nc_grouped_log_prior(const T* z, int n, int n_groups, int order,
                                 bool cyclic) {
  const int d = nc_grouped_normal_dim(n, n_groups, order, cyclic);
  if (d == 0) return T(0.0);
  T quad = T(0.0), cbar = T(0.0);
  for (int g = 0; g < n_groups; g++) {
    quad = quad + nc_increment_quad(z + g * n, n, order, cyclic);
    cbar = cbar + nc_sum_coord(z + g * n, n, order, cyclic);
  }
  cbar = cbar / T(static_cast<double>(n_groups));
  quad = quad + T(static_cast<double>(n_groups)) * cbar * cbar;
  return T(-0.5) * quad - T(0.5 * d) * T(std::log(2.0 * M_PI));
}

// d(rw_nc_grouped_log_prior)/dz, added to `gz`.
inline void rw_nc_grouped_log_prior_grad(const double* z, int n, int n_groups,
                                         int order, bool cyclic, double* gz) {
  if (nc_grouped_normal_dim(n, n_groups, order, cyclic) == 0) return;
  const int d = nc_normal_dim(n, order, cyclic);
  const int slot = nc_sum_slot(n, order, cyclic);
  const double dn = static_cast<double>(n);

  double cbar = 0.0;
  for (int g = 0; g < n_groups; g++) {
    cbar += nc_sum_coord(z + g * n, n, order, cyclic);
  }
  cbar /= static_cast<double>(n_groups);

  for (int g = 0; g < n_groups; g++) {
    const double* zg = z + g * n;
    double* out = gz + g * n;
    if (slot >= 0) {
      for (int k = 0; k < d; k++) out[k] += (k == slot) ? -cbar : -zg[k];
      continue;
    }
    // Cyclic: the sum coordinate is the block's mean, so the increments are
    // what is left once it is removed and the global direction reads it.
    double mean = 0.0;
    for (int t = 0; t < n; t++) mean += zg[t];
    mean /= dn;
    for (int t = 0; t < n; t++) out[t] += -(zg[t] - mean) - cbar / std::sqrt(dn);
  }
}

// dL/dz and dL/d(log sigma2) from dL/dphi, for G blocks sharing one augmented
// constant. The re-mixing is linear, so its adjoint is the primitive's adjoint
// per block followed by the transpose of the split: a sum coordinate collects
// the average of what the blocks send it, and the unscaled offset sends back
// the deviation of its own block's likelihood sum.
//
// The block's own field is rebuilt here rather than read from the caller. The
// sigma derivative is a dot product against the scaled part of phi, and the
// unscaled offset the re-mixing adds is a per-group constant, which a
// per-group likelihood sum does not annihilate the way the global one does.
inline void rw_nc_grouped_backward(const double* g, const double* z, int n,
                                   int n_groups, int order, bool cyclic,
                                   double sigma, double* gz,
                                   double* g_log_sigma2) {
  const double dn = static_cast<double>(n);
  const double dG = static_cast<double>(n_groups);
  const double sqrt_n = std::sqrt(dn);
  const int slot = nc_sum_slot(n, order, cyclic);
  const bool grouped = nc_applies(n, order, cyclic) && n_groups > 1;

  double cbar = 0.0;
  if (grouped) {
    for (int gg = 0; gg < n_groups; gg++) {
      cbar += nc_sum_coord(z + gg * n, n, order, cyclic);
    }
    cbar /= dG;
  }

  std::vector<double> ghat(static_cast<size_t>(n) * n_groups, 0.0);
  std::vector<double> zg(n, 0.0), phig(n, 0.0), lik_sum(n_groups, 0.0);
  double lik_sum_bar = 0.0;
  for (int gg = 0; gg < n_groups; gg++) {
    if (grouped) {
      nc_with_sum_coord(z + gg * n, n, order, cyclic, cbar, zg.data());
    } else {
      for (int t = 0; t < n; t++) zg[t] = z[gg * n + t];
    }
    rw_nc_forward(zg.data(), n, order, cyclic, sigma, phig.data());
    double s = 0.0;
    for (int t = 0; t < n; t++) s += g[gg * n + t];
    lik_sum[gg] = s;
    lik_sum_bar += s;
    rw_nc_backward(g + gg * n, phig.data(), zg.data(), n, order, cyclic, sigma,
                   ghat.data() + gg * n, g_log_sigma2);
  }
  if (!grouped) {
    for (int k = 0; k < n * n_groups; k++) gz[k] += ghat[k];
    return;
  }
  lik_sum_bar /= dG;

  if (slot >= 0) {
    double slot_total = 0.0;
    for (int gg = 0; gg < n_groups; gg++) slot_total += ghat[gg * n + slot];
    for (int gg = 0; gg < n_groups; gg++) {
      for (int k = 0; k < n; k++) {
        if (k != slot) gz[gg * n + k] += ghat[gg * n + k];
      }
      gz[gg * n + slot] +=
          slot_total / dG + (lik_sum[gg] - lik_sum_bar) / sqrt_n;
    }
    return;
  }

  std::vector<double> mean_g(n_groups, 0.0);
  double mean_bar = 0.0;
  for (int gg = 0; gg < n_groups; gg++) {
    double s = 0.0;
    for (int t = 0; t < n; t++) s += ghat[gg * n + t];
    mean_g[gg] = s / dn;
    mean_bar += mean_g[gg];
  }
  mean_bar /= dG;
  for (int gg = 0; gg < n_groups; gg++) {
    const double off_share = (lik_sum[gg] - lik_sum_bar) / dn;
    for (int t = 0; t < n; t++) {
      gz[gg * n + t] += ghat[gg * n + t] - mean_g[gg] + mean_bar + off_share;
    }
  }
}

// log N(z; 0, I) over the coordinates that carry it. The free linear direction
// of a non-cyclic RW2 keeps the flat prior the centred coordinate gave it, so
// it contributes nothing here.
template <typename T>
inline T rw_nc_log_prior(const T* z, int n, int order, bool cyclic) {
  const int d = nc_normal_dim(n, order, cyclic);
  T quad = T(0.0);
  for (int k = 0; k < d; k++) quad = quad + z[k] * z[k];
  return T(-0.5) * quad - T(0.5 * d) * T(std::log(2.0 * M_PI));
}

}  // namespace ratiod_temporal_nc

#endif  // RATIOD_HMC_TEMPORAL_NC_H
