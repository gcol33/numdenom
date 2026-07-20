// Hard sum-to-zero constraint for intrinsic (rank-deficient) spatial fields.
//
// An ICAR precision Q satisfies Q1 = 0, so the field's mean is unidentified
// and trades off against the intercept. Penalising sum(phi) softly cannot
// resolve this: the penalty puts precision lambda*J on the constant direction
// while the field's own directions sit at tau, and the resulting stiffness
// forces the NUTS step size down by sqrt(lambda*J/tau). Tight enough to
// identify and loose enough to sample are incompatible requirements.
//
// Instead the constant direction is removed rather than penalised. The
// sampled vector phi_raw is centred before it reaches the likelihood, so the
// likelihood cannot see its mean, and the intercept carries the level exactly.
// The freed direction is then given the field's own scale by using the proper
// precision tau*(Q + 11'/J) in place of tau*Q:
//
//   phi_raw' (Q + 11'/J) phi_raw = phi_raw' Q phi_raw + (sum phi_raw)^2 / J
//
// Q1 = 0 makes the quadratic form identical for phi_raw and its centred
// version, so only the second term is new. That matrix is full rank with
// log-determinant contribution J/2 * log(tau) rather than (J-1)/2, and its
// constant-direction precision is tau -- the same order as everything else,
// which is what removes the stiffness.
//
// This is the construction behind INLA's constr=TRUE and Stan's
// sum_to_zero_vector.

#ifndef RATIOD_SPATIAL_FIELD_CONSTRAINT_H
#define RATIOD_SPATIAL_FIELD_CONSTRAINT_H

#include <vector>

namespace ratiod_constraints {

// Centres the spatial block of a parameter vector in place and returns the raw
// sum. Applied once at a function's boundary, so that every downstream reader
// sees the constrained field -- including vectorized fast paths that index
// params directly rather than through a bound pointer. Centring at individual
// binding sites instead leaves those paths reading raw values.
template <typename T>
inline T center_spatial_block(std::vector<T>& params, int start, int n) {
  T sum = T(0);
  for (int j = 0; j < n; j++) sum = sum + params[start + j];
  const T mean = sum / T(n);
  for (int j = 0; j < n; j++) params[start + j] = params[start + j] - mean;
  return sum;
}

// Quadratic-form contribution of the freed constant direction: sum^2 / n.
template <typename T>
inline T free_direction_quad(const T& raw_sum, int n) {
  return raw_sum * raw_sum / T(n);
}

// d/d(phi_raw[s]) of -0.5 * scale * sum^2 / n, identical for every s.
template <typename T>
inline T free_direction_grad(const T& raw_sum, int n, const T& scale) {
  return -scale * raw_sum / T(n);
}

// The likelihood sees only the centred field, so the gradient with respect to
// the raw parameters is the projection (I - 11'/n) applied to the likelihood
// gradient: subtract its mean.
inline void project_gradient(double* g, int n) {
  double mean = 0.0;
  for (int i = 0; i < n; i++) mean += g[i];
  mean /= static_cast<double>(n);
  for (int i = 0; i < n; i++) g[i] -= mean;
}

}  // namespace ratiod_constraints

#endif  // RATIOD_SPATIAL_FIELD_CONSTRAINT_H
