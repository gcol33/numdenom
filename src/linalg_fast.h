// linalg_fast.h
// Fast linear algebra operations for ratiod HMC
// Uses cache-friendly algorithms and SIMD-friendly patterns

#ifndef QUOTR_LINALG_FAST_H
#define QUOTR_LINALG_FAST_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_linalg {

// ============================================================================
// Vector operations (SIMD-friendly)
// ============================================================================

// Dot product with loop unrolling
inline double dot_product(const double* x, const double* y, int n) {
  double sum = 0.0;
  int i = 0;

  // Process 4 elements at a time (SIMD-friendly)
  for (; i + 3 < n; i += 4) {
    sum += x[i] * y[i] + x[i+1] * y[i+1] +
           x[i+2] * y[i+2] + x[i+3] * y[i+3];
  }

  // Handle remaining elements
  for (; i < n; i++) {
    sum += x[i] * y[i];
  }

  return sum;
}

// Dot product with stride
inline double dot_product_strided(const double* x, int stride_x,
                                   const double* y, int stride_y, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i * stride_x] * y[i * stride_y];
  }
  return sum;
}

// Vector sum
inline double vector_sum(const double* x, int n) {
  double sum = 0.0;
  int i = 0;

  for (; i + 3 < n; i += 4) {
    sum += x[i] + x[i+1] + x[i+2] + x[i+3];
  }
  for (; i < n; i++) {
    sum += x[i];
  }

  return sum;
}

// Vector L2 norm squared
inline double norm_squared(const double* x, int n) {
  return dot_product(x, x, n);
}

// axpy: y = a*x + y
inline void axpy(double a, const double* x, double* y, int n) {
  int i = 0;
  for (; i + 3 < n; i += 4) {
    y[i] += a * x[i];
    y[i+1] += a * x[i+1];
    y[i+2] += a * x[i+2];
    y[i+3] += a * x[i+3];
  }
  for (; i < n; i++) {
    y[i] += a * x[i];
  }
}

// Scale vector: x = a*x
inline void scale(double a, double* x, int n) {
  int i = 0;
  for (; i + 3 < n; i += 4) {
    x[i] *= a;
    x[i+1] *= a;
    x[i+2] *= a;
    x[i+3] *= a;
  }
  for (; i < n; i++) {
    x[i] *= a;
  }
}

// ============================================================================
// Matrix-vector operations (row-major storage)
// ============================================================================

// Matrix-vector multiply: y = X * beta
// X is N x p stored row-major (X_flat[i*p + j] = X[i,j])
inline void matvec(const double* X_flat, const double* beta,
                   double* y, int N, int p) {

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    y[i] = dot_product(&X_flat[i * p], beta, p);
  }
}

// Matrix-vector multiply with accumulation: y += X * beta
inline void matvec_add(const double* X_flat, const double* beta,
                       double* y, int N, int p) {

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    y[i] += dot_product(&X_flat[i * p], beta, p);
  }
}

// Transposed matrix-vector multiply: y = X' * x
// Returns p-dimensional vector
inline void matvec_transpose(const double* X_flat, const double* x,
                             double* y, int N, int p) {

  // Initialize output to zero
  std::fill(y, y + p, 0.0);

  // Sequential accumulation (thread-safe without reduction)
  for (int i = 0; i < N; i++) {
    double xi = x[i];
    const double* row = &X_flat[i * p];
    for (int j = 0; j < p; j++) {
      y[j] += row[j] * xi;
    }
  }
}

// ============================================================================
// Batch linear predictor computation
// ============================================================================

// Compute linear predictors for all observations
// eta_num[i] = X_num[i,:] * beta_num
// eta_denom[i] = X_denom[i,:] * beta_denom
inline void compute_linear_predictors(
    const double* X_num_flat, const double* beta_num, int p_num,
    const double* X_denom_flat, const double* beta_denom, int p_denom,
    double* eta_num, double* eta_denom, int N, int n_threads = 1) {

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
  #endif
  for (int i = 0; i < N; i++) {
    eta_num[i] = dot_product(&X_num_flat[i * p_num], beta_num, p_num);
    eta_denom[i] = dot_product(&X_denom_flat[i * p_denom], beta_denom, p_denom);
  }
}

// ============================================================================
// Sparse operations for adjacency matrices
// ============================================================================

// Sparse matrix-vector multiply (CSR format)
// For ICAR: y = A * x where A is adjacency
inline void sparse_matvec_csr(
    const int* row_ptr, const int* col_idx, const double* values,
    const double* x, double* y, int n_rows) {

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < n_rows; i++) {
    double sum = 0.0;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      sum += values[k] * x[col_idx[k]];
    }
    y[i] = sum;
  }
}

// Sparse quadratic form: x' * L * x for Laplacian L
// L = D - A where D is diagonal of degrees, A is adjacency
// Uses: x'Lx = sum_edges (x_i - x_j)^2 for unweighted graph
inline double sparse_laplacian_quadform(
    const int* row_ptr, const int* col_idx,
    const double* x, int n_rows) {

  double quad_form = 0.0;

  // Sum over all edges (count each once)
  for (int i = 0; i < n_rows; i++) {
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      int j = col_idx[k];
      if (j > i) {  // Count each edge once
        double diff = x[i] - x[j];
        quad_form += diff * diff;
      }
    }
  }

  return quad_form;
}

// ============================================================================
// Memory-efficient operations
// ============================================================================

// Block processing for large datasets
// Processes data in chunks to improve cache efficiency
template<typename Func>
inline double block_reduce(int N, int block_size, Func f) {
  double total = 0.0;
  int n_blocks = (N + block_size - 1) / block_size;

  for (int b = 0; b < n_blocks; b++) {
    int start = b * block_size;
    int end = std::min(start + block_size, N);
    total += f(start, end);
  }

  return total;
}

// Parallel block reduce
template<typename Func>
inline double parallel_block_reduce(int N, int block_size, int n_threads, Func f) {
  double total = 0.0;
  int n_blocks = (N + block_size - 1) / block_size;

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:total) num_threads(n_threads)
  #endif
  for (int b = 0; b < n_blocks; b++) {
    int start = b * block_size;
    int end = std::min(start + block_size, N);
    total += f(start, end);
  }

  return total;
}

// ============================================================================
// Numerical utilities
// ============================================================================

// Log-sum-exp for numerical stability
inline double log_sum_exp(double a, double b) {
  double max_val = std::max(a, b);
  if (!std::isfinite(max_val)) return max_val;
  return max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
}

// Vectorized log-sum-exp
inline double log_sum_exp_vec(const double* x, int n) {
  if (n == 0) return -std::numeric_limits<double>::infinity();

  double max_val = *std::max_element(x, x + n);
  if (!std::isfinite(max_val)) return max_val;

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += std::exp(x[i] - max_val);
  }

  return max_val + std::log(sum);
}

// Softmax (in-place)
inline void softmax_inplace(double* x, int n) {
  double max_val = *std::max_element(x, x + n);
  double sum = 0.0;

  for (int i = 0; i < n; i++) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }

  for (int i = 0; i < n; i++) {
    x[i] /= sum;
  }
}

}  // namespace ratiod_linalg

#endif  // QUOTR_LINALG_FAST_H
