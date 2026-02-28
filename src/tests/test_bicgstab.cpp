// Test Conjugate Gradient (CG) iterative solver for SPD systems
// Compile: g++ -O2 -o test_cg test_cg.cpp -I..

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "../linalg_fast.h"

using namespace ratiod_linalg;

// Simple dense matrix-vector product for testing
// A is stored as vector<vector<double>>
template<typename MatType>
auto make_dense_matvec(const MatType& A, int n) {
  return [&A, n](const double* x, double* y) {
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        sum += A[i][j] * x[j];
      }
      y[i] = sum;
    }
  };
}

// Generate a random SPD matrix for testing
std::vector<std::vector<double>> make_spd_matrix(int n, double cond = 10.0) {
  // Create a symmetric matrix with controlled eigenvalues
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));

  // Start with diagonal matrix with eigenvalues from 1 to cond
  for (int i = 0; i < n; i++) {
    double lambda = 1.0 + (cond - 1.0) * i / (n - 1);
    A[i][i] = lambda;
  }

  // Apply random orthogonal transformation (simplified: just mix rows/cols)
  // For simplicity, just add small off-diagonal terms symmetrically
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double val = 0.1 * std::sin(i * 0.3 + j * 0.7);
      A[i][j] = val;
      A[j][i] = val;
    }
  }

  // Ensure strong diagonal dominance
  for (int i = 0; i < n; i++) {
    double row_sum = 0.0;
    for (int j = 0; j < n; j++) {
      if (i != j) row_sum += std::abs(A[i][j]);
    }
    A[i][i] = std::max(A[i][i], row_sum + 1.0);
  }

  return A;
}

// Direct solve using Cholesky for comparison
std::vector<double> cholesky_solve(const std::vector<std::vector<double>>& A,
                                    const std::vector<double>& b) {
  int n = static_cast<int>(b.size());
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

  // Cholesky decomposition: A = L * L^T
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      if (j == i) {
        for (int k = 0; k < j; k++) {
          sum += L[j][k] * L[j][k];
        }
        L[j][j] = std::sqrt(A[j][j] - sum);
      } else {
        for (int k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }

  // Forward substitution: L * y = b
  std::vector<double> y(n);
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = (b[i] - sum) / L[i][i];
  }

  // Backward substitution: L^T * x = y
  std::vector<double> x(n);
  for (int i = n - 1; i >= 0; i--) {
    double sum = 0.0;
    for (int j = i + 1; j < n; j++) {
      sum += L[j][i] * x[j];
    }
    x[i] = (y[i] - sum) / L[i][i];
  }

  return x;
}

void test_cg_basic(int n) {
  std::cout << "\n=== Test CG with n=" << n << " ===" << std::endl;

  // Create SPD matrix
  auto A = make_spd_matrix(n);

  // Create RHS b = A * ones
  std::vector<double> x_true(n, 1.0);
  std::vector<double> b(n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      b[i] += A[i][j] * x_true[j];
    }
  }

  // Solve with CG
  auto A_func = make_dense_matvec(A, n);

  auto start = std::chrono::high_resolution_clock::now();
  auto result = cg_solve(A_func, b);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Compute error
  double max_err = 0.0;
  for (int i = 0; i < n; i++) {
    max_err = std::max(max_err, std::abs(result.x[i] - x_true[i]));
  }

  std::cout << "  Converged: " << (result.converged ? "yes" : "NO") << std::endl;
  std::cout << "  Iterations: " << result.iterations << std::endl;
  std::cout << "  Residual: " << result.residual_norm << std::endl;
  std::cout << "  Max error vs true: " << max_err << std::endl;
  std::cout << "  Time: " << duration.count() << " us" << std::endl;

  // Compare with Cholesky
  start = std::chrono::high_resolution_clock::now();
  auto x_chol = cholesky_solve(A, b);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double chol_err = 0.0;
  for (int i = 0; i < n; i++) {
    chol_err = std::max(chol_err, std::abs(x_chol[i] - x_true[i]));
  }
  std::cout << "  Cholesky max error: " << chol_err << " (time: " << duration.count() << " us)" << std::endl;
}

void test_gp_kernel_solve(int N) {
  std::cout << "\n=== Test GP Kernel Solve with N=" << N << " ===" << std::endl;

  // Create random coordinates
  std::vector<double> coords(2 * N);
  for (int i = 0; i < N; i++) {
    coords[2*i] = std::sin(i * 0.1) * 10.0;
    coords[2*i + 1] = std::cos(i * 0.1) * 10.0;
  }

  double sigma_sq = 1.0;
  double lengthscale = 2.0;

  // Create kernel matvec function
  auto K_func = make_se_kernel_matvec(coords.data(), N, sigma_sq, lengthscale);

  // Create RHS (random)
  std::vector<double> b(N);
  for (int i = 0; i < N; i++) {
    b[i] = std::sin(i * 0.5);
  }

  // Solve with CG
  auto start = std::chrono::high_resolution_clock::now();
  auto result = cg_solve(K_func, b, {}, 1e-6, 500);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Verify: compute K * x and compare to b
  std::vector<double> Kx(N);
  K_func(result.x.data(), Kx.data());

  double residual_check = 0.0;
  for (int i = 0; i < N; i++) {
    residual_check += (b[i] - Kx[i]) * (b[i] - Kx[i]);
  }
  residual_check = std::sqrt(residual_check);

  std::cout << "  Converged: " << (result.converged ? "yes" : "NO") << std::endl;
  std::cout << "  Iterations: " << result.iterations << std::endl;
  std::cout << "  Reported residual: " << result.residual_norm << std::endl;
  std::cout << "  Verified residual: " << residual_check << std::endl;
  std::cout << "  Time: " << duration.count() << " ms" << std::endl;
}

void test_pcg_with_diagonal_precond(int n) {
  std::cout << "\n=== Test PCG with diagonal preconditioner, n=" << n << " ===" << std::endl;

  auto A = make_spd_matrix(n, 100.0);  // Higher condition number

  std::vector<double> x_true(n, 1.0);
  std::vector<double> b(n, 0.0);
  std::vector<double> diag(n);
  for (int i = 0; i < n; i++) {
    diag[i] = A[i][i];
    for (int j = 0; j < n; j++) {
      b[i] += A[i][j] * x_true[j];
    }
  }

  auto A_func = make_dense_matvec(A, n);
  auto M_solve = make_diagonal_precond(diag.data(), n);

  // CG without preconditioning
  auto start = std::chrono::high_resolution_clock::now();
  auto result_cg = cg_solve(A_func, b, {}, 1e-8, 1000);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_cg = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // PCG with diagonal preconditioning
  start = std::chrono::high_resolution_clock::now();
  auto result_pcg = pcg_solve(A_func, M_solve, b, {}, 1e-8, 1000);
  end = std::chrono::high_resolution_clock::now();
  auto duration_pcg = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "  CG:  iter=" << result_cg.iterations
            << ", residual=" << result_cg.residual_norm
            << ", time=" << duration_cg.count() << "us" << std::endl;
  std::cout << "  PCG: iter=" << result_pcg.iterations
            << ", residual=" << result_pcg.residual_norm
            << ", time=" << duration_pcg.count() << "us" << std::endl;
}

int main() {
  std::cout << "Conjugate Gradient (CG) Solver Tests for SPD Systems" << std::endl;
  std::cout << "=====================================================" << std::endl;

  // Basic correctness tests
  test_cg_basic(50);
  test_cg_basic(100);

  // Test with preconditioning
  test_pcg_with_diagonal_precond(100);

  // GP kernel test
  test_gp_kernel_solve(100);
  test_gp_kernel_solve(200);

  std::cout << "\n=== All tests completed ===" << std::endl;
  return 0;
}
