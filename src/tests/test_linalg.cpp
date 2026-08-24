// test_linalg.cpp
// Catch2 tests for linalg_fast.h

#include "catch.hpp"
#include "../linalg_fast.h"
#include <vector>
#include <cmath>

using namespace ratiod_linalg;

TEST_CASE("dot_product computes correct inner product", "[linalg]") {
  SECTION("simple vectors") {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    REQUIRE(dot_product(x, y, 3) == Approx(32.0));  // 1*4 + 2*5 + 3*6 = 32
  }

  SECTION("single element") {
    double x[] = {3.0};
    double y[] = {4.0};
    REQUIRE(dot_product(x, y, 1) == Approx(12.0));
  }

  SECTION("longer vector (tests unrolling)") {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double y[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    REQUIRE(dot_product(x, y, 8) == Approx(36.0));  // sum(1:8) = 36
  }

  SECTION("zeros") {
    double x[] = {0.0, 0.0, 0.0};
    double y[] = {1.0, 2.0, 3.0};
    REQUIRE(dot_product(x, y, 3) == Approx(0.0));
  }
}

TEST_CASE("vector_sum computes correct sum", "[linalg]") {
  SECTION("simple sum") {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    REQUIRE(vector_sum(x, 5) == Approx(15.0));
  }

  SECTION("longer vector") {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    REQUIRE(vector_sum(x, 10) == Approx(55.0));
  }
}

TEST_CASE("norm_squared computes squared L2 norm", "[linalg]") {
  SECTION("unit vector") {
    double x[] = {1.0, 0.0, 0.0};
    REQUIRE(norm_squared(x, 3) == Approx(1.0));
  }

  SECTION("3-4-5 triangle") {
    double x[] = {3.0, 4.0};
    REQUIRE(norm_squared(x, 2) == Approx(25.0));
  }
}

TEST_CASE("axpy performs y = a*x + y", "[linalg]") {
  SECTION("simple case") {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    axpy(2.0, x, y, 3);
    REQUIRE(y[0] == Approx(6.0));   // 2*1 + 4
    REQUIRE(y[1] == Approx(9.0));   // 2*2 + 5
    REQUIRE(y[2] == Approx(12.0));  // 2*3 + 6
  }

  SECTION("longer vector") {
    double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    axpy(3.0, x, y, 8);
    for (int i = 0; i < 8; i++) {
      REQUIRE(y[i] == Approx(3.0));
    }
  }
}

TEST_CASE("scale multiplies vector by scalar", "[linalg]") {
  SECTION("simple case") {
    double x[] = {1.0, 2.0, 3.0};
    scale(2.0, x, 3);
    REQUIRE(x[0] == Approx(2.0));
    REQUIRE(x[1] == Approx(4.0));
    REQUIRE(x[2] == Approx(6.0));
  }

  SECTION("scale by zero") {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    scale(0.0, x, 5);
    for (int i = 0; i < 5; i++) {
      REQUIRE(x[i] == Approx(0.0));
    }
  }
}

TEST_CASE("log_sum_exp is numerically stable", "[linalg]") {
  SECTION("equal values") {
    double result = log_sum_exp(0.0, 0.0);
    REQUIRE(result == Approx(std::log(2.0)));
  }

  SECTION("large difference") {
    double result = log_sum_exp(100.0, 0.0);
    REQUIRE(result == Approx(100.0).margin(1e-10));
  }

  SECTION("negative values") {
    double result = log_sum_exp(-1.0, -2.0);
    double expected = std::log(std::exp(-1.0) + std::exp(-2.0));
    REQUIRE(result == Approx(expected));
  }
}

TEST_CASE("log_sum_exp_vec handles vectors", "[linalg]") {
  SECTION("single element") {
    double x[] = {5.0};
    REQUIRE(log_sum_exp_vec(x, 1) == Approx(5.0));
  }

  SECTION("multiple elements") {
    double x[] = {1.0, 2.0, 3.0};
    double expected = std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0));
    REQUIRE(log_sum_exp_vec(x, 3) == Approx(expected));
  }
}

TEST_CASE("softmax_inplace normalizes to probability simplex", "[linalg]") {
  SECTION("simple case") {
    double x[] = {1.0, 2.0, 3.0};
    softmax_inplace(x, 3);

    // Should sum to 1
    double sum = x[0] + x[1] + x[2];
    REQUIRE(sum == Approx(1.0));

    // Larger values should have higher probability
    REQUIRE(x[2] > x[1]);
    REQUIRE(x[1] > x[0]);
  }

  SECTION("equal inputs give uniform") {
    double x[] = {0.0, 0.0, 0.0};
    softmax_inplace(x, 3);
    REQUIRE(x[0] == Approx(1.0/3.0));
    REQUIRE(x[1] == Approx(1.0/3.0));
    REQUIRE(x[2] == Approx(1.0/3.0));
  }
}

TEST_CASE("matvec computes matrix-vector product", "[linalg]") {
  SECTION("2x2 matrix") {
    // Row-major: [[1, 2], [3, 4]]
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double beta[] = {1.0, 1.0};
    double y[] = {0.0, 0.0};
    matvec(X, beta, y, 2, 2);
    REQUIRE(y[0] == Approx(3.0));  // 1*1 + 2*1
    REQUIRE(y[1] == Approx(7.0));  // 3*1 + 4*1
  }

  SECTION("3x2 matrix") {
    // Row-major: [[1, 2], [3, 4], [5, 6]]
    double X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double beta[] = {2.0, 3.0};
    double y[] = {0.0, 0.0, 0.0};
    matvec(X, beta, y, 3, 2);
    REQUIRE(y[0] == Approx(8.0));   // 1*2 + 2*3
    REQUIRE(y[1] == Approx(18.0));  // 3*2 + 4*3
    REQUIRE(y[2] == Approx(28.0));  // 5*2 + 6*3
  }
}

TEST_CASE("matvec_add adds to existing y", "[linalg]") {
  double X[] = {1.0, 2.0, 3.0, 4.0};
  double beta[] = {1.0, 1.0};
  double y[] = {10.0, 20.0};
  matvec_add(X, beta, y, 2, 2);
  REQUIRE(y[0] == Approx(13.0));  // 10 + 1*1 + 2*1
  REQUIRE(y[1] == Approx(27.0));  // 20 + 3*1 + 4*1
}

TEST_CASE("matvec_transpose computes X'x", "[linalg]") {
  // Row-major: [[1, 2], [3, 4], [5, 6]]
  double X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double x[] = {1.0, 1.0, 1.0};
  double y[] = {0.0, 0.0};
  matvec_transpose(X, x, y, 3, 2);
  REQUIRE(y[0] == Approx(9.0));   // 1 + 3 + 5
  REQUIRE(y[1] == Approx(12.0));  // 2 + 4 + 6
}

TEST_CASE("dot_product_strided handles strides", "[linalg]") {
  // Extract every other element
  double x[] = {1.0, 0.0, 2.0, 0.0, 3.0};
  double y[] = {4.0, 0.0, 5.0, 0.0, 6.0};
  double result = dot_product_strided(x, 2, y, 2, 3);
  REQUIRE(result == Approx(32.0));  // 1*4 + 2*5 + 3*6
}

TEST_CASE("sparse_laplacian_quadform computes x'Lx", "[linalg]") {
  // Simple 3-node chain: 0--1--2
  // L = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
  int row_ptr[] = {0, 1, 3, 4};  // CSR format
  int col_idx[] = {1, 0, 2, 1};

  SECTION("constant vector has zero quadform") {
    double x[] = {1.0, 1.0, 1.0};
    REQUIRE(sparse_laplacian_quadform(row_ptr, col_idx, x, 3) == Approx(0.0));
  }

  SECTION("linear gradient") {
    double x[] = {0.0, 1.0, 2.0};
    // x'Lx = sum over edges of (x_i - x_j)^2
    // = (0-1)^2 + (1-2)^2 = 2
    REQUIRE(sparse_laplacian_quadform(row_ptr, col_idx, x, 3) == Approx(2.0));
  }
}
