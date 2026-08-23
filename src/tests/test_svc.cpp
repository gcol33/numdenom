// test_svc.cpp
// Catch2 tests for hmc_svc.h (Spatially-Varying Coefficients)

#include "catch.hpp"
#include "../hmc_svc.h"
#include <cmath>

using namespace ratiod_svc;

// ============================================================================
// Covariance functions
// ============================================================================

TEST_CASE("cov_exponential computes exponential covariance", "[svc]") {
  double sigma2 = 1.0;
  double phi = 1.0;

  SECTION("zero distance gives sigma2") {
    REQUIRE(cov_exponential(0.0, sigma2, phi) == Approx(sigma2));
  }

  SECTION("exponential decay") {
    double d1 = 1.0;
    double d2 = 2.0;
    double c1 = cov_exponential(d1, sigma2, phi);
    double c2 = cov_exponential(d2, sigma2, phi);
    REQUIRE(c1 > c2);
    REQUIRE(c2 / c1 == Approx(std::exp(-1.0)));
  }

  SECTION("phi scales distance") {
    double d = 2.0;
    double c1 = cov_exponential(d, sigma2, 1.0);
    double c2 = cov_exponential(d, sigma2, 2.0);
    // phi=2 should give higher covariance (slower decay)
    REQUIRE(c2 > c1);
  }

  SECTION("sigma2 scales covariance") {
    double d = 1.0;
    double c1 = cov_exponential(d, 1.0, phi);
    double c2 = cov_exponential(d, 4.0, phi);
    REQUIRE(c2 == Approx(4.0 * c1));
  }
}

TEST_CASE("cov_matern32 computes Matern 3/2 covariance", "[svc]") {
  double sigma2 = 1.0;
  double phi = 1.0;

  SECTION("zero distance gives sigma2") {
    REQUIRE(cov_matern32(0.0, sigma2, phi) == Approx(sigma2));
  }

  SECTION("decay with distance") {
    double c1 = cov_matern32(0.5, sigma2, phi);
    double c2 = cov_matern32(1.0, sigma2, phi);
    double c3 = cov_matern32(2.0, sigma2, phi);
    REQUIRE(c1 > c2);
    REQUIRE(c2 > c3);
  }

  SECTION("phi=1 at d=1") {
    double d = 1.0;
    double c = cov_matern32(d, 1.0, 1.0);
    // (1 + sqrt(3)*1) * exp(-sqrt(3)*1) ≈ 0.233
    double expected = (1.0 + std::sqrt(3.0)) * std::exp(-std::sqrt(3.0));
    REQUIRE(c == Approx(expected));
  }
}

TEST_CASE("cov_gaussian computes squared exponential covariance", "[svc]") {
  double sigma2 = 1.0;
  double phi = 1.0;

  SECTION("zero distance gives sigma2") {
    REQUIRE(cov_gaussian(0.0, sigma2, phi) == Approx(sigma2));
  }

  SECTION("rapid decay") {
    // exp(-0.5 d^2/phi^2) falls below exp(-d/phi) once d > 2 phi, and the two
    // are equal at d = 2 phi exactly.
    double c_exp = cov_exponential(3.0, sigma2, phi);
    double c_gau = cov_gaussian(3.0, sigma2, phi);
    REQUIRE(c_gau < c_exp);
    REQUIRE(cov_gaussian(2.0, sigma2, phi) ==
            Approx(cov_exponential(2.0, sigma2, phi)));
  }

  SECTION("expected value at d=phi") {
    double d = 1.0;
    double c = cov_gaussian(d, 1.0, 1.0);
    // exp(-0.5 * d^2 / phi^2) = exp(-0.5)
    REQUIRE(c == Approx(std::exp(-0.5)));
  }
}

TEST_CASE("cov_spherical computes spherical covariance", "[svc]") {
  double sigma2 = 1.0;
  double phi = 2.0;  // range parameter

  SECTION("zero distance gives sigma2") {
    REQUIRE(cov_spherical(0.0, sigma2, phi) == Approx(sigma2));
  }

  SECTION("beyond range gives zero") {
    REQUIRE(cov_spherical(3.0, sigma2, phi) == Approx(0.0));
    REQUIRE(cov_spherical(5.0, sigma2, phi) == Approx(0.0));
  }

  SECTION("at range boundary") {
    REQUIRE(cov_spherical(phi, sigma2, phi) == Approx(0.0));
  }

  SECTION("mid-range value") {
    double d = 1.0;  // d/phi = 0.5
    double r = d / phi;
    double expected = sigma2 * (1.0 - 1.5 * r + 0.5 * r * r * r);
    REQUIRE(cov_spherical(d, sigma2, phi) == Approx(expected));
  }
}

// ============================================================================
// compute_cov dispatcher
// ============================================================================

TEST_CASE("compute_cov dispatches to correct function", "[svc]") {
  double sigma2 = 2.0;
  double phi = 1.5;
  double d = 1.0;

  SECTION("EXPONENTIAL") {
    REQUIRE(compute_cov(d, sigma2, phi, CovType::EXPONENTIAL) ==
            Approx(cov_exponential(d, sigma2, phi)));
  }

  SECTION("MATERN") {
    REQUIRE(compute_cov(d, sigma2, phi, CovType::MATERN) ==
            Approx(cov_matern32(d, sigma2, phi)));
  }

  SECTION("GAUSSIAN") {
    REQUIRE(compute_cov(d, sigma2, phi, CovType::GAUSSIAN) ==
            Approx(cov_gaussian(d, sigma2, phi)));
  }

  SECTION("SPHERICAL") {
    REQUIRE(compute_cov(d, sigma2, phi, CovType::SPHERICAL) ==
            Approx(cov_spherical(d, sigma2, phi)));
  }
}
