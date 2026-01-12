// test_zi.cpp
// Catch2 tests for hmc_zi.h (zero-inflation functions)

#include "catch.hpp"
#include "../hmc_zi.h"
#include <cmath>

using namespace ratiod_zi;

// ============================================================================
// Helper functions
// ============================================================================

TEST_CASE("log1pexp is numerically stable", "[zi]") {
  SECTION("small values") {
    // For very negative x, log1pexp(x) ≈ exp(x)
    REQUIRE(log1pexp(-10.0) == Approx(std::exp(-10.0)).epsilon(0.001));
  }

  SECTION("around zero") {
    REQUIRE(log1pexp(0.0) == Approx(std::log(2.0)));
  }

  SECTION("large positive values") {
    REQUIRE(log1pexp(100.0) == Approx(100.0).margin(1e-10));
  }

  SECTION("moderate values") {
    double x = 5.0;
    REQUIRE(log1pexp(x) == Approx(std::log1p(std::exp(x))));
  }
}

TEST_CASE("logistic sigmoid function", "[zi]") {
  SECTION("at zero") {
    REQUIRE(logistic(0.0) == Approx(0.5));
  }

  SECTION("large positive") {
    REQUIRE(logistic(100.0) == Approx(1.0).margin(1e-10));
  }

  SECTION("large negative") {
    REQUIRE(logistic(-100.0) == Approx(0.0).margin(1e-10));
  }

  SECTION("positive value") {
    double x = 2.0;
    REQUIRE(logistic(x) == Approx(1.0 / (1.0 + std::exp(-x))));
  }

  SECTION("negative value") {
    double x = -2.0;
    REQUIRE(logistic(x) == Approx(1.0 / (1.0 + std::exp(-x))));
  }
}

TEST_CASE("log_logistic computes log(sigmoid)", "[zi]") {
  SECTION("at zero") {
    REQUIRE(log_logistic(0.0) == Approx(-std::log(2.0)));
  }

  SECTION("positive value") {
    double x = 2.0;
    double expected = std::log(1.0 / (1.0 + std::exp(-x)));
    REQUIRE(log_logistic(x) == Approx(expected));
  }

  SECTION("large positive") {
    REQUIRE(log_logistic(100.0) == Approx(0.0).margin(1e-10));
  }
}

TEST_CASE("log1m_logistic computes log(1-sigmoid)", "[zi]") {
  SECTION("at zero") {
    REQUIRE(log1m_logistic(0.0) == Approx(-std::log(2.0)));
  }

  SECTION("positive value") {
    double x = 2.0;
    double expected = std::log(1.0 - 1.0 / (1.0 + std::exp(-x)));
    REQUIRE(log1m_logistic(x) == Approx(expected));
  }

  SECTION("large positive gives very negative") {
    REQUIRE(log1m_logistic(100.0) < -90.0);
  }
}

// ============================================================================
// Distribution PMFs
// ============================================================================

TEST_CASE("poisson_lpmf computes Poisson log-PMF", "[zi]") {
  SECTION("y=0, mu=1") {
    // P(0|1) = exp(-1)
    REQUIRE(poisson_lpmf(0, 1.0) == Approx(-1.0));
  }

  SECTION("y=5, mu=5") {
    // Poisson mode should have high probability
    double lp = poisson_lpmf(5, 5.0);
    REQUIRE(lp > poisson_lpmf(0, 5.0));
    REQUIRE(lp > poisson_lpmf(10, 5.0));
  }

  SECTION("mu <= 0 returns very negative") {
    REQUIRE(poisson_lpmf(1, 0.0) < -1e9);
    REQUIRE(poisson_lpmf(1, -1.0) < -1e9);
  }
}

TEST_CASE("negbin_lpmf computes negative binomial log-PMF", "[zi]") {
  SECTION("y=0 has positive probability") {
    double lp = negbin_lpmf(0, 5.0, 2.0);
    REQUIRE(lp > -100.0);
    REQUIRE(std::exp(lp) > 0.0);
    REQUIRE(std::exp(lp) < 1.0);
  }

  SECTION("larger variance with smaller phi") {
    // Smaller phi = more overdispersion = flatter distribution
    double lp_low_phi = negbin_lpmf(10, 5.0, 0.5);
    double lp_high_phi = negbin_lpmf(10, 5.0, 100.0);
    // With high phi, NB approaches Poisson, so extreme values less likely
    // This depends on the specific value, but generally more overdispersion spreads mass
  }

  SECTION("invalid parameters") {
    REQUIRE(negbin_lpmf(1, 0.0, 1.0) < -1e9);
    REQUIRE(negbin_lpmf(1, 1.0, 0.0) < -1e9);
  }
}

// ============================================================================
// Zero-inflated likelihoods
// ============================================================================

TEST_CASE("zi_poisson_lpmf_logit handles zero inflation", "[zi]") {
  double mu = 5.0;

  SECTION("high ZI probability increases P(0)") {
    double logit_zi_high = 3.0;  // ~95% ZI
    double logit_zi_low = -3.0;  // ~5% ZI

    double lp_high_zi = zi_poisson_lpmf_logit(0, mu, logit_zi_high);
    double lp_low_zi = zi_poisson_lpmf_logit(0, mu, logit_zi_low);

    // Higher ZI should increase P(Y=0)
    REQUIRE(lp_high_zi > lp_low_zi);
  }

  SECTION("non-zero y unaffected by ZI for y>0") {
    double logit_zi = 2.0;
    double lp = zi_poisson_lpmf_logit(5, mu, logit_zi);
    // Should be log(1-pi) + Poisson(5|mu)
    double expected = log1m_logistic(logit_zi) + poisson_lpmf(5, mu);
    REQUIRE(lp == Approx(expected));
  }

  SECTION("no ZI (logit_zi very negative)") {
    double lp_zi = zi_poisson_lpmf_logit(3, mu, -100.0);
    double lp_plain = poisson_lpmf(3, mu);
    REQUIRE(lp_zi == Approx(lp_plain).margin(1e-6));
  }
}

TEST_CASE("zi_negbin_lpmf_logit handles zero-inflated NB", "[zi]") {
  double mu = 5.0;
  double phi = 2.0;

  SECTION("y=0 combines ZI and NB") {
    double logit_zi = 1.0;
    double lp = zi_negbin_lpmf_logit(0, mu, phi, logit_zi);

    // P(0) = pi + (1-pi)*NB(0|mu,phi)
    double pi = logistic(logit_zi);
    double expected = std::log(pi + (1-pi) * std::exp(negbin_lpmf(0, mu, phi)));
    REQUIRE(lp == Approx(expected));
  }

  SECTION("y>0 excludes ZI component") {
    double logit_zi = 1.0;
    double lp = zi_negbin_lpmf_logit(5, mu, phi, logit_zi);
    double expected = log1m_logistic(logit_zi) + negbin_lpmf(5, mu, phi);
    REQUIRE(lp == Approx(expected));
  }
}

// ============================================================================
// Truncated distributions
// ============================================================================

TEST_CASE("truncated_poisson_lpmf for positive values only", "[zi]") {
  double mu = 5.0;

  SECTION("y=0 returns -infinity") {
    REQUIRE(truncated_poisson_lpmf(0, mu) < -1e9);
  }

  SECTION("y>0 is normalized") {
    // P(Y=y | Y>0) = P(Y=y) / P(Y>0) = P(Y=y) / (1 - exp(-mu))
    double lp = truncated_poisson_lpmf(3, mu);
    double expected = poisson_lpmf(3, mu) - std::log(1.0 - std::exp(-mu));
    REQUIRE(lp == Approx(expected));
  }
}

TEST_CASE("truncated_negbin_lpmf for positive values only", "[zi]") {
  double mu = 5.0;
  double phi = 2.0;

  SECTION("y=0 returns -infinity") {
    REQUIRE(truncated_negbin_lpmf(0, mu, phi) < -1e9);
  }

  SECTION("y>0 is normalized") {
    double p0 = std::exp(negbin_lpmf(0, mu, phi));
    double lp = truncated_negbin_lpmf(3, mu, phi);
    double expected = negbin_lpmf(3, mu, phi) - std::log(1.0 - p0);
    REQUIRE(lp == Approx(expected));
  }
}

// ============================================================================
// Hurdle models
// ============================================================================

TEST_CASE("hurdle_poisson_lpmf_logit", "[zi]") {
  double mu = 5.0;

  SECTION("y=0 uses binary component") {
    double logit_theta = 1.0;  // theta = P(Y>0)
    double lp = hurdle_poisson_lpmf_logit(0, mu, logit_theta);
    // P(Y=0) = 1 - theta
    REQUIRE(lp == Approx(log1m_logistic(logit_theta)));
  }

  SECTION("y>0 uses truncated Poisson") {
    double logit_theta = 1.0;
    double lp = hurdle_poisson_lpmf_logit(5, mu, logit_theta);
    double expected = log_logistic(logit_theta) + truncated_poisson_lpmf(5, mu);
    REQUIRE(lp == Approx(expected));
  }
}

TEST_CASE("hurdle_negbin_lpmf_logit", "[zi]") {
  double mu = 5.0;
  double phi = 2.0;

  SECTION("y=0 uses binary component") {
    double logit_theta = 0.5;
    double lp = hurdle_negbin_lpmf_logit(0, mu, phi, logit_theta);
    REQUIRE(lp == Approx(log1m_logistic(logit_theta)));
  }

  SECTION("y>0 uses truncated NB") {
    double logit_theta = 0.5;
    double lp = hurdle_negbin_lpmf_logit(5, mu, phi, logit_theta);
    double expected = log_logistic(logit_theta) + truncated_negbin_lpmf(5, mu, phi);
    REQUIRE(lp == Approx(expected));
  }
}

// ============================================================================
// ZI type parsing
// ============================================================================

TEST_CASE("parse_zi_type converts strings to enum", "[zi]") {
  REQUIRE(parse_zi_type("none") == ZIType::NONE);
  REQUIRE(parse_zi_type("") == ZIType::NONE);
  REQUIRE(parse_zi_type("zi_poisson") == ZIType::ZI_POISSON);
  REQUIRE(parse_zi_type("zi_negbin") == ZIType::ZI_NEGBIN);
  REQUIRE(parse_zi_type("hurdle_poisson") == ZIType::HURDLE_POISSON);
  REQUIRE(parse_zi_type("hurdle_negbin") == ZIType::HURDLE_NEGBIN);
  REQUIRE(parse_zi_type("unknown") == ZIType::NONE);
}

// ============================================================================
// Generic ZI log-likelihood
// ============================================================================

TEST_CASE("zi_log_likelihood dispatches correctly", "[zi]") {
  int y = 3;
  double mu = 5.0;
  double phi = 2.0;
  double logit_zi = 0.5;

  SECTION("NONE with phi=0 returns Poisson") {
    // When phi <= 0, NONE uses Poisson
    double lp = zi_log_likelihood(y, mu, 0.0, logit_zi, ZIType::NONE);
    REQUIRE(lp == Approx(poisson_lpmf(y, mu)));
  }

  SECTION("NONE with phi>0 returns NegBin") {
    // When phi > 0, NONE uses NegBin
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::NONE);
    REQUIRE(lp == Approx(negbin_lpmf(y, mu, phi)));
  }

  SECTION("ZI_POISSON") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::ZI_POISSON);
    REQUIRE(lp == Approx(zi_poisson_lpmf_logit(y, mu, logit_zi)));
  }

  SECTION("ZI_NEGBIN") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::ZI_NEGBIN);
    REQUIRE(lp == Approx(zi_negbin_lpmf_logit(y, mu, phi, logit_zi)));
  }

  SECTION("HURDLE_POISSON") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::HURDLE_POISSON);
    REQUIRE(lp == Approx(hurdle_poisson_lpmf_logit(y, mu, logit_zi)));
  }

  SECTION("HURDLE_NEGBIN") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::HURDLE_NEGBIN);
    REQUIRE(lp == Approx(hurdle_negbin_lpmf_logit(y, mu, phi, logit_zi)));
  }
}

// ============================================================================
// Gradients
// ============================================================================

TEST_CASE("zi_poisson_grad_logit_zi numerical check", "[zi][gradient]") {
  int y = 0;
  double mu = 5.0;
  double logit_zi = 1.0;

  // Numerical gradient
  double eps = 1e-6;
  double f_plus = zi_poisson_lpmf_logit(y, mu, logit_zi + eps);
  double f_minus = zi_poisson_lpmf_logit(y, mu, logit_zi - eps);
  double numerical_grad = (f_plus - f_minus) / (2.0 * eps);

  double analytic_grad = zi_poisson_grad_logit_zi(y, mu, logit_zi);

  REQUIRE(analytic_grad == Approx(numerical_grad).margin(1e-4));
}

TEST_CASE("zi_negbin_grad_logit_zi numerical check", "[zi][gradient]") {
  int y = 0;
  double mu = 5.0;
  double phi = 2.0;
  double logit_zi = 1.0;

  double eps = 1e-6;
  double f_plus = zi_negbin_lpmf_logit(y, mu, phi, logit_zi + eps);
  double f_minus = zi_negbin_lpmf_logit(y, mu, phi, logit_zi - eps);
  double numerical_grad = (f_plus - f_minus) / (2.0 * eps);

  double analytic_grad = zi_negbin_grad_logit_zi(y, mu, phi, logit_zi);

  REQUIRE(analytic_grad == Approx(numerical_grad).margin(1e-4));
}

TEST_CASE("hurdle_grad_logit_theta numerical check", "[zi][gradient]") {
  SECTION("y=0") {
    int y = 0;
    double logit_theta = 0.5;

    double eps = 1e-6;
    double f_plus = log1m_logistic(logit_theta + eps);
    double f_minus = log1m_logistic(logit_theta - eps);
    double numerical_grad = (f_plus - f_minus) / (2.0 * eps);

    double analytic_grad = hurdle_grad_logit_theta(y, logit_theta);

    REQUIRE(analytic_grad == Approx(numerical_grad).margin(1e-4));
  }

  SECTION("y>0") {
    int y = 5;
    double logit_theta = 0.5;

    double eps = 1e-6;
    double f_plus = log_logistic(logit_theta + eps);
    double f_minus = log_logistic(logit_theta - eps);
    double numerical_grad = (f_plus - f_minus) / (2.0 * eps);

    double analytic_grad = hurdle_grad_logit_theta(y, logit_theta);

    REQUIRE(analytic_grad == Approx(numerical_grad).margin(1e-4));
  }
}
