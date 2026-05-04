suppressPackageStartupMessages({
  devtools::load_all('.', quiet = TRUE)
  library(testthat)
})

cat("--- B1b regression: 7 base ratio families ---\n")
testthat::test_file("tests/testthat/test-specs-all-families.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- B1c: ZI / hurdle / OI / ZOIB ---\n")
testthat::test_file("tests/testthat/test-specs-zi.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- tulpa-bridge sanity ---\n")
testthat::test_file("tests/testthat/test-tulpa-bridge.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- B1b binomial smoke (original PoC) ---\n")
testthat::test_file("tests/testthat/test-specs-binomial.R",
                    reporter = testthat::SummaryReporter$new())
