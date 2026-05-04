suppressPackageStartupMessages({
  devtools::load_all('.', quiet = TRUE)
  library(testthat)
})

cat("--- B1b: 7 base families ---\n")
testthat::test_file("tests/testthat/test-specs-all-families.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- B1c: ZI/hurdle/OI/ZOIB ---\n")
testthat::test_file("tests/testthat/test-specs-zi.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- B1d Step 1: RE on spec path ---\n")
testthat::test_file("tests/testthat/test-specs-re.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- B2: H-kernel (handcoded gradient) ---\n")
testthat::test_file("tests/testthat/test-specs-h-kernel.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- B1b PoC binomial ---\n")
testthat::test_file("tests/testthat/test-specs-binomial.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- tulpa-bridge sanity ---\n")
testthat::test_file("tests/testthat/test-tulpa-bridge.R",
                    reporter = testthat::SummaryReporter$new())
