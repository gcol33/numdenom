suppressPackageStartupMessages({
  library(Rcpp)
  library(testthat)
})

cat("--- Step 1: Rcpp::compileAttributes() ---\n")
Rcpp::compileAttributes(".", verbose = FALSE)

cat("--- Step 2: devtools::load_all() (will recompile) ---\n")
suppressPackageStartupMessages(devtools::load_all('.', quiet = TRUE))
cat("PKG_LOADED\n")

cat("--- Step 3: B1b LikelihoodSpec PoC tests ---\n")
testthat::test_file("tests/testthat/test-specs-binomial.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- Step 4: tulpa-bridge sanity tests ---\n")
testthat::test_file("tests/testthat/test-tulpa-bridge.R",
                    reporter = testthat::SummaryReporter$new())
