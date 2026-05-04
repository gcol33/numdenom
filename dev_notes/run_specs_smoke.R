suppressPackageStartupMessages({
  devtools::load_all('.', quiet = TRUE)
  library(testthat)
})

cat("--- B1a LikelihoodSpec PoC smoke test ---\n", sep = "")
res <- testthat::test_file(
  "tests/testthat/test-specs-binomial.R",
  reporter = testthat::SummaryReporter$new()
)
cat("\n--- tulpa-bridge ABI/PG bridge tests ---\n", sep = "")
res2 <- testthat::test_file(
  "tests/testthat/test-tulpa-bridge.R",
  reporter = testthat::SummaryReporter$new()
)
