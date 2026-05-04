suppressPackageStartupMessages({
  library(Rcpp)
  library(testthat)
})

cat("--- Step 1: Rcpp::compileAttributes() ---\n")
Rcpp::compileAttributes(".", verbose = FALSE)

cat("--- Step 2: devtools::load_all() (will recompile) ---\n")
suppressPackageStartupMessages(devtools::load_all('.', quiet = TRUE))
cat("PKG_LOADED\n")

cat("\n--- Step 3: B1b regression (all 7 base families) ---\n")
testthat::test_file("tests/testthat/test-specs-all-families.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- Step 4: B1c regression (ZI/hurdle/OI/ZOIB) ---\n")
testthat::test_file("tests/testthat/test-specs-zi.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- Step 5: B1d Step 1 (RE on spec path) ---\n")
testthat::test_file("tests/testthat/test-specs-re.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- Step 6: B1d Step 2 (spatial ICAR/BYM2 on spec path) ---\n")
testthat::test_file("tests/testthat/test-specs-spatial.R",
                    reporter = testthat::SummaryReporter$new())

cat("\n--- Step 7: B1d Step 3 (temporal RW1/AR1 on spec path) ---\n")
testthat::test_file("tests/testthat/test-specs-temporal.R",
                    reporter = testthat::SummaryReporter$new())
