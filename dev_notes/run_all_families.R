suppressPackageStartupMessages({
  devtools::load_all('.', quiet = TRUE)
  library(testthat)
})

cat("--- B1b: 7 families through X_list bridge ---\n")
testthat::test_file("tests/testthat/test-specs-all-families.R",
                    reporter = testthat::SummaryReporter$new())
