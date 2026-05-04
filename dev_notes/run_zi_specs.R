suppressPackageStartupMessages({
  devtools::load_all('.', quiet = TRUE)
  library(testthat)
})

cat("--- B1c: ZI/hurdle/OI/ZOIB through spec path ---\n")
testthat::test_file("tests/testthat/test-specs-zi.R",
                    reporter = testthat::SummaryReporter$new())
