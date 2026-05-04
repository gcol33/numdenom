#!/usr/bin/env Rscript
setwd("C:/Users/Gilles Colling/Documents/dev/tulpaRatio")
Sys.setenv(NOT_CRAN = "true")
Sys.unsetenv("_R_CHECK_LIMIT_CORES_")
suppressPackageStartupMessages({ library(devtools) })
devtools::load_all(quiet = TRUE)

cat("===== test-specs-zi.R =====\n")
res <- testthat::test_file("tests/testthat/test-specs-zi.R")
print(res)
