# The Gibbs spatial backend must return the same fit contract as every other
# backend: the number of chains that was requested, a 2-D draws matrix carrying
# parameter names, and a per-chain samples list. Without those, split-Rhat is
# not computable and a spatial fit cannot be checked for convergence at all.

gibbs_test_data <- function(n = 300, n_sites = 25) {
  set.seed(123)
  x <- rnorm(n)
  site <- factor(rep(seq_len(n_sites), length.out = n))
  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = seq_len(n_side), lat = seq_len(n_side))[seq_len(n_sites), ]
  adj <- matrix(0, n_sites, n_sites)
  for (i in seq_len(n_sites)) for (j in seq_len(n_sites)) {
    d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
    if (i != j && d <= 1.5) adj[i, j] <- 1
  }
  dimnames(adj) <- list(levels(site), levels(site))
  trials <- sample(10:50, n, replace = TRUE)
  y <- rbinom(n, trials, plogis(0.5 + 0.3 * x))
  list(df = data.frame(y = y, trials = trials, x = x, site = site,
                       spatial_site = site),
       adj = adj)
}

gibbs_fit <- function(d, chains = 4, seed = 99, spatial = NULL) {
  if (is.null(spatial)) {
    spatial <- spatial_car(d$adj, level = "group", group_var = "spatial_site")
  }
  ratiod(y | trials ~ x + (1 | site), data = d$df,
         family = ratiod_binomial(), spatial = spatial,
         iter = 600, warmup = 300, chains = chains, seed = seed,
         verbose = FALSE)
}

test_that("Gibbs backend honours the requested number of chains", {
  skip_on_cran()
  d <- gibbs_test_data()
  fit <- gibbs_fit(d, chains = 4)

  expect_identical(fit$backend, "gibbs")
  expect_identical(as.integer(fit$chains), 4L)
  expect_length(fit$samples, 4L)
  expect_identical(nrow(as.matrix(fit$draws)),
                   as.integer(fit$n_save_per_chain * 4L))
})

test_that("Gibbs draws are a named 2-D matrix including the spatial field", {
  skip_on_cran()
  d <- gibbs_test_data()
  dr <- as.matrix(gibbs_fit(d, chains = 2)$draws)

  expect_length(dim(dr), 2L)
  expect_false(is.null(colnames(dr)))
  expect_true(all(c("beta_num[1]", "beta_num[2]", "log_tau") %in% colnames(dr)))
  # The spatial field is part of the posterior, not a side channel.
  expect_true(any(grepl("^phi\\[", colnames(dr))))
})

test_that("split-Rhat is computable on a Gibbs fit and the chains agree", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  d <- gibbs_test_data()
  fit <- gibbs_fit(d, chains = 4)

  dr <- as.matrix(fit$draws)
  per <- nrow(dr) / fit$chains
  for (nm in c("beta_num[1]", "beta_num[2]")) {
    m <- matrix(dr[, nm], nrow = per, ncol = fit$chains)
    expect_lt(posterior::rhat(m), 1.05)
    expect_gt(posterior::ess_bulk(m), 100)
  }
})

test_that("Gibbs chains are independent but the fit is seed-reproducible", {
  skip_on_cran()
  d <- gibbs_test_data()
  fit <- gibbs_fit(d, chains = 3, seed = 7)

  # Distinct seeds per chain: no two chains may be the same draws.
  expect_false(identical(fit$samples[[1]], fit$samples[[2]]))
  expect_false(identical(fit$samples[[2]], fit$samples[[3]]))

  again <- gibbs_fit(d, chains = 3, seed = 7)
  expect_identical(as.matrix(fit$draws), as.matrix(again$draws))

  other <- gibbs_fit(d, chains = 3, seed = 8)
  expect_false(identical(as.matrix(fit$draws), as.matrix(other$draws)))
})

test_that("Gibbs recovers the fixed effect under a spatial field", {
  skip_on_cran()
  d <- gibbs_test_data()
  dr <- as.matrix(gibbs_fit(d, chains = 4)$draws)

  ci <- unname(quantile(dr[, "beta_num[2]"], c(0.025, 0.975)))
  expect_lt(ci[1], 0.3)
  expect_gt(ci[2], 0.3)
})

test_that("single-chain and BYM2 fits keep the contract", {
  skip_on_cran()
  d <- gibbs_test_data()

  one <- gibbs_fit(d, chains = 1)
  expect_identical(as.integer(one$chains), 1L)
  expect_false(is.null(colnames(as.matrix(one$draws))))

  bym2 <- gibbs_fit(d, chains = 2,
                    spatial = spatial_bym2(d$adj, level = "group",
                                           group_var = "spatial_site"))
  expect_identical(as.integer(bym2$chains), 2L)
  expect_false(is.null(colnames(as.matrix(bym2$draws))))
  expect_true("logit_rho" %in% colnames(as.matrix(bym2$draws)))
})
