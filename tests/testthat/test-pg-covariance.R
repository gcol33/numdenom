# spatial_gp(cov = ) offers four kernels and the PG binomial GP sampler used to
# distinguish two of them: its selector tested cov_type 0 and 1 and sent
# everything else to an unconditional else branch computing Matern 2.5, so
# "gaussian" and "spherical" both ran a kernel the user did not ask for
# (gcol33/tulpaRatio#35).
#
# The sampler cannot arbitrate this. Its sigma2_gp rails -- on the fixture below
# it reaches 1e4 against a true 0.7 -- and its draws do not reproduce at a fixed
# seed, so two runs differ whichever kernel ran. The conditional is driven
# directly instead, against the four kernels written out here and an NNGP
# conditional derived from its definition rather than read off the C++.

pg_kernels <- list(
  exponential = function(d, s2, phi) s2 * exp(-d / phi),
  matern      = function(d, s2, phi) {
    u <- sqrt(3) * d / phi
    s2 * (1 + u) * exp(-u)
  },
  gaussian    = function(d, s2, phi) s2 * exp(-0.5 * (d / phi)^2),
  spherical   = function(d, s2, phi) {
    r <- d / phi
    ifelse(d >= phi, 0, s2 * (1 - 1.5 * r + 0.5 * r^3))
  },
  # Not a spatial_gp() choice: what the old else branch returned for the two
  # kernels it did not recognise.
  matern25    = function(d, s2, phi) {
    u <- sqrt(5) * d / phi
    s2 * (1 + u + u * u / 3) * exp(-u)
  }
)

pg_cov_fixture <- function(seed = 21L, n = 12L, nn = 4L) {
  set.seed(seed)
  co <- cbind(runif(n), runif(n))
  ord <- order(co[, 1])
  D <- as.matrix(dist(co))
  nn_idx <- matrix(0L, n, nn)
  nn_dist <- matrix(0, n, nn)
  for (i in seq_len(n)) {
    prev <- seq_len(i - 1L)
    if (!length(prev)) next
    k <- min(nn, length(prev))
    cand <- prev[order(D[ord[i], ord[prev]])][seq_len(k)]
    nn_idx[i, seq_len(k)] <- as.integer(cand)
    nn_dist[i, seq_len(k)] <- D[ord[i], ord[cand]]
  }
  list(coords = co, ord = ord, D = D, nn_idx = nn_idx, nn_dist = nn_dist,
       nn = nn, w = rnorm(n), sigma2 = 0.7, phi = 0.35, i0 = 8L)
}

# The NNGP conditional at one location, from its definition. The ridge matches
# ratiod_cov::NNGP_RIDGE, which is what every neighbour block in the package now
# carries.
pg_cond_reference <- function(f, fx) {
  k <- sum(fx$nn_idx[fx$i0, ] > 0)
  cvec <- vapply(seq_len(k),
                 function(j) f(fx$nn_dist[fx$i0, j], fx$sigma2, fx$phi), 0)
  C <- matrix(0, k, k)
  for (a in seq_len(k)) {
    for (b in seq_len(k)) {
      C[a, b] <- if (a == b) {
        fx$sigma2
      } else {
        f(fx$D[fx$ord[fx$nn_idx[fx$i0, a]], fx$ord[fx$nn_idx[fx$i0, b]]],
          fx$sigma2, fx$phi)
      }
    }
  }
  L <- t(chol(C + diag(1e-8, k)))
  alpha <- backsolve(t(L), forwardsolve(L, cvec))
  c(cond_mean = sum(alpha * fx$w[fx$ord[fx$nn_idx[fx$i0, seq_len(k)]]]),
    cond_var  = max(1e-10, fx$sigma2 - sum(cvec * alpha)))
}

pg_cond_cpp <- function(cov_type, fx) {
  as.numeric(tulpaRatio:::cpp_pg_nngp_conditional_probe(
    i = fx$i0 - 1L, w = fx$w, sigma2 = fx$sigma2, phi_gp = fx$phi,
    cov_type = cov_type, coords = fx$coords, nn_idx = fx$nn_idx,
    nn_dist = fx$nn_dist, nn_order = as.integer(fx$ord - 1L), nn = fx$nn
  ))
}

PG_COV_CODES <- c(exponential = 0L, matern = 1L, gaussian = 2L, spherical = 3L)

test_that("every spatial_gp(cov = ) choice runs its own kernel under PG", {
  fx <- pg_cov_fixture()
  for (cv in names(PG_COV_CODES)) {
    expect_equal(
      pg_cond_cpp(PG_COV_CODES[[cv]], fx),
      unname(pg_cond_reference(pg_kernels[[cv]], fx)),
      tolerance = 1e-12,
      label = sprintf("cov = %s", cv)
    )
  }
})

test_that("gaussian and spherical are not the Matern 2.5 the else branch gave", {
  fx <- pg_cov_fixture()
  old <- unname(pg_cond_reference(pg_kernels$matern25, fx))
  for (cv in c("gaussian", "spherical")) {
    got <- pg_cond_cpp(PG_COV_CODES[[cv]], fx)
    # Both used to equal `old` exactly. The conditional mean moves by 0.11 on
    # gaussian and 0.34 on spherical at this fixture, and the variance by 0.11
    # and 0.38, so this is not a tolerance question.
    expect_gt(max(abs(got - old)), 0.1)
  }
  expect_false(isTRUE(all.equal(pg_cond_cpp(2L, fx), pg_cond_cpp(3L, fx),
                                tolerance = 0)))
})

test_that("an unrecognised covariance code is refused, not resolved", {
  fx <- pg_cov_fixture()
  # The old selector sent 2, 3 and anything else to Matern 2.5. Every value R
  # can send is now mapped explicitly and the rest error.
  expect_error(pg_cond_cpp(4L, fx), "Unknown covariance code")
  expect_error(pg_cond_cpp(-1L, fx), "Unknown covariance code")
})

test_that("tratio(mode = 'pg') honours control$seed and control$chains", {
  skip_on_cran()
  # fit_pg_binomial() takes both and threads seed down to a per-chain
  # set.seed(seed + chain - 1), but tratio() called it without either, so PG
  # fits ran off whatever RNG state they inherited and always used the
  # backend's own default of 4 chains.
  set.seed(3)
  n_loc <- 20L
  reps <- 4L
  co <- cbind(runif(n_loc), runif(n_loc))
  S <- 0.7 * exp(-as.matrix(dist(co)) / 0.3)
  w <- as.vector(t(chol(S + diag(1e-8, n_loc))) %*% rnorm(n_loc))
  loc <- rep(seq_len(n_loc), each = reps)
  x <- rnorm(n_loc * reps)
  n <- rep(30L, n_loc * reps)
  df <- data.frame(
    y = rbinom(n_loc * reps, n, plogis(-0.2 + 0.5 * x + w[loc])),
    n = n, x = x, lon = co[loc, 1], lat = co[loc, 2],
    g = factor(rep(1:5, length.out = n_loc * reps))
  )
  run <- function(chains) {
    tratio(y | n ~ x + (1 | g), data = df, family = ratiod_binomial(),
           mode = "pg", spatial = spatial_gp(~ lon + lat, nn = 5),
           control = list(iter = 120, warmup = 60, chains = chains,
                          verbose = FALSE, seed = 5))$draws
  }
  a <- run(1L)
  b <- run(1L)
  expect_equal(a, b, tolerance = 0)
  expect_identical(dim(a)[2], 1L)
  expect_identical(dim(run(2L))[2], 2L)
})

test_that("the PG GP sampler does not read past its per-location arrays", {
  skip_on_cran()
  # backend_pg.R passed nn_order 1-based while the C++ uses its values directly
  # as indices into w, coords and the per-location likelihood accumulators, all
  # sized n_spatial -- so every one could reach n_spatial, one past the end.
  # backend_hmc.R converts the same vector with - 1L; PG did not. The symptom
  # was STATUS_HEAP_CORRUPTION (0xC0000374) killing the R process, reached at
  # some parameter states and not others: on the original code 20 locations
  # crashed at nn = 5 and nn = 6, while 25 locations survived.
  mk <- function(n_loc) {
    set.seed(3)
    reps <- 4L
    co <- cbind(runif(n_loc), runif(n_loc))
    S <- 0.7 * exp(-as.matrix(dist(co)) / 0.3)
    w <- as.vector(t(chol(S + diag(1e-8, n_loc))) %*% rnorm(n_loc))
    loc <- rep(seq_len(n_loc), each = reps)
    x <- rnorm(n_loc * reps)
    n <- rep(30L, n_loc * reps)
    data.frame(y = rbinom(n_loc * reps, n, plogis(-0.2 + 0.5 * x + w[loc])),
               n = n, x = x, lon = co[loc, 1], lat = co[loc, 2],
               g = factor(rep(1:5, length.out = n_loc * reps)))
  }
  for (cfg in list(c(20L, 1L, 5L), c(20L, 4L, 5L), c(20L, 1L, 6L),
                   c(15L, 2L, 6L))) {
    d <- tratio(y | n ~ x + (1 | g), data = mk(cfg[1]),
                family = ratiod_binomial(), mode = "pg",
                spatial = spatial_gp(~ lon + lat, nn = cfg[3]),
                control = list(iter = 120, warmup = 60, chains = cfg[2],
                               verbose = FALSE, seed = 5))$draws
    expect_identical(dim(d)[2], cfg[2])
    expect_true(all(is.finite(d)))
  }
})
