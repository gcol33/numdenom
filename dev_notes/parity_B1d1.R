# B1d-1 parity: single-grouping-factor RE on the spec path vs. legacy.
#
# For each (family x re_kind x gradient_mode):
#   * fit legacy (use_specs=FALSE) seed 42 and 43
#   * fit specs  (use_specs=TRUE)  seed 42 and 43
#   * compute cross max-abs |mean_legacy - mean_specs|  per param
#   * compute within max-abs |mean_seed42 - mean_seed43| (legacy only) as MC noise
#   * report ratio cross / within; pass if ratio < 4
#
# Note: H mode is not supported once RE is on (bridge sets gradient_fn = nullptr
# and falls through to autodiff). We still run gradient_mode = "H" requests so
# the engine's auto-fallback path is exercised.

suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
})

set.seed(2024)
N <- 200L
n_groups <- 8L

simulate_data <- function(family, with_slope = FALSE) {
  group <- factor(rep(seq_len(n_groups), length.out = N))
  x_fix <- rnorm(N)
  x_re  <- rnorm(N)

  if (family == "binomial") {
    n <- rpois(N, 8) + 5L
    eta <- 0.3 + 0.4 * x_fix +
           rnorm(n_groups, sd = 0.5)[as.integer(group)] +
           if (with_slope) (rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re) else 0
    p <- plogis(eta)
    y <- rbinom(N, n, p)
    list(y = y, n = n, x_fix = x_fix, x_re = x_re, group = group)
  } else if (family == "beta_binomial") {
    n <- rpois(N, 8) + 5L
    eta <- 0.3 + 0.4 * x_fix +
           rnorm(n_groups, sd = 0.5)[as.integer(group)] +
           if (with_slope) (rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re) else 0
    p <- plogis(eta)
    phi <- 10
    a <- p * phi; b <- (1 - p) * phi
    p_obs <- rbeta(N, a, b)
    y <- rbinom(N, n, p_obs)
    list(y = y, n = n, x_fix = x_fix, x_re = x_re, group = group)
  } else if (family %in% c("poisson_gamma", "negbin_gamma")) {
    eta_n <- 0.3 + 0.4 * x_fix +
             rnorm(n_groups, sd = 0.5)[as.integer(group)] +
             if (with_slope) (rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re) else 0
    eta_d <- 0.5 + 0.2 * x_fix
    mu_n <- exp(eta_n); mu_d <- exp(eta_d)
    y_n  <- rpois(N, mu_n)
    y_d  <- rgamma(N, shape = 2, rate = 2 / mu_d)
    list(y_num = y_n, y_denom = y_d, x_fix = x_fix, x_re = x_re, group = group)
  } else if (family == "negbin_negbin") {
    eta_n <- 0.3 + 0.4 * x_fix +
             rnorm(n_groups, sd = 0.5)[as.integer(group)] +
             if (with_slope) (rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re) else 0
    eta_d <- 0.5 + 0.2 * x_fix
    mu_n <- exp(eta_n); mu_d <- exp(eta_d)
    y_n  <- rnbinom(N, size = 2, mu = mu_n)
    y_d  <- rnbinom(N, size = 2, mu = mu_d)
    list(y_num = y_n, y_denom = y_d, x_fix = x_fix, x_re = x_re, group = group)
  } else if (family == "gamma_gamma") {
    eta_n <- 0.3 + 0.4 * x_fix +
             rnorm(n_groups, sd = 0.5)[as.integer(group)] +
             if (with_slope) (rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re) else 0
    eta_d <- 0.5 + 0.2 * x_fix
    mu_n <- exp(eta_n); mu_d <- exp(eta_d)
    y_n  <- rgamma(N, shape = 2, rate = 2 / mu_n)
    y_d  <- rgamma(N, shape = 2, rate = 2 / mu_d)
    list(y_num = y_n, y_denom = y_d, x_fix = x_fix, x_re = x_re, group = group)
  } else if (family == "lognormal") {
    eta_n <- 0.3 + 0.4 * x_fix +
             rnorm(n_groups, sd = 0.5)[as.integer(group)] +
             if (with_slope) (rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re) else 0
    eta_d <- 0.5 + 0.2 * x_fix
    y_n   <- exp(eta_n + rnorm(N, sd = 0.5))
    y_d   <- exp(eta_d + rnorm(N, sd = 0.5))
    list(y_num = y_n, y_denom = y_d, x_fix = x_fix, x_re = x_re, group = group)
  } else stop("unknown family ", family)
}

build_formula <- function(family, with_slope) {
  re_term <- if (with_slope) "(1 + x_re || group)" else "(1 | group)"
  if (family %in% c("binomial", "beta_binomial")) {
    as.formula(paste0("successes | trials ~ x_fix + ", re_term))
  } else {
    as.formula(paste0("y_num | y_denom ~ x_fix + ", re_term))
  }
}

build_data <- function(family, sim) {
  if (family %in% c("binomial", "beta_binomial")) {
    data.frame(successes = sim$y, trials = sim$n,
               x_fix = sim$x_fix, x_re = sim$x_re, group = sim$group)
  } else {
    data.frame(y_num = sim$y_num, y_denom = sim$y_denom,
               x_fix = sim$x_fix, x_re = sim$x_re, group = sim$group)
  }
}

family_obj <- function(family) {
  switch(family,
    binomial      = ratiod_binomial(),
    beta_binomial = ratiod_beta_binomial(),
    poisson_gamma = ratiod_poisson_gamma(),
    negbin_gamma  = ratiod_negbin_gamma(),
    negbin_negbin = ratiod_negbin_negbin(),
    gamma_gamma   = ratiod_gamma_gamma(),
    lognormal     = ratiod_lognormal(),
    stop("Unknown family ", family)
  )
}

run_fit <- function(family, with_slope, seed, use_specs, gradient_mode = "A_r",
                    iter = 500L, warmup = 300L) {
  sim <- simulate_data(family, with_slope = with_slope)
  data <- build_data(family, sim)
  formula <- build_formula(family, with_slope)
  options(tulpaRatio.use_specs = use_specs)
  on.exit(options(tulpaRatio.use_specs = FALSE), add = TRUE)
  ratiod(
    formula = formula, data = data, family = family_obj(family),
    mode = "hmc",
    iter = iter, warmup = warmup,
    chains = 1L, seed = seed, verbose = FALSE,
    gradient_mode = gradient_mode
  )
}

posterior_means <- function(fit) {
  draws <- fit$draws
  if (is.list(draws) && !is.matrix(draws)) draws <- draws[[1]]
  if (is.matrix(draws)) {
    colMeans(draws)
  } else {
    summary(fit)$coefficients[, "Estimate"]
  }
}

cases <- expand.grid(
  family = c("binomial", "beta_binomial", "poisson_gamma",
             "negbin_gamma", "negbin_negbin", "gamma_gamma", "lognormal"),
  re_kind = c("intercept_only", "intercept_slope"),
  gradient_mode = c("A_r"),
  stringsAsFactors = FALSE
)
# H-mode runs as a smoke check on binomial only (the bridge falls back to
# autodiff when RE is on; verify the auto-fallback does not crash).
cases <- rbind(cases,
  data.frame(family = c("binomial", "binomial"),
             re_kind = c("intercept_only", "intercept_slope"),
             gradient_mode = c("H", "H"),
             stringsAsFactors = FALSE))

results <- list()
for (i in seq_len(nrow(cases))) {
  fam   <- cases$family[i]
  re_k  <- cases$re_kind[i]
  gm    <- cases$gradient_mode[i]
  with_s <- (re_k == "intercept_slope")

  msg <- sprintf("[%2d/%d] %s | %s | %s",
                 i, nrow(cases), fam, re_k, gm)
  cat(msg, "\n", sep = "")

  fit_legacy_42 <- tryCatch(run_fit(fam, with_s, 42L, FALSE, gm),
                            error = function(e) e)
  fit_legacy_43 <- tryCatch(run_fit(fam, with_s, 43L, FALSE, gm),
                            error = function(e) e)
  fit_specs_42  <- tryCatch(run_fit(fam, with_s, 42L, TRUE, gm),
                            error = function(e) e)

  if (inherits(fit_legacy_42, "error") || inherits(fit_legacy_43, "error") ||
      inherits(fit_specs_42, "error")) {
    err <- if (inherits(fit_legacy_42, "error")) fit_legacy_42$message
           else if (inherits(fit_legacy_43, "error")) fit_legacy_43$message
           else fit_specs_42$message
    results[[i]] <- list(family = fam, re_kind = re_k, gradient_mode = gm,
                         status = "ERROR", error = err)
    cat("    ERROR:", err, "\n")
    next
  }

  m_l_42 <- posterior_means(fit_legacy_42)
  m_l_43 <- posterior_means(fit_legacy_43)
  m_s_42 <- posterior_means(fit_specs_42)

  # Match on shared parameter names (legacy and specs may have slightly
  # different naming; align by intersecting names if available).
  align <- function(a, b) {
    if (!is.null(names(a)) && !is.null(names(b))) {
      common <- intersect(names(a), names(b))
      a <- a[common]; b <- b[common]
    } else {
      n <- min(length(a), length(b))
      a <- a[seq_len(n)]; b <- b[seq_len(n)]
    }
    list(a = a, b = b)
  }

  cross <- align(m_l_42, m_s_42)
  within <- align(m_l_42, m_l_43)
  d_cross  <- max(abs(cross$a  - cross$b),  na.rm = TRUE)
  d_within <- max(abs(within$a - within$b), na.rm = TRUE)
  ratio    <- d_cross / max(d_within, 1e-8)

  pass <- ratio < 4
  results[[i]] <- list(family = fam, re_kind = re_k, gradient_mode = gm,
                       status = if (pass) "PASS" else "FAIL",
                       d_cross = d_cross, d_within = d_within, ratio = ratio,
                       n_par = length(cross$a))
  cat(sprintf("    cross=%.4f  within=%.4f  ratio=%.2f  -> %s\n",
              d_cross, d_within, ratio, if (pass) "PASS" else "FAIL"))
}

cat("\n", strrep("=", 78), "\n", sep = "")
cat("B1d-1 parity summary\n")
cat(strrep("=", 78), "\n", sep = "")
hdr <- sprintf("%-15s %-18s %-5s %-6s %-9s %-9s %-7s %s",
               "family", "re_kind", "grad", "status", "d_cross", "d_within",
               "ratio", "n_par")
cat(hdr, "\n", sep = "")
for (r in results) {
  if (is.null(r$d_cross)) {
    line <- sprintf("%-15s %-18s %-5s %-6s %s",
                    r$family, r$re_kind, r$gradient_mode, r$status,
                    substr(r$error %||% "", 1, 80))
  } else {
    line <- sprintf("%-15s %-18s %-5s %-6s %-9.4f %-9.4f %-7.2f %d",
                    r$family, r$re_kind, r$gradient_mode, r$status,
                    r$d_cross, r$d_within, r$ratio, r$n_par)
  }
  cat(line, "\n", sep = "")
}

invisible(results)
