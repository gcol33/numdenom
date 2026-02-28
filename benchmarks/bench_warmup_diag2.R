# bench_warmup_diag2.R
library(numdenom)
set.seed(42)

N <- 500; S <- 50
df <- data.frame(site = rep(1:S, each = N/S), x = rnorm(N))
eta <- 1.5 + 0.3 * df$x + rnorm(S, 0, 0.3)[df$site]
df$y_num <- rnbinom(N, size = 5, mu = exp(eta))
df$y_denom <- pmax(rnbinom(N, size = 5, mu = exp(eta + 0.5)), 1)
df$effort <- pmax(rgamma(N, shape = 5, rate = 5 / exp(eta + 0.5)), 0.01)

cat("\n=== NB+RE ===\n")
t1 <- system.time({
  fit_nb <- ratiod(y_num | y_denom ~ x + (1 | site), data = df,
    family = ratiod_negbin_negbin(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})["elapsed"]

# Check what diagnostics are available
cat("Diagnostics names:", names(fit_nb$diagnostics), "\n")
td <- fit_nb$diagnostics$treedepth
if (is.list(td)) td <- td[[1]]
cat("TD length:", length(td), "  first 5:", head(td, 5), "  last 5:", tail(td, 5), "\n")
n_lf <- fit_nb$diagnostics$n_leapfrog
if (is.list(n_lf)) n_lf <- n_lf[[1]]
cat("n_leapfrog length:", length(n_lf), "\n")
if (!is.null(n_lf) && length(n_lf) > 0) {
  cat("  first 5:", head(n_lf, 5), "  last 5:", tail(n_lf, 5), "\n")
  cat("  Total LF:", sum(n_lf, na.rm = TRUE), "\n")
}
cat(sprintf("  Time: %.2f s\n", t1))

# Use n_leapfrog from diagnostics directly
total_lf_nb <- sum(n_lf, na.rm = TRUE)
cat(sprintf("  True ms/kLF: %.2f\n", 1000 * t1 / total_lf_nb))

cat("\n=== PG+RE ===\n")
t2 <- system.time({
  fit_pg <- ratiod(y_num | effort ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})["elapsed"]
n_lf2 <- fit_pg$diagnostics$n_leapfrog
if (is.list(n_lf2)) n_lf2 <- n_lf2[[1]]
cat("n_leapfrog length:", length(n_lf2), "\n")
if (!is.null(n_lf2) && length(n_lf2) > 0) {
  cat("  first 5:", head(n_lf2, 5), "  last 5:", tail(n_lf2, 5), "\n")
  cat("  Total LF:", sum(n_lf2, na.rm = TRUE), "\n")
}
cat(sprintf("  Time: %.2f s\n", t2))
total_lf_pg <- sum(n_lf2, na.rm = TRUE)
cat(sprintf("  True ms/kLF: %.2f\n", 1000 * t2 / total_lf_pg))
