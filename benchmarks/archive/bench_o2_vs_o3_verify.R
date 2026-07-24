#!/usr/bin/env Rscript
# Verify the most divergent O2 vs O3 results with 3 reps
# Pass "O2" or "O3" as argument
args <- commandArgs(trailingOnly = TRUE)
label <- if (length(args) > 0) args[1] else "unknown"

library(numdenom)

set.seed(42)
N <- 500; S <- 50; T_max <- 20
site <- rep(1:S, length.out = N)
region <- rep(1:10, each = N/10)
x1 <- rnorm(N)
time_idx <- rep(1:T_max, length.out = N)
adj_mat <- matrix(0L, S, S)
for (i in 1:(S - 1)) { adj_mat[i, i + 1] <- 1L; adj_mat[i + 1, i] <- 1L }

lambda_num <- exp(1 + 0.3 * x1)
lambda_denom <- exp(2 + 0.1 * x1)
y_num <- rnbinom(N, mu = lambda_num, size = 5)
y_denom <- rnbinom(N, mu = lambda_denom, size = 5)
y_denom[y_denom == 0] <- 1
trials <- rpois(N, 20) + 10
successes <- rbinom(N, size = trials, prob = 0.3)

df <- data.frame(y_num=y_num, y_denom=y_denom, successes=successes, trials=trials,
                 x1=x1, site=factor(site), region=factor(region), time=time_idx)

run_model <- function(name, f, fam, sp = NULL, temp = NULL, seed) {
  set.seed(seed)
  family <- switch(fam, pg=ratiod_poisson_gamma(), nb=ratiod_negbin_negbin(), bin=ratiod_binomial())
  spatial <- if (!is.null(sp)) switch(sp, icar=spatial_car(adj=adj_mat, group_var="site"),
                                           bym2=spatial_bym2(adj=adj_mat, group_var="site")) else NULL
  temporal <- if (!is.null(temp)) switch(temp, rw1=temporal_rw1(time_var="time")) else NULL
  system.time({
    tratio(f, data=df, family=family, spatial=spatial, temporal=temporal,
           mode="hmc",
           control = list(iter=500, warmup=250, chains=1, gradient_mode="H", verbose=FALSE))
  })["elapsed"]
}

# Models with biggest divergence between O2 and O3
models <- list(
  list(name="pg_slopes_corr", f=y_num|y_denom~x1+(1+x1|site), fam="pg"),
  list(name="pg_slopes_uncorr", f=y_num|y_denom~x1+(1+x1||site), fam="pg"),
  list(name="nb_icar", f=y_num|y_denom~x1, fam="nb", sp="icar"),
  list(name="nb_rw1", f=y_num|y_denom~x1, fam="nb", temp="rw1"),
  list(name="nb_icar_rw1", f=y_num|y_denom~x1, fam="nb", sp="icar", temp="rw1"),
  list(name="nb_bym2", f=y_num|y_denom~x1, fam="nb", sp="bym2"),
  list(name="nb_base", f=y_num|y_denom~x1, fam="nb"),
  list(name="nb_crossed", f=y_num|y_denom~x1+(1|site)+(1|region), fam="nb"),
  list(name="pg_icar", f=y_num|y_denom~x1, fam="pg", sp="icar"),
  list(name="pg_crossed", f=y_num|y_denom~x1+(1|site)+(1|region), fam="pg")
)

cat(sprintf("=== %s Verify (3 reps, N=%d, 500 iter) ===\n\n", label, N))
cat(sprintf("%-20s  %6s  %6s  %6s  %8s\n", "Model", "r1", "r2", "r3", "median"))
cat(paste(rep("-", 55), collapse=""), "\n")

results <- list()
for (m in models) {
  times <- numeric(3)
  for (r in 1:3) {
    times[r] <- tryCatch(
      run_model(m$name, m$f, m$fam, m$sp, m$temp, seed = 100 + r),
      error = function(e) { cat("  ERROR:", e$message, "\n"); NA_real_ }
    )
  }
  med <- median(times, na.rm = TRUE)
  cat(sprintf("%-20s  %6.1f  %6.1f  %6.1f  %8.1f\n", m$name, times[1], times[2], times[3], med))
  results[[m$name]] <- med
}

saveRDS(results, sprintf("benchmarks/o2_vs_o3_verify_%s.rds", label))
cat(sprintf("\nSaved to benchmarks/o2_vs_o3_verify_%s.rds\n", label))
