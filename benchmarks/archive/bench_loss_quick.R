library(numdenom)

N <- 500L; N_s <- 50L; N_t <- 20L

run_model <- function(label, fit_fn, seeds = 1:3) {
  cat(sprintf("\n=== %s ===\n", label))
  times <- numeric(length(seeds))
  for (i in seq_along(seeds)) {
    set.seed(seeds[i])
    site <- factor(rep(1:N_s, length.out = N))
    time_num <- rep(1:N_t, length.out = N)
    x <- rnorm(N)
    n_side <- ceiling(sqrt(N_s))
    grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
    adj_mat <- matrix(0L, N_s, N_s)
    for (ii in 1:N_s) for (jj in 1:N_s) {
      if (ii != jj && sqrt((grid$lon[ii]-grid$lon[jj])^2+(grid$lat[ii]-grid$lat[jj])^2) <= 1.5)
        adj_mat[ii, jj] <- 1L
    }
    env <- list(x = x, site = site, time_num = time_num,
                spatial_site = site, adj_mat = adj_mat)
    t <- system.time({ fit <- fit_fn(env) })[["elapsed"]]
    times[i] <- t
    cat(sprintf("  seed=%d: %.1fs\n", seeds[i], t))
  }
  med <- median(times)
  cat(sprintf("  MEDIAN: %.1fs\n", med))
  med
}

results <- list()

# NB+ICAR (Stan 3.5s)
results[["NB+ICAR"]] <- run_model("NB+ICAR", function(env) {
  df <- data.frame(y = rnbinom(N, mu = exp(2 + 0.3*env$x), size = 5),
                   denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
                   x = env$x, site = env$site, spatial_site = env$spatial_site)
  tratio(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
         spatial = spatial_car(env$adj_mat, level = "group", group_var = "spatial_site"),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

# PG+GP_t (Stan 10.0s)
results[["PG+GP_t"]] <- run_model("PG+GP_t", function(env) {
  df <- data.frame(y = rpois(N, exp(2 + 0.5*env$x)),
                   denom = rgamma(N, 10, 1),
                   x = env$x, time_num = env$time_num)
  tratio(y | denom ~ x, data = df, family = ratiod_poisson_gamma(),
         temporal = temporal_gp("time_num"),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

# Bin+GP_t (Stan 3.3s)
results[["Bin+GP_t"]] <- run_model("Bin+GP_t", function(env) {
  trials <- sample(10:50, N, replace = TRUE)
  df <- data.frame(y = rbinom(N, trials, plogis(0.5 + 0.3*env$x)),
                   trials = trials, x = env$x, time_num = env$time_num)
  tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
         temporal = temporal_gp("time_num"),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

cat("\n\n========== RESULTS ==========\n")
stan_ref <- c("NB+ICAR" = 3.5, "PG+GP_t" = 10.0, "Bin+GP_t" = 3.3)
cat(sprintf("%-15s %8s %8s %8s %6s\n", "Model", "Median", "Stan", "Ratio", ""))
cat(paste0(rep("-", 50), collapse = ""), "\n")
for (nm in names(results)) {
  stan <- stan_ref[nm]
  ratio <- results[[nm]] / stan
  v <- if (ratio <= 1) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
  cat(sprintf("%-15s %7.1fs %7.1fs %7.2fx %s\n", nm, results[[nm]], stan, ratio, v))
}
