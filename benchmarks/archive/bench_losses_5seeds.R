#!/usr/bin/env Rscript
# Run 4 LOSS models with 5 seeds each to get stable medians
library(numdenom)

N <- 500L; N_s <- 50L; N_t <- 20L

run_seeds <- function(label, fit_fn, seeds = 1:5) {
  cat(sprintf("\n=== %s ===\n", label))
  times <- numeric(length(seeds))
  divs <- numeric(length(seeds))
  epsilons <- numeric(length(seeds))
  for (i in seq_along(seeds)) {
    set.seed(seeds[i])
    # Regenerate data each seed for robustness
    site <- factor(rep(1:N_s, length.out = N))
    time_num <- rep(1:N_t, length.out = N)
    x <- rnorm(N)
    n_side <- ceiling(sqrt(N_s))
    grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
    si <- as.integer(site)
    lon <- grid$lon[si]; lat <- grid$lat[si]
    adj_mat <- matrix(0L, N_s, N_s)
    for (ii in 1:N_s) for (jj in 1:N_s) {
      if (ii != jj && sqrt((grid$lon[ii]-grid$lon[jj])^2+(grid$lat[ii]-grid$lat[jj])^2) <= 1.5)
        adj_mat[ii, jj] <- 1L
    }
    env <- list(x=x, site=site, time_num=time_num, lon=lon, lat=lat,
                spatial_site=site, adj_mat=adj_mat)
    t <- system.time({ fit <- fit_fn(env) })[["elapsed"]]
    d <- tryCatch(sum(unlist(lapply(fit$fit_raw$divergent, sum))), error=function(e) NA)
    times[i] <- t; divs[i] <- d
    cat(sprintf("  seed=%d: %.1fs  div=%s\n", seeds[i], t, ifelse(is.na(d), "?", as.character(d))))
  }
  cat(sprintf("  MEDIAN: %.1fs  div=%s\n",
              median(times), median(divs, na.rm=TRUE)))
  list(label=label, times=times, median=median(times), divs=divs)
}

results <- list()

# 1. NB+ICAR
results[[1]] <- run_seeds("NB+ICAR", function(env) {
  df <- data.frame(y=rnbinom(N, mu=exp(2+0.3*env$x), size=5),
                   denom={d<-rnbinom(N, mu=100, size=10); d[d==0]<-1L; d},
                   x=env$x, site=env$site, spatial_site=env$spatial_site)
  ratiod(y | denom ~ x + (1|site), data=df, family=ratiod_negbin_negbin(),
         spatial=spatial_car(env$adj_mat, level="group", group_var="spatial_site"),
         iter=500, warmup=250, chains=1, verbose=FALSE)
})

# 2. NB+HSGP (no RE — matching Stan which has no RE)
results[[2]] <- run_seeds("NB+HSGP", function(env) {
  df <- data.frame(y=rnbinom(N, mu=exp(2+0.3*env$x), size=5),
                   denom={d<-rnbinom(N, mu=100, size=10); d[d==0]<-1L; d},
                   x=env$x, lon=env$lon, lat=env$lat)
  ratiod(y | denom ~ x, data=df, family=ratiod_negbin_negbin(),
         spatial=spatial_hsgp(coords = ~ lon + lat),
         iter=500, warmup=250, chains=1, verbose=FALSE)
})

# 3. PG+GP_t (no RE — matching Stan which has no RE)
results[[3]] <- run_seeds("PG+GP_t", function(env) {
  df <- data.frame(y=rpois(N, exp(2+0.5*env$x)),
                   denom=rgamma(N, 10, 1),
                   x=env$x, time_num=env$time_num)
  ratiod(y | denom ~ x, data=df, family=ratiod_poisson_gamma(),
         temporal=temporal_gp("time_num"),
         iter=500, warmup=250, chains=1, verbose=FALSE)
})

# 4. Bin+GP_t (no RE — matching Stan which has no RE)
results[[4]] <- run_seeds("Bin+GP_t", function(env) {
  trials <- sample(10:50, N, replace=TRUE)
  df <- data.frame(y=rbinom(N, trials, plogis(0.5+0.3*env$x)),
                   trials=trials, x=env$x, time_num=env$time_num)
  ratiod(y | trials ~ x, data=df, family=ratiod_binomial(),
         temporal=temporal_gp("time_num"),
         iter=500, warmup=250, chains=1, verbose=FALSE)
})

cat("\n\n========== SUMMARY ==========\n")
stan_ref <- c("NB+ICAR"=3.5, "NB+HSGP"=2.9, "PG+GP_t"=10.0, "Bin+GP_t"=3.3)
cat(sprintf("%-15s %8s %8s %8s %6s\n", "Model", "Median", "Stan", "Ratio", ""))
cat(paste0(rep("-", 50), collapse=""), "\n")
for (r in results) {
  stan <- stan_ref[r$label]
  ratio <- r$median / stan
  v <- if (ratio <= 1) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
  cat(sprintf("%-15s %7.1fs %7.1fs %7.1fx %s  [%s]\n",
              r$label, r$median, stan, 1/ratio, v,
              paste(round(r$times, 1), collapse=", ")))
}
