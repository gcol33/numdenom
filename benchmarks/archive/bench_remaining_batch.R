# Benchmark remaining configurations in batches
# Run with: Rscript benchmarks/bench_remaining_batch.R <batch_number>
# Standard parameters: N=500, iter=500, warmup=250, chains=1

library(numdenom)
set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
batch_num <- if (length(args) > 0) as.integer(args[1]) else 1

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

# Create spatial grid
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Adjacency matrix
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))

# Data for each family
y_pg <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y=y_pg, effort=effort, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

y_nb <- rnbinom(N, mu=exp(2 + 0.3*x), size=5)
denom_nb <- rnbinom(N, mu=100, size=10)
denom_nb[denom_nb == 0] <- 1
df_nb <- data.frame(y=y_nb, denom=denom_nb, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

trials <- sample(10:50, N, replace=TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y=y_bin, trials=trials, x=x, site=site, time=time,
                     lon=lon, lat=lat, spatial_site=spatial_site)

# gamma_gamma data
y_gamma <- rgamma(N, shape=5, rate=0.5)
denom_gamma <- rgamma(N, shape=10, rate=1)
df_gamma <- data.frame(y=y_gamma, denom=denom_gamma, x=x, site=site, time=time,
                       lon=lon, lat=lat, spatial_site=spatial_site)

# lognormal data
y_ln <- rlnorm(N, meanlog=2 + 0.3*x, sdlog=0.5)
denom_ln <- rlnorm(N, meanlog=3, sdlog=0.3)
df_ln <- data.frame(y=y_ln, denom=denom_ln, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

# beta_binomial data (same as binomial)
df_bb <- df_bin

bench_one <- function(name, row, ...) {
  cat(sprintf("\n=== Row %d: %s ===\n", row, name))
  result <- list(row=row, name=name)
  cat("  H: ")
  flush.console()
  tryCatch({
    time <- system.time({
      fit <- tratio(..., control = list(iter=500, warmup=250, chains=1, verbose=FALSE, gradient_mode="H"))
    })["elapsed"]
    cat(sprintf("%.1fs\n", time))
    result$H <- time
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    result$H <- NA
    result$H_error <- conditionMessage(e)
  })
  result
}

# Define all remaining benchmarks
remaining_benchmarks <- list(
  # Batch 1: negbin GP/HSGP/MSGP/pCAR (rows 37-40)
  list(batch=1, row=37, name="nb_gp",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_gp(~ lon + lat)))),
  list(batch=1, row=38, name="nb_hsgp",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_hsgp(~ lon + lat, m=8)))),
  list(batch=1, row=39, name="nb_msgp",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_multiscale(~ lon + lat)))),
  list(batch=1, row=40, name="nb_pcar",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_car(adj_mat, proper=TRUE, level="group", group_var="spatial_site")))),

  # Batch 2: negbin temporal GP/MS + GP combos (rows 44-45, 51-53)
  list(batch=2, row=44, name="nb_gp_t",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         temporal=temporal_gp("time")))),
  list(batch=2, row=45, name="nb_ms_t",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         temporal=temporal_multiscale("time")))),
  list(batch=2, row=51, name="nb_gp_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_gp(~ lon + lat),
                         temporal=temporal_rw1("time")))),
  list(batch=2, row=52, name="nb_hsgp_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_hsgp(~ lon + lat, m=8),
                         temporal=temporal_rw1("time")))),
  list(batch=2, row=53, name="nb_msgp_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatial=spatial_multiscale(~ lon + lat),
                         temporal=temporal_rw1("time")))),

  # Batch 3: negbin ST + latent (rows 58-60)
  list(batch=3, row=58, name="nb_st1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatiotemporal=spatiotemporal(
                           spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                           temporal=temporal_rw1("time"),
                           type="I")))),
  list(batch=3, row=59, name="nb_st4",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         spatiotemporal=spatiotemporal(
                           spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                           temporal=temporal_rw1("time"),
                           type="IV")))),
  list(batch=3, row=60, name="nb_latent",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                         latent=latent_factor(n_factors=2)))),

  # Batch 4: binomial GP/HSGP/MSGP/pCAR (rows 67-70)
  list(batch=4, row=67, name="bin_gp",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_gp(~ lon + lat)))),
  list(batch=4, row=68, name="bin_hsgp",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_hsgp(~ lon + lat, m=8)))),
  list(batch=4, row=69, name="bin_msgp",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_multiscale(~ lon + lat)))),
  list(batch=4, row=70, name="bin_pcar",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_car(adj_mat, proper=TRUE, level="group", group_var="spatial_site")))),

  # Batch 5: binomial temporal GP/MS + GP combos (rows 74-75, 83-85)
  list(batch=5, row=74, name="bin_gp_t",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         temporal=temporal_gp("time")))),
  list(batch=5, row=75, name="bin_ms_t",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         temporal=temporal_multiscale("time")))),
  list(batch=5, row=83, name="bin_gp_rw1",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_gp(~ lon + lat),
                         temporal=temporal_rw1("time")))),
  list(batch=5, row=84, name="bin_hsgp_rw1",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_hsgp(~ lon + lat, m=8),
                         temporal=temporal_rw1("time")))),
  list(batch=5, row=85, name="bin_msgp_rw1",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatial=spatial_multiscale(~ lon + lat),
                         temporal=temporal_rw1("time")))),

  # Batch 6: binomial ST + latent (rows 90-92)
  list(batch=6, row=90, name="bin_st1",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatiotemporal=spatiotemporal(
                           spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                           temporal=temporal_rw1("time"),
                           type="I")))),
  list(batch=6, row=91, name="bin_st4",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         spatiotemporal=spatiotemporal(
                           spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                           temporal=temporal_rw1("time"),
                           type="IV")))),
  list(batch=6, row=92, name="bin_latent",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                         latent=latent_factor(n_factors=2)))),

  # Batch 7: poisson_gamma slow ones (rows 9, 21, 23, 28-30)
  list(batch=7, row=9, name="pg_msgp",
       call=quote(tratio(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                         spatial=spatial_multiscale(~ lon + lat)))),
  list(batch=7, row=21, name="pg_gp_rw1",
       call=quote(tratio(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                         spatial=spatial_gp(~ lon + lat),
                         temporal=temporal_rw1("time")))),
  list(batch=7, row=23, name="pg_msgp_rw1",
       call=quote(tratio(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                         spatial=spatial_multiscale(~ lon + lat),
                         temporal=temporal_rw1("time")))),
  list(batch=7, row=28, name="pg_st1",
       call=quote(tratio(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                         spatiotemporal=spatiotemporal(
                           spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                           temporal=temporal_rw1("time"),
                           type="I")))),
  list(batch=7, row=29, name="pg_st4",
       call=quote(tratio(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                         spatiotemporal=spatiotemporal(
                           spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                           temporal=temporal_rw1("time"),
                           type="IV")))),

  # Batch 8: poisson_gamma latent
  list(batch=8, row=30, name="pg_latent",
       call=quote(tratio(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                         latent=latent_factor(n_factors=2)))),

  # Batch 9: gamma_gamma (rows 93-97)
  list(batch=9, row=93, name="gg_base",
       call=quote(tratio(y | denom ~ x, data=df_gamma, family=ratiod_gamma_gamma()))),
  list(batch=9, row=94, name="gg_re",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_gamma, family=ratiod_gamma_gamma()))),
  list(batch=9, row=95, name="gg_icar",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_gamma, family=ratiod_gamma_gamma(),
                         spatial=spatial_car(adj_mat, level="group", group_var="spatial_site")))),
  list(batch=9, row=96, name="gg_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_gamma, family=ratiod_gamma_gamma(),
                         temporal=temporal_rw1("time")))),
  list(batch=9, row=97, name="gg_icar_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_gamma, family=ratiod_gamma_gamma(),
                         spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                         temporal=temporal_rw1("time")))),

  # Batch 10: lognormal (rows 98-102)
  list(batch=10, row=98, name="ln_base",
       call=quote(tratio(y | denom ~ x, data=df_ln, family=ratiod_lognormal()))),
  list(batch=10, row=99, name="ln_re",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_ln, family=ratiod_lognormal()))),
  list(batch=10, row=100, name="ln_icar",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_ln, family=ratiod_lognormal(),
                         spatial=spatial_car(adj_mat, level="group", group_var="spatial_site")))),
  list(batch=10, row=101, name="ln_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_ln, family=ratiod_lognormal(),
                         temporal=temporal_rw1("time")))),
  list(batch=10, row=102, name="ln_icar_rw1",
       call=quote(tratio(y | denom ~ x + (1 | site), data=df_ln, family=ratiod_lognormal(),
                         spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                         temporal=temporal_rw1("time")))),

  # Batch 11: beta_binomial (rows 103-107)
  list(batch=11, row=103, name="bb_base",
       call=quote(tratio(y | trials ~ x, data=df_bb, family=ratiod_beta_binomial()))),
  list(batch=11, row=104, name="bb_re",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bb, family=ratiod_beta_binomial()))),
  list(batch=11, row=105, name="bb_icar",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bb, family=ratiod_beta_binomial(),
                         spatial=spatial_car(adj_mat, level="group", group_var="spatial_site")))),
  list(batch=11, row=106, name="bb_rw1",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bb, family=ratiod_beta_binomial(),
                         temporal=temporal_rw1("time")))),
  list(batch=11, row=107, name="bb_icar_rw1",
       call=quote(tratio(y | trials ~ x + (1 | site), data=df_bb, family=ratiod_beta_binomial(),
                         spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                         temporal=temporal_rw1("time"))))
)

# Filter to requested batch
batch_tests <- Filter(function(x) x$batch == batch_num, remaining_benchmarks)

if (length(batch_tests) == 0) {
  cat(sprintf("No tests for batch %d. Valid batches: 1-11\n", batch_num))
  q(save="no")
}

cat(sprintf("=== BATCH %d: %d tests ===\n", batch_num, length(batch_tests)))

results <- list()
errors <- list()

# Benchmark parameters (shorter for benchmarks)
ITER <- 500
WARMUP <- 250
CHAINS <- 1

for (test in batch_tests) {
  cat(sprintf("\n--- Row %d: %s ---\n", test$row, test$name))
  cat("  H: ")
  flush.console()
  tryCatch({
    # Add benchmark parameters to the call
    call_with_params <- test$call
    call_with_params$iter <- ITER
    call_with_params$warmup <- WARMUP
    call_with_params$chains <- CHAINS
    call_with_params$verbose <- FALSE
    call_with_params$gradient_mode <- "H"

    time <- system.time({
      fit <- eval(call_with_params)
    })["elapsed"]
    cat(sprintf("%.1fs\n", time))
    results[[as.character(test$row)]] <- list(
      row = test$row,
      name = test$name,
      H = time
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    errors[[as.character(test$row)]] <- list(
      row = test$row,
      name = test$name,
      error = conditionMessage(e)
    )
  })
}

# Print summary
cat("\n\n=== BATCH", batch_num, "RESULTS ===\n")
cat(sprintf("%-6s %-20s %8s %s\n", "Row", "Name", "H(s)", "Status"))
cat(paste(rep("-", 50), collapse=""), "\n")

for (test in batch_tests) {
  r <- results[[as.character(test$row)]]
  e <- errors[[as.character(test$row)]]
  if (!is.null(r)) {
    cat(sprintf("%-6d %-20s %8.1f OK\n", test$row, test$name, r$H))
  } else if (!is.null(e)) {
    cat(sprintf("%-6d %-20s %8s ERROR: %s\n", test$row, test$name, "-",
                substr(e$error, 1, 40)))
  }
}

if (length(errors) > 0) {
  cat("\n=== ERRORS ===\n")
  for (e in errors) {
    cat(sprintf("Row %d (%s): %s\n", e$row, e$name, e$error))
  }
}

# Save results
outfile <- sprintf("benchmarks/results_batch%d.rds", batch_num)
saveRDS(list(results=results, errors=errors), outfile)
cat(sprintf("\nResults saved to %s\n", outfile))
