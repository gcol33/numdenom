# Benchmark: numdenom (optimized NUTS) vs Stan — 3 replications, median timing
# Proper before/after comparison using historical data from gradient_methods.md
#
# Historical "before" times (pre-optimization, with NUTS):
#   Row 1  pg_base:    nd=1.9s,  Stan=1.2s  (ratio 1.58)
#   Row 2  pg_re:      nd=4.7s,  Stan=2.5s  (ratio 1.88)
#   Row 4  pg_crossed: nd=11.3s, Stan=2.9s  (ratio 3.90)
#   Row 31 nb_base:    nd=3.3s,  Stan=1.5s  (ratio 2.20)
#   Row 32 nb_re:      nd=6.9s,  Stan=2.9s  (ratio 2.38)
#   Row 34 nb_crossed: nd=20.3s, Stan=9.9s  (ratio 2.05)
#   Row 35 nb_icar:    nd=5.9s,  Stan=7.9s  (ratio 0.75)

library(numdenom)
library(cmdstanr)

set.seed(42)

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_SITES <- 50
N_REPS <- 3

# Generate data (same seed as standard benchmarks)
x <- rnorm(N_OBS)
site <- rep(1:N_SITES, each = N_OBS / N_SITES)

y_pg <- rpois(N_OBS, lambda = exp(2 + 0.3 * x))
effort <- rgamma(N_OBS, shape = 10, rate = 1)
effort[effort < 0.01] <- 0.01

y_nb_num <- rnbinom(N_OBS, mu = exp(2 + 0.3 * x), size = 5)
y_nb_denom <- rnbinom(N_OBS, mu = 100, size = 10)
y_nb_denom[y_nb_denom == 0] <- 1

df_pg <- data.frame(y = y_pg, effort = effort, x = x, site = factor(site))
df_nb <- data.frame(y_num = y_nb_num, y_denom = y_nb_denom, x = x, site = factor(site))

site2 <- sample(1:10, N_OBS, replace = TRUE)
df_nb$site2 <- factor(site2)

X <- cbind(1, x)

# Adjacency for ICAR
adj_mat <- matrix(0, N_SITES, N_SITES)
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j && sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2) <= 1.5)
      adj_mat[i, j] <- 1
  }
}
edges <- which(adj_mat == 1 & upper.tri(adj_mat), arr.ind = TRUE)
n_edges <- nrow(edges)
n_neighbors <- rowSums(adj_mat)

# Pre-compile all Stan models once (exclude from timing)
cat("Pre-compiling Stan models...\n")
stan_models <- list()
for (f in c("joint_pg_base.stan", "joint_nb_base.stan",
            "joint_pg_re.stan", "joint_nb_re.stan",
            "joint_nb_crossed.stan", "joint_nb_icar.stan")) {
  stan_models[[f]] <- cmdstan_model(file.path("benchmarks/stan", f))
}
cat("Done.\n\n")

# Timing helpers
time_nd <- function(formula, data, family, ...) {
  system.time({
    tratio(formula, data = data, family = family, ...,
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = 1, gradient_mode = "H", verbose = FALSE))
  })[3]
}

time_stan <- function(stan_file, stan_data) {
  mod <- stan_models[[stan_file]]
  system.time({
    mod$sample(data = stan_data,
               iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
               chains = 1, refresh = 0, show_messages = FALSE,
               show_exceptions = FALSE)
  })[3]
}

# Model configs
configs <- list(
  list(name = "PG base", row = 1,
       nd = function() time_nd(y | effort ~ x, df_pg, ratiod_poisson_gamma()),
       stan = function() time_stan("joint_pg_base.stan",
         list(N = N_OBS, y_num = y_pg, y_denom = effort, p = 2, X = X)),
       old_nd = 1.9, old_stan = 1.2),

  list(name = "NB base", row = 31,
       nd = function() time_nd(y_num | y_denom ~ x, df_nb, ratiod_negbin_negbin()),
       stan = function() time_stan("joint_nb_base.stan",
         list(N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X)),
       old_nd = 3.3, old_stan = 1.5),

  list(name = "PG + RE", row = 2,
       nd = function() time_nd(y | effort ~ x + (1 | site), df_pg, ratiod_poisson_gamma()),
       stan = function() time_stan("joint_pg_re.stan",
         list(N = N_OBS, y_num = y_pg, y_denom = effort, p = 2, X = X,
              n_groups = N_SITES, group_idx = as.integer(df_pg$site))),
       old_nd = 4.7, old_stan = 2.5),

  list(name = "NB + RE", row = 32,
       nd = function() time_nd(y_num | y_denom ~ x + (1 | site), df_nb, ratiod_negbin_negbin()),
       stan = function() time_stan("joint_nb_re.stan",
         list(N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X,
              n_groups = N_SITES, group_idx = as.integer(df_nb$site))),
       old_nd = 6.9, old_stan = 2.9),

  list(name = "NB + crossed", row = 34,
       nd = function() time_nd(y_num | y_denom ~ x + (1 | site) + (1 | site2), df_nb, ratiod_negbin_negbin()),
       stan = function() time_stan("joint_nb_crossed.stan",
         list(N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X,
              n_groups1 = N_SITES, group_idx1 = as.integer(df_nb$site),
              n_groups2 = 10, group_idx2 = as.integer(df_nb$site2))),
       old_nd = 20.3, old_stan = 9.9),

  list(name = "NB + ICAR", row = 35,
       nd = function() time_nd(y_num | y_denom ~ x + (1 | site), df_nb, ratiod_negbin_negbin(),
                               spatial = spatial_car(adj_mat, group_var = "site")),
       stan = function() time_stan("joint_nb_icar.stan",
         list(N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X,
              n_groups = N_SITES, group_idx = as.integer(df_nb$site),
              J = N_SITES, spatial_idx = as.integer(df_nb$site),
              n_edges = n_edges, edge1 = edges[, 1], edge2 = edges[, 2],
              n_neighbors = as.array(n_neighbors))),
       old_nd = 5.9, old_stan = 7.9)
)

cat("================================================================\n")
cat("NUMDENOM vs STAN — 3 reps, median timing\n")
cat(sprintf("N=%d, iter=%d, warmup=%d\n", N_OBS, N_ITER, N_WARMUP))
cat("================================================================\n\n")

results <- data.frame(
  row = integer(), model = character(),
  old_nd = numeric(), new_nd = numeric(), nd_speedup = numeric(),
  stan = numeric(), old_ratio = numeric(), new_ratio = numeric(),
  stringsAsFactors = FALSE
)

for (cfg in configs) {
  cat(sprintf("Row %d: %s (%d reps)...\n", cfg$row, cfg$name, N_REPS))

  nd_times <- numeric(N_REPS)
  stan_times <- numeric(N_REPS)

  for (r in 1:N_REPS) {
    nd_times[r] <- cfg$nd()
    stan_times[r] <- cfg$stan()
    cat(sprintf("  rep %d: nd=%.2fs, stan=%.2fs\n", r, nd_times[r], stan_times[r]))
  }

  nd_med <- median(nd_times)
  stan_med <- median(stan_times)
  old_ratio <- cfg$old_nd / cfg$old_stan
  new_ratio <- nd_med / stan_med
  nd_speedup <- cfg$old_nd / nd_med

  results <- rbind(results, data.frame(
    row = cfg$row, model = cfg$name,
    old_nd = cfg$old_nd, new_nd = round(nd_med, 2),
    nd_speedup = round(nd_speedup, 2),
    stan = round(stan_med, 2),
    old_ratio = round(old_ratio, 2),
    new_ratio = round(new_ratio, 2)
  ))

  cat(sprintf("  median: nd=%.2fs (was %.1fs, %.1fx faster), stan=%.2fs\n",
              nd_med, cfg$old_nd, nd_speedup, stan_med))
  cat(sprintf("  ratio: %.2f (was %.2f)\n\n", new_ratio, old_ratio))
}

cat("\n================================================================\n")
cat("SUMMARY\n")
cat("================================================================\n\n")
cat("old_nd / new_nd = numdenom speedup from optimization\n")
cat("old_ratio / new_ratio = nd/Stan ratio (< 1 = numdenom faster)\n\n")
print(results, row.names = FALSE)
cat(sprintf("\nMean numdenom speedup: %.2fx\n", mean(results$nd_speedup)))
cat(sprintf("Mean old nd/Stan ratio: %.2f\n", mean(results$old_ratio)))
cat(sprintf("Mean new nd/Stan ratio: %.2f\n", mean(results$new_ratio)))
