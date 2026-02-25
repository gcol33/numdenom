# Benchmark: numdenom (optimized NUTS) vs Stan
# Tests whether the NUTS infrastructure optimization closed the 2-4x gap
#
# Uses custom joint Stan models (NOT brms) per benchmarking rules
# Standard parameters: N=500, 500 iter, 250 warmup, 1 chain

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50

# Generate data
x <- rnorm(N_OBS)
site <- rep(1:N_SITES, each = N_OBS / N_SITES)

# PG data
y_pg <- rpois(N_OBS, lambda = exp(2 + 0.3 * x))
effort <- rgamma(N_OBS, shape = 10, rate = 1)
effort[effort < 0.01] <- 0.01

# NB data
y_nb_num <- rnbinom(N_OBS, mu = exp(2 + 0.3 * x), size = 5)
y_nb_denom <- rnbinom(N_OBS, mu = 100, size = 10)
y_nb_denom[y_nb_denom == 0] <- 1

df_pg <- data.frame(y = y_pg, effort = effort, x = x, site = factor(site))
df_nb <- data.frame(y_num = y_nb_num, y_denom = y_nb_denom, x = x, site = factor(site))

# Second grouping for crossed RE
site2 <- sample(1:10, N_OBS, replace = TRUE)
df_nb$site2 <- factor(site2)

# Design matrix for Stan
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

cat("================================================================\n")
cat("NUMDENOM vs STAN TIMING BENCHMARK (post-NUTS optimization)\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat("================================================================\n\n")

results <- data.frame(
  model = character(),
  numdenom_s = numeric(),
  stan_s = numeric(),
  ratio = numeric(),
  stringsAsFactors = FALSE
)

# Helper: time a numdenom fit
time_nd <- function(formula, data, family, ...) {
  system.time({
    fit <- ratiod(formula, data = data, family = family,
                  iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
                  gradient_mode = "H", verbose = FALSE, ...)
  })[3]
}

# Helper: time a Stan fit (using pre-compiled .exe if available)
time_stan <- function(stan_file, stan_data) {
  stan_path <- file.path("benchmarks/stan", stan_file)
  mod <- cmdstan_model(stan_path)
  t <- system.time({
    fit <- mod$sample(
      data = stan_data,
      iter_sampling = N_ITER - N_WARMUP,
      iter_warmup = N_WARMUP,
      chains = N_CHAINS,
      refresh = 0,
      show_messages = FALSE,
      show_exceptions = FALSE
    )
  })[3]
  t
}

# ============================================================
# 1. PG base (Row 1)
# ============================================================
cat("1. PG base... ")
t_nd <- time_nd(y | effort ~ x, df_pg, ratiod_poisson_gamma())
t_stan <- time_stan("joint_pg_base.stan", list(
  N = N_OBS, y_num = y_pg, y_denom = effort, p = 2, X = X
))
results <- rbind(results, data.frame(
  model = "PG base", numdenom_s = t_nd, stan_s = t_stan,
  ratio = round(t_nd / t_stan, 2)
))
cat(sprintf("nd=%.2fs, stan=%.2fs, ratio=%.2f\n", t_nd, t_stan, t_nd / t_stan))

# ============================================================
# 2. NB base (Row 31)
# ============================================================
cat("2. NB base... ")
t_nd <- time_nd(y_num | y_denom ~ x, df_nb, ratiod_negbin_negbin())
t_stan <- time_stan("joint_nb_base.stan", list(
  N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X
))
results <- rbind(results, data.frame(
  model = "NB base", numdenom_s = t_nd, stan_s = t_stan,
  ratio = round(t_nd / t_stan, 2)
))
cat(sprintf("nd=%.2fs, stan=%.2fs, ratio=%.2f\n", t_nd, t_stan, t_nd / t_stan))

# ============================================================
# 3. PG + RE (Row 2)
# ============================================================
cat("3. PG + RE... ")
t_nd <- time_nd(y | effort ~ x + (1 | site), df_pg, ratiod_poisson_gamma())
t_stan <- time_stan("joint_pg_re.stan", list(
  N = N_OBS, y_num = y_pg, y_denom = effort, p = 2, X = X,
  n_groups = N_SITES, group_idx = as.integer(df_pg$site)
))
results <- rbind(results, data.frame(
  model = "PG + RE", numdenom_s = t_nd, stan_s = t_stan,
  ratio = round(t_nd / t_stan, 2)
))
cat(sprintf("nd=%.2fs, stan=%.2fs, ratio=%.2f\n", t_nd, t_stan, t_nd / t_stan))

# ============================================================
# 4. NB + RE (Row 32)
# ============================================================
cat("4. NB + RE... ")
t_nd <- time_nd(y_num | y_denom ~ x + (1 | site), df_nb, ratiod_negbin_negbin())
t_stan <- time_stan("joint_nb_re.stan", list(
  N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X,
  n_groups = N_SITES, group_idx = as.integer(df_nb$site)
))
results <- rbind(results, data.frame(
  model = "NB + RE", numdenom_s = t_nd, stan_s = t_stan,
  ratio = round(t_nd / t_stan, 2)
))
cat(sprintf("nd=%.2fs, stan=%.2fs, ratio=%.2f\n", t_nd, t_stan, t_nd / t_stan))

# ============================================================
# 5. NB + crossed RE (Row 34)
# ============================================================
cat("5. NB + crossed... ")
t_nd <- time_nd(y_num | y_denom ~ x + (1 | site) + (1 | site2), df_nb, ratiod_negbin_negbin())
t_stan <- time_stan("joint_nb_crossed.stan", list(
  N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X,
  n_groups1 = N_SITES, group_idx1 = as.integer(df_nb$site),
  n_groups2 = 10, group_idx2 = as.integer(df_nb$site2)
))
results <- rbind(results, data.frame(
  model = "NB + crossed", numdenom_s = t_nd, stan_s = t_stan,
  ratio = round(t_nd / t_stan, 2)
))
cat(sprintf("nd=%.2fs, stan=%.2fs, ratio=%.2f\n", t_nd, t_stan, t_nd / t_stan))

# ============================================================
# 6. NB + ICAR (Row 35)
# ============================================================
cat("6. NB + ICAR... ")
t_nd <- time_nd(y_num | y_denom ~ x + (1 | site), df_nb, ratiod_negbin_negbin(),
                spatial = spatial_car(adj_mat, group_var = "site"))
t_stan <- time_stan("joint_nb_icar.stan", list(
  N = N_OBS, y_num = y_nb_num, y_denom = y_nb_denom, p = 2, X = X,
  n_groups = N_SITES, group_idx = as.integer(df_nb$site),
  J = N_SITES, spatial_idx = as.integer(df_nb$site),
  n_edges = n_edges, edge1 = edges[, 1], edge2 = edges[, 2],
  n_neighbors = as.array(n_neighbors)
))
results <- rbind(results, data.frame(
  model = "NB + ICAR", numdenom_s = t_nd, stan_s = t_stan,
  ratio = round(t_nd / t_stan, 2)
))
cat(sprintf("nd=%.2fs, stan=%.2fs, ratio=%.2f\n", t_nd, t_stan, t_nd / t_stan))

# ============================================================
# Summary
# ============================================================
cat("\n================================================================\n")
cat("SUMMARY\n")
cat("================================================================\n")
print(results, row.names = FALSE)
cat(sprintf("\nMean ratio (numdenom/Stan): %.2f\n", mean(results$ratio)))
cat(sprintf("Min ratio: %.2f, Max ratio: %.2f\n", min(results$ratio), max(results$ratio)))
cat("\nratio < 1 means numdenom is FASTER than Stan\n")
cat("ratio = 1 means parity\n")
cat("ratio > 1 means Stan is faster\n")
