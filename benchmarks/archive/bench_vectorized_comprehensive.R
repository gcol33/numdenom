# Comprehensive vectorized gradient benchmark
# Tests all configurations that use vectorized dispatch
# N=500, 500 iter, 1 chain, H mode (NUTS)
.libPaths(c("C:/Users/Gilles Colling/AppData/Local/R/win-library/4.5", .libPaths()))
devtools::load_all()

cat("=== Comprehensive Vectorized Gradient Benchmark ===\n")
cat("Date:", format(Sys.time()), "\n\n")

N <- 500; S <- 50; T <- 10
set.seed(42)
x <- rnorm(N)
site <- factor(sample(1:S, N, replace = TRUE))
time <- factor(sample(1:T, N, replace = TRUE))

# --- NB data ---
y_num_nb <- rnbinom(N, mu = exp(1.0 + 0.5 * x), size = 3)
y_denom_nb <- rnbinom(N, mu = exp(0.8 + 0.2 * x), size = 5)
df_nb <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x, site = site, time = time)

# --- PG data ---
y_num_pg <- rpois(N, exp(0.5 + 0.3 * x))
y_denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(0.2 + 0.1 * x))
df_pg <- data.frame(y_num = y_num_pg, y_denom = y_denom_pg, x = x, site = site, time = time)

# --- Binomial data ---
n_trials <- sample(10:50, N, replace = TRUE)
y_binom <- rbinom(N, n_trials, plogis(0.3 + 0.5 * x))
df_bin <- data.frame(y = y_binom, n = n_trials, x = x, site = site, time = time)

# --- Adjacency for ICAR/BYM2 ---
adj <- Matrix::Matrix(0, S, S, sparse = TRUE)
for (i in 1:(S-1)) { adj[i, i+1] <- 1; adj[i+1, i] <- 1 }

timed <- function(desc, expr) {
  cat(sprintf("  %-35s", desc))
  gc(FALSE)
  t <- system.time(tryCatch(eval(expr), error = function(e) cat("ERROR:", e$message)))[["elapsed"]]
  cat(sprintf(" %7.2fs\n", t))
  t
}

results <- list()

# =====================================================
# FUSED PATH (p <= 4): base models, all 3 families
# =====================================================
cat("--- FUSED single-pass (p<=4) ---\n")

results$nb_base <- timed("NB base", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$pg_base <- timed("PG base", quote(
  ratiod(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$bin_base <- timed("Bin base", quote(
  ratiod(y | n ~ x, data = df_bin, family = ratiod_binomial(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# VECTORIZED 3-PASS: +RE models
# =====================================================
cat("\n--- VECTORIZED 3-pass (+RE) ---\n")

results$nb_re <- timed("NB + RE", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$pg_re <- timed("PG + RE", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg, family = ratiod_poisson_gamma(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$bin_re <- timed("Bin + RE", quote(
  ratiod(y | n ~ x + (1 | site), data = df_bin, family = ratiod_binomial(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# VECTORIZED 3-PASS: +ICAR
# =====================================================
cat("\n--- VECTORIZED 3-pass (+ICAR) ---\n")

results$nb_icar <- timed("NB + ICAR", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_car(adj, group_var = "site"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$pg_icar <- timed("PG + ICAR", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg, family = ratiod_poisson_gamma(),
         spatial = spatial_car(adj, group_var = "site"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# VECTORIZED 3-PASS: +BYM2
# =====================================================
cat("\n--- VECTORIZED 3-pass (+BYM2) ---\n")

results$nb_bym2 <- timed("NB + BYM2", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_bym2(adj, group_var = "site"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# VECTORIZED 3-PASS: +temporal
# =====================================================
cat("\n--- VECTORIZED 3-pass (+temporal) ---\n")

results$nb_rw1 <- timed("NB + RW1", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$pg_rw1 <- timed("PG + RW1", quote(
  ratiod(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         temporal = temporal_rw1(time_var = "time"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

results$nb_ar1 <- timed("NB + AR1", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_ar1(time_var = "time"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# VECTORIZED 3-PASS: +RE+temporal (combined)
# =====================================================
cat("\n--- VECTORIZED 3-pass (+RE+temporal) ---\n")

results$nb_re_rw1 <- timed("NB + RE + RW1", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# SCALAR FALLBACK: complex models (ZI, slopes, etc.)
# =====================================================
cat("\n--- SCALAR fallback (ZI, slopes) ---\n")

results$nb_zi <- timed("NB + ZI", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_zinegbin_negbin(),
         iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)
))

# =====================================================
# Summary
# =====================================================
cat("\n=== RESULTS SUMMARY ===\n")
cat(sprintf("%-35s %8s  %8s  %8s\n", "Config", "Time(s)", "Stan(s)", "Ratio"))
cat(sprintf("%-35s %8s  %8s  %8s\n", "------", "------", "------", "-----"))

stan_ref <- c(
  nb_base = 1.5, pg_base = 1.2, bin_base = 0.3,
  nb_re = 2.9, pg_re = 2.5, bin_re = NA,
  nb_icar = 3.2, pg_icar = 2.8,
  nb_bym2 = 5.0,
  nb_rw1 = 2.5, pg_rw1 = 2.0, nb_ar1 = 2.8,
  nb_re_rw1 = 5.0,
  nb_zi = 4.0
)

for (nm in names(results)) {
  t <- results[[nm]]
  stan <- stan_ref[nm]
  ratio <- if (!is.na(stan)) sprintf("%.2fx", t / stan) else "N/A"
  cat(sprintf("%-35s %8.2f  %8s  %8s\n", nm, t,
              if (!is.na(stan)) sprintf("%.1f", stan) else "N/A", ratio))
}

cat("\nDone.\n")
