# Narrowing down gradient bug: test configs progressively
devtools::load_all("C:/Users/Gilles Colling/Documents/dev/numdenom", quiet = TRUE)
set.seed(42)

N <- 100L; S <- 10L
site <- factor(rep(1:S, each = N/S))
x <- rnorm(N)

y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
df <- data.frame(y = y, effort = eff, x = x, spatial_site = site)

test_config <- function(label, ...) {
  cat(sprintf("\n=== %s ===\n", label))
  fit <- tryCatch({
    ratiod(y | effort ~ x + (1 | spatial_site), data = df,
           family = ratiod_poisson_gamma(),
           iter = 10, warmup = 5, chains = 1,
           gradient_mode = "H", verbose = FALSE, ...)
  }, error = function(e) paste0("ERROR: ", e$message))
  if (is.character(fit)) cat(fit, "\n") else cat("SUCCESS\n")
}

# Test 1: PG base (no RE, no spatial)
cat("\n=== PG base ===\n")
fit1 <- tryCatch({
  ratiod(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit1)) cat(fit1, "\n") else cat("SUCCESS\n")

# Test 2: PG + RE
cat("\n=== PG+RE ===\n")
fit2 <- tryCatch({
  ratiod(y | effort ~ x + (1 | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit2)) cat(fit2, "\n") else cat("SUCCESS\n")

# Test 3: PG + ICAR
W <- matrix(0, S, S)
for (i in 1:(S-1)) W[i,i+1] <- W[i+1,i] <- 1
cat("\n=== PG+ICAR ===\n")
fit3 <- tryCatch({
  ratiod(y | effort ~ x + (1 | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit3)) cat(fit3, "\n") else cat("SUCCESS\n")

# Test 4: NB base (no RE, no spatial)
y_nb <- rnbinom(N, size = 5, mu = exp(0.5 + 0.3*x))
y_d <- rnbinom(N, size = 5, mu = exp(1.0))
df_nb <- data.frame(y_num = y_nb, y_denom = y_d, x = x, spatial_site = site)
cat("\n=== NB base ===\n")
fit4 <- tryCatch({
  ratiod(y_num | y_denom ~ x, data = df_nb,
         family = ratiod_negbin_negbin(),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit4)) cat(fit4, "\n") else cat("SUCCESS\n")

# Test 5: NB + RE
cat("\n=== NB+RE ===\n")
fit5 <- tryCatch({
  ratiod(y_num | y_denom ~ x + (1 | spatial_site), data = df_nb,
         family = ratiod_negbin_negbin(),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit5)) cat(fit5, "\n") else cat("SUCCESS\n")

# Test 6: NB + ICAR
cat("\n=== NB+ICAR ===\n")
fit6 <- tryCatch({
  ratiod(y_num | y_denom ~ x + (1 | spatial_site), data = df_nb,
         family = ratiod_negbin_negbin(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit6)) cat(fit6, "\n") else cat("SUCCESS\n")

# Test 7: Bin + ICAR
y_bin <- rbinom(N, size = 20, prob = plogis(0.5 + 0.3*x))
df_bin <- data.frame(y = y_bin, trials = rep(20L, N), x = x, spatial_site = site)
cat("\n=== Bin+ICAR ===\n")
fit7 <- tryCatch({
  ratiod(y | trials ~ x + (1 | spatial_site), data = df_bin,
         family = ratiod_binomial(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = FALSE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit7)) cat(fit7, "\n") else cat("SUCCESS\n")
