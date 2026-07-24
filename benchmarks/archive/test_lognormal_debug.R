# Debug LOGNORMAL gradient computation

library(numdenom)

set.seed(42)
N <- 50
x <- rnorm(N)

# Very simple: no slope, just intercept
y_num <- rlnorm(N, meanlog = 2, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3, sdlog = 0.4)
df <- data.frame(y = y_num, denom = y_denom)

cat("=== Intercept-only LOGNORMAL ===\n")
cat("\nSample statistics:\n")
cat("  mean(log(y_num)):", mean(log(y_num)), "\n")
cat("  mean(log(y_denom)):", mean(log(y_denom)), "\n")

cat("\n=== Short runs to compare gradients ===\n")

cat("\nA_t mode (10 iter):\n")
fit_At <- tratio(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 10, warmup = 1, chains = 1, gradient_mode = "A_t", verbose = TRUE)
)

cat("\nH mode (10 iter):\n")
fit_H <- tratio(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 10, warmup = 1, chains = 1, gradient_mode = "H", verbose = TRUE)
)

# Compare first draws
draws_At <- as.matrix(fit_At$draws)
draws_H <- as.matrix(fit_H$draws)

cat("\n=== Draw comparison (first 5) ===\n")
cat("A_t beta_num[1]:", head(draws_At[,"beta_num[1]"], 5), "\n")
cat("H beta_num[1]:", head(draws_H[,"beta_num[1]"], 5), "\n")

cat("\nA_t sigma_num:", head(draws_At[,"sigma_num"], 5), "\n")
cat("H sigma_num:", head(draws_H[,"sigma_num"], 5), "\n")

cat("\n=== DONE ===\n")
