# Test LOGNORMAL with separate formulas

library(numdenom)

set.seed(123)

cat("=== Testing LOGNORMAL with covariate ===\n")

N <- 200
x <- rnorm(N)

# Generate data:
# y_num ~ lognormal(2 + 0.3*x, 0.5)
# y_denom ~ lognormal(3 + 0.2*x, 0.4)
y_num <- rlnorm(N, meanlog = 2 + 0.3 * x, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3 + 0.2 * x, sdlog = 0.4)

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("\nSimple regression check (log scale):\n")
cat("  lm(log(y_num) ~ x):\n")
lm_num <- lm(log(y_num) ~ x)
cat("    intercept:", coef(lm_num)[1], " (true: 2)\n")
cat("    slope:", coef(lm_num)[2], " (true: 0.3)\n")
cat("  lm(log(y_denom) ~ x):\n")
lm_denom <- lm(log(y_denom) ~ x)
cat("    intercept:", coef(lm_denom)[1], " (true: 3)\n")
cat("    slope:", coef(lm_denom)[2], " (true: 0.2)\n")

cat("\n=== numdenom with separate formulas ===\n")
# Use separate formula_num and formula_denom to get separate slopes
fit <- tratio(
  y | denom ~ 1,  # intercept only in main formula (shared)
  formula_num = ~ x,   # x only affects numerator
  formula_denom = ~ x, # x only affects denominator
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE)
)

draws <- as.matrix(fit$draws)
cat("Parameter names:", colnames(draws), "\n")
cat("\nParameter estimates:\n")
for (p in colnames(draws)) {
  cat("  ", p, ":", mean(draws[,p]), "+/-", sd(draws[,p]), "\n")
}

cat("\n=== numdenom with shared formula (y | denom ~ x) ===\n")
fit2 <- tratio(
  y | denom ~ x,  # x affects both num and denom
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE)
)

draws2 <- as.matrix(fit2$draws)
cat("Parameter names:", colnames(draws2), "\n")
cat("\nParameter estimates:\n")
for (p in colnames(draws2)) {
  cat("  ", p, ":", mean(draws2[,p]), "+/-", sd(draws2[,p]), "\n")
}

cat("\n=== DONE ===\n")
