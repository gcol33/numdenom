suppressMessages(devtools::load_all("C:/GillesC/Documents/dev/tulpaRatio", quiet = TRUE))
set.seed(20260503)

n <- 200
x1 <- rnorm(n); x2 <- rnorm(n)
mu_num   <- exp(1.0 + 0.4 * x1)
mu_denom <- exp(0.7 + 0.3 * x2)
phi_num  <- 5.0; phi_denom <- 5.0
y_num   <- rnbinom(n, size = phi_num,   mu = mu_num)
y_denom <- rnbinom(n, size = phi_denom, mu = mu_denom)
dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
fam <- tulpaRatio::ratiod_negbin_negbin()
form <- y_num | y_denom ~ x1 + x2

fit_one <- function(use_specs, seed_val, gradient_mode = "H") {
  op <- options(tulpaRatio.use_specs = use_specs)
  on.exit(options(op), add = TRUE)
  tulpaRatio::ratiod(
    formula = form, data = dat, family = fam,
    mode = "hmc", iter = 2000L, warmup = 500L, chains = 1L,
    seed = seed_val, verbose = FALSE, gradient_mode = gradient_mode
  )
}

# Compare colMeans across both backends, both modes
fits <- list(
  legacy_H_42  = fit_one(FALSE, 42L, "H"),
  specs_H_42   = fit_one(TRUE,  42L, "H"),
  specs_Ar_42  = fit_one(TRUE,  42L, "A_r"),
  legacy_H_43  = fit_one(FALSE, 43L, "H"),
  specs_H_43   = fit_one(TRUE,  43L, "H")
)
means <- sapply(fits, function(f) colMeans(f$draws))
print(round(means, 4))
cat("\n--- legacy_H_42 vs specs_H_42 ---\n")
print(round(means[, "legacy_H_42"] - means[, "specs_H_42"], 4))
cat("\n--- specs_H_42 vs specs_Ar_42 ---\n")
print(round(means[, "specs_H_42"] - means[, "specs_Ar_42"], 4))
cat("\n--- specs_H_42 vs specs_H_43 (within-MC for specs/H) ---\n")
print(round(means[, "specs_H_42"] - means[, "specs_H_43"], 4))
