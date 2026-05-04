# dev_notes/parity_B2_diag.R
# Diagnostic: compare specs path at H-mode vs at A_r-mode for the same data.
# This isolates the gradient cost from the engine-frontend cost.
suppressMessages(devtools::load_all("C:/GillesC/Documents/dev/tulpaRatio", quiet = TRUE))
set.seed(20260503)

fit_one <- function(formula, data, family, use_specs, seed_val,
                    gradient_mode = "H") {
  op <- options(tulpaRatio.use_specs = use_specs)
  on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::ratiod(
    formula = formula, data = data, family = family,
    mode = "hmc", iter = 2000L, warmup = 500L, chains = 1L,
    seed = seed_val, verbose = FALSE, gradient_mode = gradient_mode
  )
  draws <- fit$draws
  if (is.matrix(draws)) colMeans(draws) else stop("unexpected draws shape")
}

# Set up Negbin-Negbin (the meaningful 2-process case)
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

bench <- function(label, ...) {
  t1 <- Sys.time()
  res <- fit_one(form, dat, fam, ...)
  t2 <- Sys.time()
  cat(sprintf("%-25s %.2fs\n", label, as.numeric(difftime(t2, t1, units="secs"))))
  res
}

cat("\n=== negbin_negbin engine vs gradient cost ===\n")
m_legacy_h   <- bench("legacy / H",     FALSE, 42L, "H")
m_legacy_ar  <- bench("legacy / A_r",   FALSE, 42L, "A_r")
m_specs_h    <- bench("specs / H",      TRUE,  42L, "H")
m_specs_ar   <- bench("specs / A_r",    TRUE,  42L, "A_r")

cat("\nMaximal cross-config diff between specs/H and legacy/H:\n")
common <- intersect(names(m_legacy_h), names(m_specs_h))
cat(sprintf(" max abs diff = %.4g\n",
            max(abs(m_legacy_h[common] - m_specs_h[common]))))
