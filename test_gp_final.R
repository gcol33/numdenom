# GP test with numerical gradients
library(numdenom)

set.seed(123)  # Set once at the start

cat("Testing GP models with numerical gradients...\n")

for (n in c(10, 20, 30, 50, 100)) {
  cat(sprintf("Testing n=%d... ", n))

  data <- data.frame(
    y_num = rpois(n, exp(2)),
    y_denom = rgamma(n, 5, 1),
    x = rnorm(n),
    lon = runif(n, 0, 10),
    lat = runif(n, 0, 10)
  )

  result <- tryCatch({
    fit <- ratiod(y_num | y_denom ~ x,
                  data = data,
                  family = ratiod_poisson_gamma(),
                  spatial = spatial_gp(~ lon + lat, nn = 5),
                  iter = 10, warmup = 5, chains = 1,
                  refresh = 0, cores = 1)
    TRUE
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    FALSE
  })

  if (result) cat("OK\n")
  if (!result) break
}

cat("\nDone.\n")
