# Row 12: poisson_gamma + ZI
library(numdenom)
set.seed(123)

N <- 500
x <- rnorm(N)
site <- factor(rep(1:50, length.out = N))
y <- rpois(N, exp(2 + 0.5*x))
y[sample(N, 100)] <- 0  # Add zeros
effort <- rgamma(N, 10, 1)
df <- data.frame(y=y, effort=effort, x=x, site=site)

cat("Row 12: poisson_gamma + ZI\n")
time_H <- system.time({
  fit <- tratio(y | effort ~ x + (1 | site), data=df,
                family=ratiod_zipois(),
                control = list(iter=500, warmup=250, chains=1, verbose=FALSE, gradient_mode="H"))
})["elapsed"]
cat(sprintf("  H: %.1fs\n", time_H))
