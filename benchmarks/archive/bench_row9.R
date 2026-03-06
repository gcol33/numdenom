# Row 9: poisson_gamma + RW1
library(numdenom)
set.seed(123)

N <- 500
x <- rnorm(N)
site <- factor(rep(1:50, length.out = N))
time <- factor(rep(1:20, length.out = N))
y <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df <- data.frame(y=y, effort=effort, x=x, site=site, time=time)

cat("Row 9: poisson_gamma + RW1\n")
time_H <- system.time({
  fit <- ratiod(y | effort ~ x + (1 | site), data=df,
                family=ratiod_poisson_gamma(),
                temporal=temporal_rw1("time"),
                iter=500, warmup=250, chains=1, verbose=FALSE,
                gradient_mode="H")
})["elapsed"]
cat(sprintf("  H: %.1fs\n", time_H))
