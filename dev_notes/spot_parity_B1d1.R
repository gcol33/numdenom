suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
})

set.seed(42)
N <- 200L
n_groups <- 8L
group <- factor(rep(seq_len(n_groups), length.out = N))
x_fix <- rnorm(N)
x_re  <- rnorm(N)
trials <- rpois(N, 8) + 5L
eta <- 0.3 + 0.4 * x_fix +
       rnorm(n_groups, sd = 0.5)[as.integer(group)] +
       rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re
y <- rbinom(N, trials, plogis(eta))
df <- data.frame(successes = y, trials = trials,
                 x_fix = x_fix, x_re = x_re, group = group)

iter <- 600L; warmup <- 400L

run <- function(use_specs, seed) {
  options(tulpaRatio.use_specs = use_specs)
  on.exit(options(tulpaRatio.use_specs = FALSE), add = TRUE)
  ratiod(successes | trials ~ x_fix + (1 + x_re || group),
         data = df, family = ratiod_binomial(),
         mode = "hmc", iter = iter, warmup = warmup,
         chains = 1L, seed = seed, verbose = FALSE)
}

post_means <- function(fit) {
  d <- fit$draws
  if (is.list(d) && !is.matrix(d)) d <- d[[1]]
  colMeans(d)
}

f_l_42 <- run(FALSE, 42L); f_l_43 <- run(FALSE, 43L)
f_s_42 <- run(TRUE, 42L)

m_l_42 <- post_means(f_l_42)
m_l_43 <- post_means(f_l_43)
m_s_42 <- post_means(f_s_42)

cat("legacy 42 names:\n"); print(names(m_l_42))
cat("specs  42 names:\n"); print(names(m_s_42))

common <- intersect(names(m_l_42), names(m_s_42))
cross <- max(abs(m_l_42[common] - m_s_42[common]))
within <- max(abs(m_l_42[common] - m_l_43[common]))
cat(sprintf("\ncross=%.4f  within=%.4f  ratio=%.2f\n",
            cross, within, cross / max(within, 1e-8)))

cat("\nLegacy 42 values:\n");  print(round(m_l_42[common], 3))
cat("\nSpecs  42 values:\n");  print(round(m_s_42[common], 3))
cat("\nLegacy 43 values:\n");  print(round(m_l_43[common], 3))
