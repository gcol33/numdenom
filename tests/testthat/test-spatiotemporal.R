# test-spatiotemporal.R
# Tests for spatiotemporal interaction specifications

test_that("spatiotemporal creates Type I specification", {
  adj <- matrix(0, 5, 5)
  for (i in 1:4) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "I"
  )

  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "I")
  expect_true(st$shared)
})


test_that("spatiotemporal creates Type IV specification", {
  adj <- matrix(0, 5, 5)
  for (i in 1:4) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "IV"
  )

  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "IV")
})


test_that("spatiotemporal 'iid' normalizes to 'I'", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    temporal = temporal_ar1("time"),
    type = "iid"
  )

  expect_equal(st$type, "I")
})


test_that("spatiotemporal validates spatial specification", {
  expect_error(
    spatiotemporal(
      spatial = list(type = "invalid"),
      temporal = temporal_rw1("year"),
      type = "I"
    ),
    "tulpaRatio spatial specification"
  )
})


test_that("spatiotemporal validates temporal specification", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1

  expect_error(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "site"),
      temporal = list(type = "invalid"),
      type = "I"
    ),
    "tulpaRatio temporal specification"
  )
})


test_that("spatiotemporal warns for non-shared effects", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1

  expect_warning(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "site"),
      temporal = temporal_rw1("year"),
      type = "I",
      shared = FALSE
    ),
    "Non-shared"
  )
})


test_that("spatiotemporal warns for proper CAR with Type IV", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  expect_warning(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "site", proper = TRUE),
      temporal = temporal_rw1("year"),
      type = "IV"
    ),
    "identifiability"
  )
})


test_that("print.ratiod_spatiotemporal works", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw2("year"),
    type = "III"
  )

  output <- capture.output(print(st))
  expect_true(any(grepl("Spatiotemporal", output)))
  expect_true(any(grepl("Type III", output)))
})


test_that("validate_spatiotemporal computes correct dimensions", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "I"
  )

  df <- expand.grid(
    region = 1:4,
    year = 2020:2024
  )
  df$x <- rnorm(nrow(df))

  st_valid <- tulpaRatio:::validate_spatiotemporal(st, df)

  expect_equal(st_valid$n_spatial, 4)
  expect_equal(st_valid$n_times, 5)
  expect_equal(st_valid$n_params, 20)  # 4 * 5
})


test_that("build_st_index creates correct mappings", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    temporal = temporal_rw1("time"),
    type = "I"
  )

  df <- data.frame(
    site = c(1, 1, 2, 2, 3, 3),
    time = c(1, 2, 1, 2, 1, 2),
    x = rnorm(6)
  )

  st_valid <- tulpaRatio:::validate_spatiotemporal(st, df)
  idx <- st_valid$st_index

  expect_equal(idx$S, 3)
  expect_equal(idx$T, 2)
  expect_equal(idx$N, 6)
  expect_equal(length(idx$s_idx), 6)
  expect_equal(length(idx$t_idx), 6)
})


test_that("prepare_spatiotemporal_for_hmc returns correct structure", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    temporal = temporal_rw1("time"),
    type = "II"
  )

  df <- expand.grid(site = 1:3, time = 1:4)
  st_valid <- tulpaRatio:::validate_spatiotemporal(st, df)

  hmc_data <- tulpaRatio:::prepare_spatiotemporal_for_hmc(st_valid, df)

  expect_true(hmc_data$has_spatiotemporal)
  expect_equal(hmc_data$type, "II")
  expect_true(hmc_data$shared)
  expect_equal(hmc_data$n_spatial, 3)
  expect_equal(hmc_data$n_times, 4)
})


test_that("prepare_spatiotemporal_for_hmc handles NULL", {
  hmc_data <- tulpaRatio:::prepare_spatiotemporal_for_hmc(NULL, data.frame())

  expect_false(hmc_data$has_spatiotemporal)
  expect_equal(hmc_data$type, "none")
})


test_that("spatiotemporal_gp creates non-separable specification", {
  st_gp <- spatiotemporal_gp(
    ~ lon + lat,
    time_var = "year",
    nonsep_type = "gneiting"
  )

  expect_s3_class(st_gp, "ratiod_st_gp")
  expect_s3_class(st_gp, "ratiod_spatiotemporal")
  expect_equal(st_gp$type, "st_gp")
  expect_equal(st_gp$nonsep_type, "gneiting")
  expect_equal(st_gp$coord_vars, c("lon", "lat"))
  expect_equal(st_gp$time_var, "year")
})


test_that("spatiotemporal_gp validates coordinates", {
  expect_error(
    spatiotemporal_gp(
      ~ lon,  # Only one coordinate
      time_var = "year"
    ),
    "2 coordinate"
  )

  expect_error(
    spatiotemporal_gp(
      "not_a_formula",  # Invalid type
      time_var = "year"
    ),
    "formula"
  )
})


test_that("spatiotemporal_gp validates time_var", {
  expect_error(
    spatiotemporal_gp(
      ~ lon + lat,
      time_var = c("year", "month")  # Multiple time vars
    ),
    "single character"
  )
})


test_that("spatiotemporal_gp validates nn parameter", {
  expect_error(
    spatiotemporal_gp(
      ~ lon + lat,
      time_var = "year",
      nn = 0
    ),
    "positive integer"
  )
})


test_that("spatiotemporal_gp warns for non-shared effects", {
  expect_warning(
    spatiotemporal_gp(
      ~ lon + lat,
      time_var = "year",
      shared = FALSE
    ),
    "Non-shared"
  )
})


test_that("print.ratiod_st_gp works", {
  st_gp <- spatiotemporal_gp(
    ~ x + y,
    time_var = "t",
    cov_space = "matern",
    nonsep_type = "sum"
  )

  output <- capture.output(print(st_gp))
  expect_true(any(grepl("Non-Separable", output)))
  expect_true(any(grepl("matern", output)))
  expect_true(any(grepl("Sum", output)))
})


test_that("validate_st_gp computes dimensions correctly", {
  st_gp <- spatiotemporal_gp(
    ~ lon + lat,
    time_var = "time",
    nn = 5
  )

  set.seed(123)
  df <- data.frame(
    lon = runif(20, 0, 10),
    lat = runif(20, 0, 10),
    time = rep(1:4, each = 5)
  )

  st_valid <- tulpaRatio:::validate_st_gp(st_gp, df)

  expect_equal(st_valid$n_obs, 20)
  expect_equal(st_valid$n_times, 4)
  expect_equal(st_valid$n_params, 20)
  expect_equal(nrow(st_valid$coords_matrix), 20)
  expect_equal(length(st_valid$time_values), 20)
})


test_that("validate_st_gp checks for missing coordinates", {
  st_gp <- spatiotemporal_gp(~ lon + lat, time_var = "time")

  df <- data.frame(
    lon = c(1, NA, 3),
    lat = c(1, 2, 3),
    time = 1:3
  )

  expect_error(
    tulpaRatio:::validate_st_gp(st_gp, df),
    "missing values"
  )
})


test_that("validate_st_gp checks for missing time values", {
  st_gp <- spatiotemporal_gp(~ lon + lat, time_var = "time")

  df <- data.frame(
    lon = 1:3,
    lat = 1:3,
    time = c(1, NA, 3)
  )

  expect_error(
    tulpaRatio:::validate_st_gp(st_gp, df),
    "missing values"
  )
})


test_that("validate_st_gp handles Date time variable", {
  st_gp <- spatiotemporal_gp(~ lon + lat, time_var = "date")

  df <- data.frame(
    lon = runif(10),
    lat = runif(10),
    date = as.Date("2020-01-01") + 0:9
  )

  st_valid <- tulpaRatio:::validate_st_gp(st_gp, df)
  expect_true(is.numeric(st_valid$time_values))
})


test_that("compute_st_nngp_neighbors creates valid structure", {
  set.seed(42)
  coords <- cbind(runif(15), runif(15))
  time <- rep(1:3, each = 5)

  result <- tulpaRatio:::compute_st_nngp_neighbors(coords, time, k = 3)

  expect_equal(nrow(result$nn_idx), 15)
  expect_equal(ncol(result$nn_idx), 3)
  expect_equal(length(result$nn_order), 15)
  expect_equal(length(result$nn_order_inv), 15)

  # Check ordering is valid permutation
  expect_equal(sort(result$nn_order), 1:15)
  expect_equal(result$nn_order_inv[result$nn_order], 1:15)
})


test_that("all interaction types are supported", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i+1] <- adj[i+1, i] <- 1

  spatial <- spatial_car(adj, level = "group", group_var = "s")
  temporal <- temporal_rw1("t")

  for (type in c("I", "II", "III", "IV")) {
    st <- spatiotemporal(
      spatial = spatial,
      temporal = temporal,
      type = type
    )
    expect_equal(st$type, type)
  }
})


test_that("separable type works with GP spatial", {
  st <- expect_warning(
    spatiotemporal(
      spatial = spatial_gp(~ lon + lat),
      temporal = temporal_rw1("year"),
      type = "separable"
    ),
    NA  # No warning expected
  )

  expect_equal(st$type, "separable")
})


test_that("separable type warns without GP", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1

  expect_warning(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "s"),
      temporal = temporal_rw1("t"),
      type = "separable"
    ),
    "GP spatial"
  )
})
