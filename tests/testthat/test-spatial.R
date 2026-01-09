test_that("spatial_car validates adjacency matrix", {
  # Non-square matrix
  bad_adj <- matrix(1, 3, 4)
  expect_error(spatial_car(bad_adj), "square")

  # Non-symmetric matrix
  bad_adj <- matrix(c(0, 1, 0, 0), 2, 2)
  expect_error(spatial_car(bad_adj), "symmetric")

  # Valid adjacency
  good_adj <- matrix(c(0, 1, 1, 0), 2, 2)
  expect_s3_class(spatial_car(good_adj, level = "obs"), "ratiod_spatial")
})

test_that("spatial_car requires group_var for group level", {
  adj <- matrix(c(0, 1, 1, 0), 2, 2)

  expect_error(
    spatial_car(adj, level = "group"),
    "group_var"
  )

  expect_s3_class(
    spatial_car(adj, level = "group", group_var = "site"),
    "ratiod_spatial"
  )
})

test_that("spatial_bym2 computes scale factor", {
  # Simple chain graph
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_bym2(adj, level = "obs")

  expect_s3_class(sp, "ratiod_spatial")
  expect_equal(sp$type, "bym2")
  expect_true(sp$scale_factor > 0)
})

test_that("is_connected detects disconnected graphs", {
  # Connected graph
  adj_conn <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3)
  expect_true(is_connected(adj_conn))

  # Disconnected graph (two isolated nodes)
  adj_disc <- matrix(c(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0), 4, 4)
  expect_false(is_connected(adj_disc))
})


# =============================================================================
# Tests for proper CAR (spatial_car with proper = TRUE)
# =============================================================================

test_that("spatial_car creates ICAR by default", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, level = "obs")

  expect_s3_class(sp, "ratiod_spatial")
  expect_equal(sp$type, "car")
  expect_false(sp$proper)
  expect_null(sp$rho_bounds)
})


test_that("spatial_car with proper = TRUE creates proper CAR", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, level = "obs", proper = TRUE)

  expect_s3_class(sp, "ratiod_spatial")
  expect_equal(sp$type, "car_proper")
  expect_true(sp$proper)
  expect_true(!is.null(sp$rho_bounds))
  expect_true(sp$rho_bounds["lower"] >= 0)
  expect_true(sp$rho_bounds["upper"] <= 1)
  expect_true(sp$rho_bounds["lower"] < sp$rho_bounds["upper"])
})


test_that("proper CAR print method shows rho info", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, level = "obs", proper = TRUE)
  output <- capture.output(print(sp))

  expect_true(any(grepl("Proper CAR", output)))
  expect_true(any(grepl("Rho bounds", output)))
  expect_true(any(grepl("estimated", output, ignore.case = TRUE)))
})


test_that("ICAR print method shows fixed rho info", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, level = "obs", proper = FALSE)
  output <- capture.output(print(sp))

  expect_true(any(grepl("ICAR", output)))
  expect_true(any(grepl("fixed at 1", output)))
})


test_that("compute_car_rho_bounds returns valid range", {
  # Simple chain graph
  adj <- matrix(0, 5, 5)
  for (i in 1:4) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }

  bounds <- compute_car_rho_bounds(adj)

  expect_true(is.numeric(bounds))
  expect_equal(length(bounds), 2)
  expect_true("lower" %in% names(bounds))
  expect_true("upper" %in% names(bounds))
  expect_true(bounds["lower"] < bounds["upper"])
  expect_true(bounds["lower"] >= 0)
  expect_true(bounds["upper"] <= 1)
})


test_that("compute_car_rho_bounds handles isolated nodes", {
  # Graph with one isolated node
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  # Node 4 has no neighbors (isolated)

  expect_warning(
    bounds <- compute_car_rho_bounds(adj),
    regexp = "isolated"
  )

  # Should still return valid bounds
  expect_true(bounds["lower"] < bounds["upper"])
})


test_that("proper CAR works with group level", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  sp <- spatial_car(adj, level = "group", group_var = "region", proper = TRUE)

  expect_equal(sp$type, "car_proper")
  expect_equal(sp$group_var, "region")
  expect_equal(sp$level, "group")
  expect_true(sp$proper)
})


test_that("proper CAR validates adjacency like ICAR", {
  # Non-square matrix
  bad_adj <- matrix(1, 3, 4)
  expect_error(spatial_car(bad_adj, proper = TRUE), "square")

  # Non-symmetric matrix
  bad_adj <- matrix(c(0, 1, 0, 0), 2, 2)
  expect_error(spatial_car(bad_adj, proper = TRUE), "symmetric")
})


test_that("spatial_car shared argument works for both variants", {
  adj <- matrix(c(0, 1, 1, 0), 2, 2)

  # ICAR
  sp_icar <- spatial_car(adj, level = "obs", shared = FALSE)
  expect_false(sp_icar$shared)

  # Proper CAR
  sp_proper <- spatial_car(adj, level = "obs", proper = TRUE, shared = FALSE)
  expect_false(sp_proper$shared)
})
