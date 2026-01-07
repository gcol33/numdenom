test_that("spatial_car validates adjacency matrix", {
  # Non-square matrix
  bad_adj <- matrix(1, 3, 4)
  expect_error(spatial_car(bad_adj), "square")

  # Non-symmetric matrix
  bad_adj <- matrix(c(0, 1, 0, 0), 2, 2)
  expect_error(spatial_car(bad_adj), "symmetric")

  # Valid adjacency
  good_adj <- matrix(c(0, 1, 1, 0), 2, 2)
  expect_s3_class(spatial_car(good_adj, level = "obs"), "quotr_spatial")
})

test_that("spatial_car requires group_var for group level", {
  adj <- matrix(c(0, 1, 1, 0), 2, 2)

  expect_error(
    spatial_car(adj, level = "group"),
    "group_var"
  )

  expect_s3_class(
    spatial_car(adj, level = "group", group_var = "site"),
    "quotr_spatial"
  )
})

test_that("spatial_bym2 computes scale factor", {
  # Simple chain graph
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_bym2(adj, level = "obs")

  expect_s3_class(sp, "quotr_spatial")
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
