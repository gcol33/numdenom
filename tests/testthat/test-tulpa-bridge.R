test_that("tulpa engine ABI is available", {
  expect_equal(tulpaRatio:::cpp_tulpa_abi_version(), 2L)
})
