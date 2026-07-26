# Per-thread scratch buffers, against the OpenMP schedule that destroys workers.
#
# Narrowing a team makes libgomp destroy the surplus workers. A thread_local
# object holding heap buffers frees through emutls storage that a separate
# thread-exit hook may already have released, and the process dies at a later
# unrelated free. src/tls_workspace.h is the shape that does not, and every
# workspace in the samplers is held that way.

test_that("a macro-held workspace survives a team that keeps narrowing", {
  skip_on_cran()
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores to narrow a team")

  # The same schedule kills a thread_local object workspace in 5 runs of 5.
  # Reaching the expectation at all is most of the assertion: a regression in
  # tls_workspace.h takes the process down here rather than failing.
  value <- tulpaRatio:::cpp_tls_workspace_shrink(40L, 64L, 1024L)

  expect_true(is.finite(value))
  expect_gt(value, 0)
})

test_that("no workspace is declared in the shape that corrupts the heap", {
  src <- test_path("..", "..", "src")
  skip_if_not(dir.exists(src), "source not available")

  files <- list.files(src, pattern = "\\.(cpp|h)$", full.names = TRUE)
  # The macro's own declaration names its slot by token pasting, which the
  # patterns below cannot read; the runtime test above is what covers it.
  files <- files[basename(files) != "tls_workspace.h"]

  # A declaration is safe when nothing registers a thread-exit destructor for
  # it: a constant-initialized pointer, or a scalar with no destructor at all.
  safe <- paste0(
    "\\*\\s*\\w+\\s*=\\s*nullptr\\s*;",
    "|static thread_local\\s+(unsigned\\s+|signed\\s+|std::)?",
    "(int|long|short|char|bool|double|float|size_t)\\b[^*;]*;"
  )

  offenders <- as.character(unlist(lapply(files, function(f) {
    lines <- readLines(f, warn = FALSE)
    code <- sub("//.*$", "", lines)
    hit <- grep("static thread_local", code)
    hit <- hit[!grepl(safe, code[hit])]
    if (length(hit)) sprintf("%s:%d: %s", basename(f), hit, trimws(code[hit]))
  })))

  expect_identical(
    offenders, character(0),
    info = paste("declare these with RATIOD_TLS_WORKSPACE (src/tls_workspace.h)",
                 paste(offenders, collapse = "\n"), sep = "\n")
  )
})
