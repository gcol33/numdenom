# dev_notes/build_check_b2.R
# Compile + load tulpaRatio after the B2 H-kernel additions.
suppressMessages({
  Sys.setenv(R_KEEP_PKG_SOURCE = "yes")
  devtools::load_all("C:/GillesC/Documents/dev/tulpaRatio", quiet = FALSE)
})
cat("\n[OK] tulpaRatio loaded after B2 H-kernel port.\n")
