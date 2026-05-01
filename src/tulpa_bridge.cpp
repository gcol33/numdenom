#include <Rcpp.h>
#include <tulpa/nuts_api.h>

// [[Rcpp::export]]
int cpp_tulpa_abi_version() {
  tulpa::check_abi_version();
  auto fn = reinterpret_cast<tulpa::GetABIVersionFn>(
    R_GetCCallable("tulpa", "tulpa_get_abi_version")
  );
  return fn();
}
