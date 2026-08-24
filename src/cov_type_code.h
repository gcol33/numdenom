#ifndef RATIOD_COV_TYPE_CODE_H
#define RATIOD_COV_TYPE_CODE_H

#include <Rcpp.h>

#include "hmc_cov.h"

namespace ratiod_cov {

// The integer R sends for a cov = choice, in the order R/spatial.R's
// cov_type_code() maps the four documented names. One mapping, on both sides:
// a code this does not know is refused rather than resolved to whichever kernel
// a fallthrough happens to reach.
inline CovType cov_type_from_int(int cov_type) {
  switch (cov_type) {
    case 0: return CovType::EXPONENTIAL;
    case 1: return CovType::MATERN;
    case 2: return CovType::GAUSSIAN;
    case 3: return CovType::SPHERICAL;
  }
  Rcpp::stop("Unknown covariance code %d. Expected 0 (exponential), "
             "1 (matern), 2 (gaussian) or 3 (spherical).", cov_type);
}

}  // namespace ratiod_cov

#endif  // RATIOD_COV_TYPE_CODE_H
