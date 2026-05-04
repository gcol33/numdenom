// lik_grad_h_kernel.h
// =====================================================================
// Hand-coded H-mode gradient kernels for the 7 ratio families, ported
// from the legacy ratio H-kernel that previously lived in
// tulpa/src/hmc_gradient_analytical_lik_scalar.h.
//
// Single-file home for all per-family gradient functions plus a shared
// scaffolding helper. The scaffold computes the per-process linear
// predictors, accumulates X^T * resid into beta gradients, and adds
// the Gaussian log-prior on betas; per-family code only supplies the
// per-observation eta-residuals + the extra-parameter gradient piece.
//
// Each function matches tulpa::FullGradFn (== tulpa_hmc::GradientFn):
//   void grad_h_<family>(
//     const std::vector<double>& params,
//     const tulpa::ModelData& data,
//     const tulpa::ParamLayout& layout,
//     std::vector<double>& grad,
//     double* log_post_out
//   );
//
// Contract:
//   * `grad` is sized to params.size() and overwritten (not accumulated).
//   * `log_post_out`, if non-null, receives the fused log-posterior.
//   * The B2 spec path has NO spatial / temporal / RE / ZI / OI / SVC /
//     TVC / latent — every observation only sees process betas + extras
//     (log_phi / log_sigma / log_shape). The kernel asserts via Rcpp::stop
//     if any unsupported feature is detected so we never silently produce
//     a wrong gradient.
// =====================================================================

#ifndef TULPARATIO_LIK_GRAD_H_KERNEL_H
#define TULPARATIO_LIK_GRAD_H_KERNEL_H

#include <vector>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>

namespace tulpaRatio {

// ----- Per-family entry points ------------------------------------------
void grad_h_binomial(const std::vector<double>& params,
                     const tulpa::ModelData& data,
                     const tulpa::ParamLayout& layout,
                     std::vector<double>& grad,
                     double* log_post_out);

void grad_h_poisson_gamma(const std::vector<double>& params,
                          const tulpa::ModelData& data,
                          const tulpa::ParamLayout& layout,
                          std::vector<double>& grad,
                          double* log_post_out);

void grad_h_negbin_gamma(const std::vector<double>& params,
                         const tulpa::ModelData& data,
                         const tulpa::ParamLayout& layout,
                         std::vector<double>& grad,
                         double* log_post_out);

void grad_h_negbin_negbin(const std::vector<double>& params,
                          const tulpa::ModelData& data,
                          const tulpa::ParamLayout& layout,
                          std::vector<double>& grad,
                          double* log_post_out);

void grad_h_gamma_gamma(const std::vector<double>& params,
                        const tulpa::ModelData& data,
                        const tulpa::ParamLayout& layout,
                        std::vector<double>& grad,
                        double* log_post_out);

void grad_h_lognormal(const std::vector<double>& params,
                      const tulpa::ModelData& data,
                      const tulpa::ParamLayout& layout,
                      std::vector<double>& grad,
                      double* log_post_out);

void grad_h_beta_binomial(const std::vector<double>& params,
                          const tulpa::ModelData& data,
                          const tulpa::ParamLayout& layout,
                          std::vector<double>& grad,
                          double* log_post_out);

}  // namespace tulpaRatio

#endif  // TULPARATIO_LIK_GRAD_H_KERNEL_H
