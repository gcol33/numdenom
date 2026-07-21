#ifndef TULPARATIO_OMP_THREAD_SCOPE_H
#define TULPARATIO_OMP_THREAD_SCOPE_H

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_omp {

// Sets the OpenMP thread count for the enclosing scope and puts back what it
// was on the way out.
//
// The nthreads-var is process-wide and outlives whatever moved it, so a backend
// that sets it and walks away is choosing the width of every region that runs
// after it, in fits it knows nothing about. The backends here all take a
// per-fit thread count, which makes that a fit's private business; leaving it
// moved published it. Reading it to size the chain team was enough to serialize
// the chains of every fit following a single-threaded Laplace fit (#8).
//
// The regions inside a fit still read the nthreads-var, which is what they
// should do: within the scope it holds this fit's budget.
//
// A count of zero or less means the caller expressed no preference and the
// current value stands.
class ScopedThreadCount {
 public:
  explicit ScopedThreadCount(int n_threads) {
#ifdef _OPENMP
    if (n_threads > 0) {
      previous_ = omp_get_max_threads();
      restore_ = true;
      omp_set_num_threads(n_threads);
    }
#else
    (void)n_threads;
#endif
  }

  ~ScopedThreadCount() {
#ifdef _OPENMP
    if (restore_) omp_set_num_threads(previous_);
#endif
  }

  ScopedThreadCount(const ScopedThreadCount&) = delete;
  ScopedThreadCount& operator=(const ScopedThreadCount&) = delete;
  ScopedThreadCount(ScopedThreadCount&&) = delete;
  ScopedThreadCount& operator=(ScopedThreadCount&&) = delete;

 private:
#ifdef _OPENMP
  int previous_ = 1;
  bool restore_ = false;
#endif
};

}  // namespace ratiod_omp

#endif  // TULPARATIO_OMP_THREAD_SCOPE_H
