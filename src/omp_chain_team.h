#ifndef TULPARATIO_OMP_CHAIN_TEAM_H
#define TULPARATIO_OMP_CHAIN_TEAM_H

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_omp {

// The team used for chain-parallel work is fixed for the life of the session.
//
// Sizing it per fit lets it shrink between fits in one session, and a shrink
// makes libgomp destroy the surplus workers and run their thread_local
// destructors while later work is still in flight. On Windows that ends in
// heap corruption (STATUS_HEAP_CORRUPTION at a free() on a worker thread).
// A team that only ever grows is safe; only the shrink faults.
//
// The size is read once and cached. Reading it on every call would track the
// nthreads-var that laplace_core and the Polya-Gamma samplers set through
// omp_set_num_threads(), which is exactly the moving value to avoid.
inline int chain_team_size() {
#ifdef _OPENMP
  static const int team = std::max(1, omp_get_max_threads());
  return team;
#else
  return 1;
#endif
}

// Runs fn(c) for chains 0..n_chains-1, at most max_concurrent at a time.
//
// Concurrency is bounded by how many chains are offered to each region, not by
// resizing the team: the team stays at chain_team_size() and threads with no
// iteration simply sit out. That keeps `cores` meaning what it says while the
// team size stays constant across fits.
template <typename Fn>
inline void for_each_chain(int n_chains, int max_concurrent, Fn fn) {
#ifdef _OPENMP
  const int width = std::max(1, std::min(max_concurrent, n_chains));
  const int team = chain_team_size();
  for (int base = 0; base < n_chains; base += width) {
    const int stop = std::min(base + width, n_chains);
    #pragma omp parallel for schedule(static) num_threads(team)
    for (int c = base; c < stop; c++) {
      fn(c);
    }
  }
#else
  (void)max_concurrent;
  for (int c = 0; c < n_chains; c++) {
    fn(c);
  }
#endif
}

}  // namespace ratiod_omp

#endif  // TULPARATIO_OMP_CHAIN_TEAM_H
