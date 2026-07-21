// Measurement of how many chains the chain-parallel helper actually runs at once.
//
// `cores` is only meaningful if it reaches the OpenMP region, and a team sized
// from a value another backend can move reaches it wrong: chains serialize, the
// fit still returns the same draws, and only the wall clock says so.
#include "omp_chain_team.h"
#include <Rcpp.h>
#include <atomic>
#include <chrono>
#include <thread>

// Runs the helper over n_chains with the given cap and returns the largest
// number of chain bodies that were ever in flight together. Each body sleeps so
// that overlap is observable rather than a race against how fast it returns.
// [[Rcpp::export]]
int cpp_chain_concurrency(int n_chains, int max_concurrent) {
  std::atomic<int> live(0);
  std::atomic<int> peak(0);
  ratiod_omp::for_each_chain(n_chains, max_concurrent, [&](int) {
    int now = ++live;
    int seen = peak.load();
    while (now > seen && !peak.compare_exchange_weak(seen, now)) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    --live;
  });
  return peak.load();
}

// Number of chain bodies the helper ran, which must equal n_chains whatever the
// cap does to their timing.
// [[Rcpp::export]]
int cpp_chain_visits(int n_chains, int max_concurrent) {
  std::atomic<int> visits(0);
  ratiod_omp::for_each_chain(n_chains, max_concurrent, [&](int) { ++visits; });
  return visits.load();
}

// Moves the nthreads-var, so that the chain team can be shown to be immune to
// it without depending on a backend to leave it moved.
// [[Rcpp::export]]
void cpp_set_max_threads(int n_threads) {
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#else
  (void)n_threads;
#endif
}

// [[Rcpp::export]]
int cpp_num_procs() {
#ifdef _OPENMP
  return omp_get_num_procs();
#else
  return 1;
#endif
}
