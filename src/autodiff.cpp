// autodiff.cpp
// Implementation of global tape (deprecated) and backward-compatible functions

#include "autodiff.h"

namespace ratiod {
namespace ad {

// Define the global tape (deprecated - for backward compatibility only)
// New code should use TapeScope or create_tape()/delete_tape()
Tape* global_tape = nullptr;

// Backward-compatible constructor using global tape
// WARNING: Not thread-safe! Use Var(tape, value) in parallel code.
Var::Var(double value) : tape(global_tape) {
  if (tape != nullptr) {
    idx = tape->add_node(value);
  } else {
    idx = 0;
  }
}

} // namespace ad
} // namespace ratiod
