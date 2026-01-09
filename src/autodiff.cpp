// autodiff.cpp
// Implementation of global tape and R exports

#include "autodiff.h"

namespace ratiod {
namespace ad {

// Define the global tape (raw pointer, not thread_local to avoid issues)
Tape* global_tape = nullptr;

} // namespace ad
} // namespace ratiod
