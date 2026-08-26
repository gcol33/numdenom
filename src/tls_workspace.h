#ifndef TULPARATIO_TLS_WORKSPACE_H
#define TULPARATIO_TLS_WORKSPACE_H

// Per-thread scratch buffers that outlive the thread that allocated them.
//
// The natural spelling, `static thread_local Workspace ws;` over a struct owning
// heap buffers, is the one shape that corrupts the heap under the Rtools mingw
// toolchain. GCC gives such an object emutls-backed storage and a destructor
// registered through __cxa_thread_atexit; both are released by thread-exit hooks
// whose relative order is unspecified, so the destructor can free through
// storage that is already gone, and the process dies at some later unrelated
// free (STATUS_HEAP_CORRUPTION, or STATUS_STACK_BUFFER_OVERRUN from the
// fail-fast).
//
// A worker thread only exits when libgomp narrows a team, so this is reachable
// exactly when an OpenMP region runs narrower than one that ran before it.
// Measured against such a schedule, holding the workspace as an object dies in
// 5 runs of 5, while a function-local workspace, a trivially destructible
// thread_local, and the pointer form below each survive 5 of 5; the object form
// survives 5 of 5 against a team that never narrows. test_tls_workspace.cpp
// runs the surviving shape against the same schedule.
//
// The pointer is constant-initialized, so it carries neither an initialization
// guard nor a destructor registration: the compiled object holds no
// __emutls_v._ZGVZ guard symbol for it and makes no __cxa_thread_atexit call.
// Nothing frees the pointee. The leak is one workspace per thread that ever
// enters the region, bounded by the team width for as long as the team is not
// churned.

// Declares `name` as a reference to this thread's workspace. Statement scope.
#define RATIOD_TLS_WORKSPACE(type, name)                     \
  static thread_local type* name##_tls_slot_ = nullptr;      \
  if (name##_tls_slot_ == nullptr) name##_tls_slot_ = new type(); \
  type& name = *name##_tls_slot_

// Namespace-scope counterpart: defines `fn()` returning this thread's workspace,
// for scratch shared by several functions. External linkage, not internal: a
// function template reaches these, and a template instantiated in two
// translation units must find the same workspace in both rather than whichever
// internal-linkage copy the linker happened to keep.
#define RATIOD_TLS_WORKSPACE_FN(type, fn)  \
  inline type& fn() {                      \
    RATIOD_TLS_WORKSPACE(type, ws);        \
    return ws;                             \
  }

#endif  // TULPARATIO_TLS_WORKSPACE_H
