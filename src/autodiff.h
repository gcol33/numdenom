// autodiff.h
// Reverse-mode automatic differentiation for ratiod
// Lightweight implementation inspired by modern AD libraries

#ifndef QUOTR_AUTODIFF_H
#define QUOTR_AUTODIFF_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>

namespace ratiod {
namespace ad {

// Forward declarations
class Tape;
class Var;

// Global tape for recording operations
// Note: Using raw pointer instead of shared_ptr to avoid thread_local issues
extern Tape* global_tape;

// ---------------------------------------------------------------------
// Tape: Records computation graph for reverse-mode AD
// ---------------------------------------------------------------------
class Tape {
public:
  struct Node {
    double value;
    double adjoint;
    std::function<void()> backward;

    Node(double v = 0.0) : value(v), adjoint(0.0), backward([](){}) {}
  };

  std::vector<Node> nodes;

  size_t add_node(double value) {
    nodes.emplace_back(value);
    return nodes.size() - 1;
  }

  void clear() {
    nodes.clear();
  }

  void zero_adjoints() {
    for (auto& node : nodes) {
      node.adjoint = 0.0;
    }
  }

  void backward(size_t root_idx) {
    if (root_idx >= nodes.size()) return;

    nodes[root_idx].adjoint = 1.0;

    // Reverse pass
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
      nodes[i].backward();
    }
  }
};

// Initialize global tape
inline void init_tape() {
  if (global_tape != nullptr) {
    delete global_tape;
  }
  global_tape = new Tape();
}

inline void clear_tape() {
  if (global_tape != nullptr) {
    delete global_tape;
    global_tape = nullptr;
  }
}

// ---------------------------------------------------------------------
// Var: Automatic differentiation variable
// ---------------------------------------------------------------------
class Var {
public:
  size_t idx;

  // Construct from value
  Var(double value = 0.0) {
    if (global_tape != nullptr) {
      idx = global_tape->add_node(value);
    } else {
      idx = 0;
    }
  }

  // Get value
  double val() const {
    if (global_tape != nullptr && idx < global_tape->nodes.size()) {
      return global_tape->nodes[idx].value;
    }
    return 0.0;
  }

  // Get adjoint (gradient)
  double adj() const {
    if (global_tape != nullptr && idx < global_tape->nodes.size()) {
      return global_tape->nodes[idx].adjoint;
    }
    return 0.0;
  }

  // Set value
  void set_val(double v) {
    if (global_tape != nullptr && idx < global_tape->nodes.size()) {
      global_tape->nodes[idx].value = v;
    }
  }

  // Compute gradients via backward pass
  void backward() {
    if (global_tape != nullptr) {
      global_tape->backward(idx);
    }
  }
};

// ---------------------------------------------------------------------
// Arithmetic operations with gradient tracking
// ---------------------------------------------------------------------

// Addition
inline Var operator+(const Var& a, const Var& b) {
  Var result(a.val() + b.val());

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint;
    global_tape->nodes[b_idx].adjoint += global_tape->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator+(const Var& a, double b) {
  Var result(a.val() + b);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator+(double a, const Var& b) {
  return b + a;
}

// Subtraction
inline Var operator-(const Var& a, const Var& b) {
  Var result(a.val() - b.val());

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint;
    global_tape->nodes[b_idx].adjoint -= global_tape->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator-(const Var& a, double b) {
  return a + (-b);
}

inline Var operator-(double a, const Var& b) {
  Var result(a - b.val());

  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [b_idx, r_idx]() {
    global_tape->nodes[b_idx].adjoint -= global_tape->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator-(const Var& a) {
  return 0.0 - a;
}

// Multiplication
inline Var operator*(const Var& a, const Var& b) {
  Var result(a.val() * b.val());

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;
  double a_val = a.val();
  double b_val = b.val();

  global_tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx, a_val, b_val]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * b_val;
    global_tape->nodes[b_idx].adjoint += global_tape->nodes[r_idx].adjoint * a_val;
  };

  return result;
}

inline Var operator*(const Var& a, double b) {
  Var result(a.val() * b);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, b]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * b;
  };

  return result;
}

inline Var operator*(double a, const Var& b) {
  return b * a;
}

// Division
inline Var operator/(const Var& a, const Var& b) {
  double b_val = b.val();
  Var result(a.val() / b_val);

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;
  double a_val = a.val();

  global_tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx, a_val, b_val]() {
    double adj = global_tape->nodes[r_idx].adjoint;
    global_tape->nodes[a_idx].adjoint += adj / b_val;
    global_tape->nodes[b_idx].adjoint -= adj * a_val / (b_val * b_val);
  };

  return result;
}

inline Var operator/(const Var& a, double b) {
  return a * (1.0 / b);
}

inline Var operator/(double a, const Var& b) {
  double b_val = b.val();
  Var result(a / b_val);

  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [b_idx, r_idx, a, b_val]() {
    global_tape->nodes[b_idx].adjoint -= global_tape->nodes[r_idx].adjoint * a / (b_val * b_val);
  };

  return result;
}

// ---------------------------------------------------------------------
// Mathematical functions
// ---------------------------------------------------------------------

inline Var exp(const Var& a) {
  double exp_val = std::exp(a.val());
  Var result(exp_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, exp_val]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * exp_val;
  };

  return result;
}

inline Var log(const Var& a) {
  double a_val = a.val();
  Var result(std::log(a_val));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, a_val]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint / a_val;
  };

  return result;
}

inline Var sqrt(const Var& a) {
  double sqrt_val = std::sqrt(a.val());
  Var result(sqrt_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, sqrt_val]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint / (2.0 * sqrt_val);
  };

  return result;
}

inline Var pow(const Var& a, double p) {
  double a_val = a.val();
  double pow_val = std::pow(a_val, p);
  Var result(pow_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, a_val, p]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * p * std::pow(a_val, p - 1.0);
  };

  return result;
}

// Softplus: log(1 + exp(x)) - numerically stable
inline Var softplus(const Var& a) {
  double a_val = a.val();
  double result_val;
  double sigmoid_val;

  if (a_val > 20.0) {
    result_val = a_val;
    sigmoid_val = 1.0;
  } else if (a_val < -20.0) {
    result_val = std::exp(a_val);
    sigmoid_val = result_val;
  } else {
    result_val = std::log1p(std::exp(a_val));
    sigmoid_val = 1.0 / (1.0 + std::exp(-a_val));
  }

  Var result(result_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, sigmoid_val]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * sigmoid_val;
  };

  return result;
}

// Log-sum-exp (numerically stable)
inline Var log_sum_exp(const Var& a, const Var& b) {
  double a_val = a.val();
  double b_val = b.val();
  double max_val = std::max(a_val, b_val);
  double result_val = max_val + std::log(std::exp(a_val - max_val) + std::exp(b_val - max_val));

  Var result(result_val);

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  // Softmax weights
  double w_a = std::exp(a_val - result_val);
  double w_b = std::exp(b_val - result_val);

  global_tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx, w_a, w_b]() {
    double adj = global_tape->nodes[r_idx].adjoint;
    global_tape->nodes[a_idx].adjoint += adj * w_a;
    global_tape->nodes[b_idx].adjoint += adj * w_b;
  };

  return result;
}

// Logit function: log(x / (1-x))
inline Var logit(const Var& a) {
  double a_val = a.val();
  Var result(std::log(a_val / (1.0 - a_val)));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, a_val]() {
    double deriv = 1.0 / (a_val * (1.0 - a_val));
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * deriv;
  };

  return result;
}

// Inverse logit (sigmoid): 1 / (1 + exp(-x))
inline Var inv_logit(const Var& a) {
  double a_val = a.val();
  double sigmoid;

  if (a_val >= 0) {
    sigmoid = 1.0 / (1.0 + std::exp(-a_val));
  } else {
    double exp_a = std::exp(a_val);
    sigmoid = exp_a / (1.0 + exp_a);
  }

  Var result(sigmoid);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, sigmoid]() {
    double deriv = sigmoid * (1.0 - sigmoid);
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * deriv;
  };

  return result;
}

// Log-gamma function
inline Var lgamma(const Var& a) {
  double a_val = a.val();
  Var result(R::lgammafn(a_val));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, a_val]() {
    double digamma_val = R::digamma(a_val);
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint * digamma_val;
  };

  return result;
}

// Log1p: log(1 + x) - numerically stable for small x
inline Var log1p(const Var& a) {
  double a_val = a.val();
  Var result(std::log1p(a_val));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  global_tape->nodes[r_idx].backward = [a_idx, r_idx, a_val]() {
    global_tape->nodes[a_idx].adjoint += global_tape->nodes[r_idx].adjoint / (1.0 + a_val);
  };

  return result;
}

// ---------------------------------------------------------------------
// Utility: Vector of Var
// ---------------------------------------------------------------------

inline std::vector<Var> make_vars(const std::vector<double>& values) {
  std::vector<Var> vars;
  vars.reserve(values.size());
  for (double v : values) {
    vars.emplace_back(v);
  }
  return vars;
}

inline std::vector<double> get_values(const std::vector<Var>& vars) {
  std::vector<double> values;
  values.reserve(vars.size());
  for (const auto& v : vars) {
    values.push_back(v.val());
  }
  return values;
}

inline std::vector<double> get_adjoints(const std::vector<Var>& vars) {
  std::vector<double> adjoints;
  adjoints.reserve(vars.size());
  for (const auto& v : vars) {
    adjoints.push_back(v.adj());
  }
  return adjoints;
}

} // namespace ad
} // namespace ratiod

#endif // QUOTR_AUTODIFF_H
