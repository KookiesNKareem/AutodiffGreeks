#ifndef AUTODIFF_REVERSE_H
#define AUTODIFF_REVERSE_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

class Tape;
class Var;

/// @brief Entry in the computational tape for reverse-mode AD
struct TapeEntry {
    double adjoint = 0.0;     // adjoint (∂output/∂this)

    // indices of inputs
    int input1 = -1;
    int input2 = -1;   

    // local partials
    double d_input1 = 0.0;    // ∂this/∂input1
    double d_input2 = 0.0;    // ∂this/∂input2
};

/// @brief Computational tape for reverse-mode automatic differentiation
/// Records operations during forward pass, computes gradients during backward pass
class Tape {
public:
    std::vector<TapeEntry> entries;

    /// @brief Add new entry to tape
    /// @param d1 Partial derivative w.r.t. first input
    /// @param i1 Index of first input (-1 if none)
    /// @param d2 Partial derivative w.r.t. second input
    /// @param i2 Index of second input (-1 if none)
    /// @return Index of the new entry
    [[nodiscard]] int push(double d1 = 0, int i1 = -1, double d2 = 0, int i2 = -1) {
        int idx = static_cast<int>(entries.size());
        entries.push_back({0.0, i1, i2, d1, d2});
        return idx;
    }

    /// @brief Reset all adjoints to zero
    void reset_adjoints() noexcept {
        for (auto& entry : entries) {
            entry.adjoint = 0.0;
        }
    }

    /// @brief Perform backward pass (backpropagation)
    /// @param output_idx Index of output variable to differentiate
    void backward(int output_idx) {
        if (output_idx < 0 || output_idx >= static_cast<int>(entries.size())) {
            throw std::out_of_range(
                "Invalid output index for backward pass. "
                "Index: " + std::to_string(output_idx) +
                ", Tape size: " + std::to_string(entries.size())
            );
        }

        reset_adjoints();

        // Seed: ∂f/∂f = 1
        entries[output_idx].adjoint = 1.0;

        // Backpropagation: accumulate adjoints in reverse order
        for (int i = output_idx; i >= 0; i--) {
            const TapeEntry& e = entries[i];

            if (e.input1 >= 0) {
                entries[e.input1].adjoint += e.adjoint * e.d_input1;
            }
            if (e.input2 >= 0) {
                entries[e.input2].adjoint += e.adjoint * e.d_input2;
            }
        }
    }

    /// @brief Get gradient (adjoint) for a variable
    /// @param var_idx Index of variable in tape
    /// @return Gradient ∂output/∂var
    [[nodiscard]] double grad(int var_idx) const {
        if (var_idx < 0 || var_idx >= static_cast<int>(entries.size())) {
            throw std::out_of_range(
                "Invalid variable index for gradient. "
                "Index: " + std::to_string(var_idx) +
                ", Tape size: " + std::to_string(entries.size())
            );
        }
        return entries[var_idx].adjoint;
    }

    /// @brief Clear all tape entries
    void clear() noexcept {
        entries.clear();
    }
};

/// @brief Variable for reverse-mode automatic differentiation
/// Each variable is associated with a computational tape
class Var {
public:
    double val;                        // Variable value
    int index;                         // Position in tape
    std::shared_ptr<Tape> tape;        // Shared tape for this computation

    /// @brief Construct variable with specified value and tape
    /// Creates new tape entry
    Var(double v, std::shared_ptr<Tape> t);

    /// @brief Construct variable with value, index, and tape
    /// Used internally for operation results
    Var(double v, int idx, std::shared_ptr<Tape> t) : val(v), index(idx), tape(t) {
        if (!tape) {
            throw std::invalid_argument(
                "Var construction failed: Tape pointer is null. "
                "Ensure tape is initialized with std::make_shared<Tape>()"
            );
        }
    }
};

inline Var::Var(double v, std::shared_ptr<Tape> t) : val(v), tape(t) {
    if (!tape) {
        throw std::invalid_argument(
            "Var construction failed: Tape pointer is null. "
            "Ensure tape is initialized with std::make_shared<Tape>()"
        );
    }
    index = t->push();
}

/// @brief Validate that two variables belong to the same tape
inline void validate_same_tape(const Var& a, const Var& b) {
    if (a.tape != b.tape) {
        throw std::invalid_argument(
            "Variables must belong to the same tape. "
            "Cannot perform operations between variables from different tapes."
        );
    }
}

// ARITHMETIC OPERATORS

// Addition
[[nodiscard]] inline Var operator+(const Var& a, const Var& b) {
    validate_same_tape(a, b);
    int idx = a.tape->push(1.0, a.index, 1.0, b.index);
    return Var(a.val + b.val, idx, a.tape);
}

// Subtraction
[[nodiscard]] inline Var operator-(const Var& a, const Var& b) {
    validate_same_tape(a, b);
    int idx = a.tape->push(1.0, a.index, -1.0, b.index);
    return Var(a.val - b.val, idx, a.tape);
}

// Multiplication
[[nodiscard]] inline Var operator*(const Var& a, const Var& b) {
    validate_same_tape(a, b);
    int idx = a.tape->push(b.val, a.index, a.val, b.index);
    return Var(a.val * b.val, idx, a.tape);
}

// Division
[[nodiscard]] inline Var operator/(const Var& a, const Var& b) {
    validate_same_tape(a, b);
    if (b.val == 0.0) {
        throw std::domain_error(
            "Division by zero in Var operator/. "
            "Divisor value is zero."
        );
    }
    int idx = a.tape->push(
        1.0 / b.val, a.index,
        -a.val / (b.val * b.val), b.index
    );
    return Var(a.val / b.val, idx, a.tape);
}

// Negation
[[nodiscard]] inline Var operator-(const Var& a) {
    int idx = a.tape->push(-1.0, a.index);
    return Var(-a.val, idx, a.tape);
}

// SCALAR OPERATIONS

[[nodiscard]] inline Var operator+(const Var& a, double b) {
    int idx = a.tape->push(1.0, a.index);
    return Var(a.val + b, idx, a.tape);
}

[[nodiscard]] inline Var operator+(double a, const Var& b) {
    int idx = b.tape->push(1.0, b.index);
    return Var(a + b.val, idx, b.tape);
}

[[nodiscard]] inline Var operator*(const Var& a, double b) {
    int idx = a.tape->push(b, a.index);
    return Var(a.val * b, idx, a.tape);
}

[[nodiscard]] inline Var operator*(double a, const Var& b) {
    int idx = b.tape->push(a, b.index);
    return Var(a * b.val, idx, b.tape);
}

[[nodiscard]] inline Var operator-(const Var& a, double b) {
    int idx = a.tape->push(1.0, a.index);
    return Var(a.val - b, idx, a.tape);
}

[[nodiscard]] inline Var operator-(double a, const Var& b) {
    int idx = b.tape->push(-1.0, b.index);
    return Var(a - b.val, idx, b.tape);
}

[[nodiscard]] inline Var operator/(const Var& a, double b) {
    if (b == 0.0) {
        throw std::domain_error(
            "Division by zero in Var operator/. "
            "Scalar divisor is zero."
        );
    }
    int idx = a.tape->push(1.0 / b, a.index);
    return Var(a.val / b, idx, a.tape);
}

[[nodiscard]] inline Var operator/(double a, const Var& b) {
    if (b.val == 0.0) {
        throw std::domain_error(
            "Division by zero in Var operator/. "
            "Divisor value is zero."
        );
    }
    int idx = b.tape->push(-a / (b.val * b.val), b.index);
    return Var(a / b.val, idx, b.tape);
}

// MATHEMATICAL FUNCTIONS

// Exponential
[[nodiscard]] inline Var exp(const Var& a) {
    double e = std::exp(a.val);
    int idx = a.tape->push(e, a.index);
    return Var(e, idx, a.tape);
}

// Natural logarithm
[[nodiscard]] inline Var log(const Var& a) {
    if (a.val <= 0.0) {
        throw std::domain_error(
            "log: argument must be positive. "
            "Value: " + std::to_string(a.val)
        );
    }
    int idx = a.tape->push(1.0 / a.val, a.index);
    return Var(std::log(a.val), idx, a.tape);
}

// Square root
[[nodiscard]] inline Var sqrt(const Var& a) {
    if (a.val < 0.0) {
        throw std::domain_error(
            "sqrt: argument must be non-negative. "
            "Value: " + std::to_string(a.val)
        );
    }
    double s = std::sqrt(a.val);
    int idx = a.tape->push(0.5 / s, a.index);
    return Var(s, idx, a.tape);
}

// Power
[[nodiscard]] inline Var pow(const Var& a, double n) {
    double p = std::pow(a.val, n);
    int idx = a.tape->push(n * std::pow(a.val, n - 1), a.index);
    return Var(p, idx, a.tape);
}

// Sin
[[nodiscard]] inline Var sin(const Var& a) {
    int idx = a.tape->push(std::cos(a.val), a.index);
    return Var(std::sin(a.val), idx, a.tape);
}

// Cos
[[nodiscard]] inline Var cos(const Var& a) {
    int idx = a.tape->push(-std::sin(a.val), a.index);
    return Var(std::cos(a.val), idx, a.tape);
}

// Error function
[[nodiscard]] inline Var erf(const Var& a) {
    double erf_deriv = (2.0 / std::sqrt(M_PI)) * std::exp(-a.val * a.val);
    int idx = a.tape->push(erf_deriv, a.index);
    return Var(std::erf(a.val), idx, a.tape);
}

// Maximum (preserves derivative of the larger value)
[[nodiscard]] inline Var max(const Var& a, const Var& b) {
    validate_same_tape(a, b);
    if (a.val > b.val) {
        int idx = a.tape->push(1.0, a.index, 0.0, b.index);
        return Var(a.val, idx, a.tape);
    } else {
        int idx = a.tape->push(0.0, a.index, 1.0, b.index);
        return Var(b.val, idx, a.tape);
    }
}

[[nodiscard]] inline Var max(const Var& a, double b) {
    if (a.val > b) {
        int idx = a.tape->push(1.0, a.index);
        return Var(a.val, idx, a.tape);
    } else {
        int idx = a.tape->push(0.0, a.index);
        return Var(b, idx, a.tape);
    }
}

#endif // AUTODIFF_REVERSE_H
