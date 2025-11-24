#ifndef AUTODIFF_FORWARD_H
#define AUTODIFF_FORWARD_H

#include <cmath>
#include <type_traits>

template<typename T = double>
struct Dual;

using Dual1 = Dual<double>; // first order
using Dual2 = Dual<Dual<double>>; // second order

/// @brief Dual number for forward-mode automatic differentiation
/// Supports nested differentiation via template parameter
/// @tparam T Underlying numeric type (double or Dual<U>)
template<typename T>
struct Dual {
    T val;
    T der;

    // construct dual
    constexpr Dual(T v = T(0.0), T d = T(0.0)) noexcept : val(v), der(d) {}

    // constant - zero derivative
    [[nodiscard]] static constexpr Dual constant(T v) noexcept {
        return Dual(v, T(0.0));
    }

    /// variable - unit derivative
    [[nodiscard]] static constexpr Dual variable(T v) noexcept {
        return Dual(v, T(1.0));
    }
};

// ARITHMETIC OPERATORS

// Addition: (a, a') + (b, b') = (a+b, a'+b')
template<typename T>
[[nodiscard]] constexpr Dual<T> operator+(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>(a.val + b.val, a.der + b.der);
}

// Subtraction: (a, a') - (b, b') = (a-b, a'-b')
template<typename T>
[[nodiscard]] constexpr Dual<T> operator-(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>(a.val - b.val, a.der - b.der);
}

// Multiplication: Product rule d/dx[f*g] = f*g' + g*f'
template<typename T>
[[nodiscard]] constexpr Dual<T> operator*(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>(
        a.val * b.val,
        a.val * b.der + b.val * a.der
    );
}

// Division: Quotient rule d/dx[f/g] = (f'g - fg') / g²
template<typename T>
[[nodiscard]] constexpr Dual<T> operator/(const Dual<T>& a, const Dual<T>& b) noexcept {
    return Dual<T>(
        a.val / b.val,
        (a.der * b.val - a.val * b.der) / (b.val * b.val)
    );
}

/// Negation
template<typename T>
[[nodiscard]] constexpr Dual<T> operator-(const Dual<T>& a) noexcept {
    return Dual<T>(-a.val, -a.der);
}

// SCALAR OPERATIONS (Dual with double)

template<typename T>
[[nodiscard]] constexpr Dual<T> operator+(double a, const Dual<T>& b) noexcept {
    return Dual<T>(a + b.val, b.der);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator+(const Dual<T>& a, double b) noexcept {
    return Dual<T>(a.val + b, a.der);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator*(double a, const Dual<T>& b) noexcept {
    return Dual<T>(a * b.val, a * b.der);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator*(const Dual<T>& a, double b) noexcept {
    return Dual<T>(a.val * b, a.der * b);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator-(const Dual<T>& a, double b) noexcept {
    return Dual<T>(a.val - b, a.der);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator-(double a, const Dual<T>& b) noexcept {
    return Dual<T>(a - b.val, -b.der);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator/(const Dual<T>& a, double b) noexcept {
    return Dual<T>(a.val / b, a.der / b);
}

template<typename T>
[[nodiscard]] constexpr Dual<T> operator/(double a, const Dual<T>& b) noexcept {
    return Dual<T>(
        a / b.val,
        -a * b.der / (b.val * b.val)
    );
}

// SCALAR OPERATIONS (Dual with T)

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator+(const T& a, const Dual<T>& b) noexcept {
    return Dual<T>(a + b.val, b.der);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator+(const Dual<T>& a, const T& b) noexcept {
    return Dual<T>(a.val + b, a.der);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator*(const T& a, const Dual<T>& b) noexcept {
    return Dual<T>(a * b.val, a * b.der);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator*(const Dual<T>& a, const T& b) noexcept {
    return Dual<T>(a.val * b, a.der * b);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator-(const Dual<T>& a, const T& b) noexcept {
    return Dual<T>(a.val - b, a.der);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator-(const T& a, const Dual<T>& b) noexcept {
    return Dual<T>(a - b.val, -b.der);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator/(const Dual<T>& a, const T& b) noexcept {
    return Dual<T>(a.val / b, a.der / b);
}

template<typename T, typename = typename std::enable_if<!std::is_same<T, double>::value>::type>
[[nodiscard]] constexpr Dual<T> operator/(const T& a, const Dual<T>& b) noexcept {
    return Dual<T>(
        a / b.val,
        -a * b.der / (b.val * b.val)
    );
}

// MATHEMATICAL FUNCTIONS

// Exponential: d/dx[e^x] = e^x
template<typename T>
[[nodiscard]] Dual<T> exp(const Dual<T>& a) {
    T e = exp(a.val);
    return Dual<T>(e, e * a.der);
}

// Natural logarithm: d/dx[ln(x)] = 1/x
template<typename T>
[[nodiscard]] Dual<T> log(const Dual<T>& a) {
    return Dual<T>(log(a.val), a.der / a.val);
}

// Square root: d/dx[√x] = 1/(2√x)
template<typename T>
[[nodiscard]] Dual<T> sqrt(const Dual<T>& a) {
    T s = sqrt(a.val);
    return Dual<T>(s, a.der / (T(2.0) * s));
}

// Power: d/dx[x^n] = n*x^(n-1)
template<typename T>
[[nodiscard]] Dual<T> pow(const Dual<T>& a, double n) {
    T p = pow(a.val, n);
    return Dual<T>(p, T(n) * pow(a.val, n - 1) * a.der);
}

// Sin: d/dx[sin(x)] = cos(x)
template<typename T>
[[nodiscard]] Dual<T> sin(const Dual<T>& a) {
    return Dual<T>(sin(a.val), cos(a.val) * a.der);
}

// Cos: d/dx[cos(x)] = -sin(x)
template<typename T>
[[nodiscard]] Dual<T> cos(const Dual<T>& a) {
    return Dual<T>(cos(a.val), -sin(a.val) * a.der);
}

// Error function: d/dx[erf(x)] = (2/√π) * e^(-x²)
template<typename T>
[[nodiscard]] Dual<T> erf(const Dual<T>& a) {
    T erf_deriv = T(2.0 / std::sqrt(M_PI)) * exp(-a.val * a.val);
    return Dual<T>(erf(a.val), erf_deriv * a.der);
}

// COMPARISON OPERATORS

template<typename T>
[[nodiscard]] constexpr bool operator>(const Dual<T>& a, const Dual<T>& b) noexcept {
    return a.val > b.val;
}

template<typename T>
[[nodiscard]] constexpr bool operator<(const Dual<T>& a, const Dual<T>& b) noexcept {
    return a.val < b.val;
}

template<typename T>
[[nodiscard]] constexpr bool operator>=(const Dual<T>& a, const Dual<T>& b) noexcept {
    return a.val >= b.val;
}

template<typename T>
[[nodiscard]] constexpr bool operator<=(const Dual<T>& a, const Dual<T>& b) noexcept {
    return a.val <= b.val;
}

// Maximum
template<typename T>
[[nodiscard]] constexpr Dual<T> max(const Dual<T>& a, const Dual<T>& b) noexcept {
    return (a.val > b.val) ? a : b;
}

template<typename T>
[[nodiscard]] constexpr Dual<T> max(const Dual<T>& a, double b) noexcept {
    return (a.val > b) ? a : Dual<T>(T(b), T(0.0));
}

[[nodiscard]] inline constexpr double max(double a, double b) noexcept {
    return (a > b) ? a : b;
}

#endif // AUTODIFF_FORWARD_H
