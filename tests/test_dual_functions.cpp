#include <iostream>
#include <cmath>
#include <cassert>
#include "../Autodiff_Forward.h"

int tests_passed = 0;
int tests_failed = 0;

void test(const std::string& name, bool condition) {
    if (condition) {
        tests_passed++;
        std::cout << "✓ " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "✗ " << name << "\n";
    }
}

bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

void test_exp() {
    Dual1 x = Dual1::variable(2.0);  // x = 2, dx/dx = 1
    Dual1 result = exp(x);           // e^x

    double expected_val = std::exp(2.0);
    double expected_der = std::exp(2.0);  // d/dx[e^x] = e^x

    test("exp() value", approx_equal(result.val, expected_val));
    test("exp() derivative", approx_equal(result.der, expected_der));
}

void test_log() {
    Dual1 x = Dual1::variable(2.0);
    Dual1 result = log(x);  // ln(x)

    double expected_val = std::log(2.0);
    double expected_der = 1.0 / 2.0;  // d/dx[ln(x)] = 1/x

    test("log() value", approx_equal(result.val, expected_val));
    test("log() derivative", approx_equal(result.der, expected_der));
}

void test_sqrt() {
    Dual1 x = Dual1::variable(4.0);
    Dual1 result = sqrt(x);  // √x

    double expected_val = 2.0;
    double expected_der = 1.0 / (2.0 * 2.0);  // d/dx[√x] = 1/(2√x)

    test("sqrt() value", approx_equal(result.val, expected_val));
    test("sqrt() derivative", approx_equal(result.der, expected_der));
}

void test_pow() {
    Dual1 x = Dual1::variable(2.0);
    Dual1 result = pow(x, 3.0);  // x³

    double expected_val = 8.0;
    double expected_der = 3.0 * 4.0;  // d/dx[x³] = 3x²

    test("pow() value", approx_equal(result.val, expected_val));
    test("pow() derivative", approx_equal(result.der, expected_der));
}

void test_sin() {
    Dual1 x = Dual1::variable(M_PI / 4.0);
    Dual1 result = sin(x);

    double expected_val = std::sin(M_PI / 4.0);
    double expected_der = std::cos(M_PI / 4.0);  // d/dx[sin(x)] = cos(x)

    test("sin() value", approx_equal(result.val, expected_val));
    test("sin() derivative", approx_equal(result.der, expected_der));
}

void test_cos() {
    Dual1 x = Dual1::variable(M_PI / 4.0);
    Dual1 result = cos(x);

    double expected_val = std::cos(M_PI / 4.0);
    double expected_der = -std::sin(M_PI / 4.0);  // d/dx[cos(x)] = -sin(x)

    test("cos() value", approx_equal(result.val, expected_val));
    test("cos() derivative", approx_equal(result.der, expected_der));
}

void test_chain_rule() {
    // Test: d/dx[ln(x²)] = 2/x
    Dual1 x = Dual1::variable(2.0);
    Dual1 x_squared = x * x;
    Dual1 result = log(x_squared);

    double expected_der = 2.0 / 2.0;  // 2/x at x=2

    test("Chain rule: log(x²)", approx_equal(result.der, expected_der));
}

void test_nested_duals() {
    // Test second-order: d²/dx²[x²] = 2
    Dual2 x = Dual2::variable(Dual1::variable(3.0));
    Dual2 result = x * x;

    test("Second-order value", approx_equal(result.val.val, 9.0));
    test("First derivative (2x at x=3)", approx_equal(result.der.val, 6.0));
    test("Second derivative", approx_equal(result.der.der, 2.0));
}

int main() {
    std::cout << "Testing Dual Mathematical Functions\n";
    std::cout << "====================================\n\n";

    test_exp();
    test_log();
    test_sqrt();
    test_pow();
    test_sin();
    test_cos();
    test_chain_rule();
    test_nested_duals();

    std::cout << "\n====================================\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
