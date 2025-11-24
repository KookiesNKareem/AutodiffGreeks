#include <iostream>
#include <cmath>
#include <memory>
#include "../Autodiff_Reverse.h"

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

void test_basic_gradient() {
    auto tape = std::make_shared<Tape>();

    // f(x, y) = x + y
    Var x(2.0, tape);
    Var y(3.0, tape);
    Var result = x + y;

    tape->backward(result.index);

    test("Addition gradient wrt x", approx_equal(tape->grad(x.index), 1.0));
    test("Addition gradient wrt y", approx_equal(tape->grad(y.index), 1.0));
}

void test_product_rule() {
    auto tape = std::make_shared<Tape>();

    // f(x, y) = x * y, df/dx = y, df/dy = x
    Var x(2.0, tape);
    Var y(3.0, tape);
    Var result = x * y;

    tape->backward(result.index);

    test("Multiplication gradient wrt x", approx_equal(tape->grad(x.index), 3.0));
    test("Multiplication gradient wrt y", approx_equal(tape->grad(y.index), 2.0));
}

void test_chain_rule() {
    auto tape = std::make_shared<Tape>();

    // f(x) = (x + 1) * (x + 1) = x² + 2x + 1
    // df/dx = 2x + 2
    Var x(2.0, tape);
    Var x_plus_1 = x + 1.0;
    Var result = x_plus_1 * x_plus_1;

    tape->backward(result.index);

    double expected = 2.0 * 2.0 + 2.0;  // 2x + 2 at x=2
    test("Chain rule gradient", approx_equal(tape->grad(x.index), expected));
}

void test_exp_gradient() {
    auto tape = std::make_shared<Tape>();

    // f(x) = e^x, df/dx = e^x
    Var x(1.0, tape);
    Var result = exp(x);

    tape->backward(result.index);

    double expected = std::exp(1.0);
    test("exp() gradient", approx_equal(tape->grad(x.index), expected));
}

void test_log_gradient() {
    auto tape = std::make_shared<Tape>();

    // f(x) = ln(x), df/dx = 1/x
    Var x(2.0, tape);
    Var result = log(x);

    tape->backward(result.index);

    double expected = 1.0 / 2.0;
    test("log() gradient", approx_equal(tape->grad(x.index), expected));
}

void test_multiple_inputs() {
    auto tape = std::make_shared<Tape>();

    // f(x, y, z) = x * y + z
    // df/dx = y, df/dy = x, df/dz = 1
    Var x(2.0, tape);
    Var y(3.0, tape);
    Var z(1.0, tape);
    Var result = x * y + z;

    tape->backward(result.index);

    test("Multi-input gradient wrt x", approx_equal(tape->grad(x.index), 3.0));
    test("Multi-input gradient wrt y", approx_equal(tape->grad(y.index), 2.0));
    test("Multi-input gradient wrt z", approx_equal(tape->grad(z.index), 1.0));
}

void test_complex_function() {
    auto tape = std::make_shared<Tape>();

    // f(x) = e^(x²)
    // df/dx = 2x * e^(x²)
    Var x(2.0, tape);
    Var x_squared = x * x;
    Var result = exp(x_squared);

    tape->backward(result.index);

    double expected = 2.0 * 2.0 * std::exp(4.0);  // 2x * e^(x²) at x=2
    test("Complex function gradient", approx_equal(tape->grad(x.index), expected));
}

void test_max_gradient() {
    auto tape = std::make_shared<Tape>();

    // f(x, y) = max(x, y)
    Var x(3.0, tape);
    Var y(2.0, tape);
    Var result = max(x, y);

    tape->backward(result.index);

    test("max() gradient wrt larger", approx_equal(tape->grad(x.index), 1.0));
    test("max() gradient wrt smaller", approx_equal(tape->grad(y.index), 0.0));
}

int main() {
    std::cout << "Testing Var/Tape Reverse-Mode Operations\n";
    std::cout << "=========================================\n\n";

    test_basic_gradient();
    test_product_rule();
    test_chain_rule();
    test_exp_gradient();
    test_log_gradient();
    test_multiple_inputs();
    test_complex_function();
    test_max_gradient();

    std::cout << "\n=========================================\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
