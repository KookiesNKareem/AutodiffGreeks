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

void test_dual_addition() {
    Dual1 a(2.0, 1.0);
    Dual1 b(3.0, 2.0);
    Dual1 c = a + b;

    test("Dual addition value", approx_equal(c.val, 5.0));
    test("Dual addition derivative", approx_equal(c.der, 3.0));
}

void test_dual_multiplication() {
    Dual1 a(2.0, 1.0);  // x = 2, dx/dx = 1
    Dual1 b(3.0, 0.0);  // constant 3
    Dual1 c = a * b;    // 3x

    test("Dual multiplication value", approx_equal(c.val, 6.0));
    test("Dual multiplication derivative", approx_equal(c.der, 3.0));
}

void test_dual_division() {
    Dual1 a(6.0, 1.0);  // x = 6
    Dual1 b(2.0, 0.0);  // constant 2
    Dual1 c = a / b;    // x/2

    test("Dual division value", approx_equal(c.val, 3.0));
    test("Dual division derivative", approx_equal(c.der, 0.5));
}

void test_scalar_operations() {
    Dual1 x(2.0, 1.0);

    Dual1 r1 = x + 3.0;
    test("Dual + scalar", approx_equal(r1.val, 5.0) && approx_equal(r1.der, 1.0));

    Dual1 r2 = 3.0 * x;
    test("Scalar * Dual", approx_equal(r2.val, 6.0) && approx_equal(r2.der, 3.0));
}

void test_negation() {
    Dual1 x(2.0, 1.0);
    Dual1 neg = -x;

    test("Dual negation value", approx_equal(neg.val, -2.0));
    test("Dual negation derivative", approx_equal(neg.der, -1.0));
}

void test_comparison() {
    Dual1 a(2.0, 1.0);
    Dual1 b(3.0, 2.0);

    test("Dual comparison <", a < b);
    test("Dual comparison >", b > a);
    test("Dual comparison <=", a <= b);
    test("Dual comparison >=", b >= a);
}

void test_max() {
    Dual1 a(2.0, 1.0);
    Dual1 b(3.0, 2.0);
    Dual1 m = max(a, b);

    test("Dual max value", approx_equal(m.val, 3.0));
    test("Dual max derivative", approx_equal(m.der, 2.0));
}

int main() {
    std::cout << "Testing Dual Arithmetic Operations\n";
    std::cout << "===================================\n\n";

    test_dual_addition();
    test_dual_multiplication();
    test_dual_division();
    test_scalar_operations();
    test_negation();
    test_comparison();
    test_max();

    std::cout << "\n===================================\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
