#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <vector>
#include <cmath>
#include <stdexcept>

/// @brief Cholesky decomposition of a symmetric positive-definite matrix
/// Decomposes correlation matrix C into L*L^T where L is lower triangular
/// @param corr Symmetric positive-definite correlation matrix
/// @return Lower triangular matrix L
/// @throws std::runtime_error if matrix is not positive-definite
[[nodiscard]] inline std::vector<std::vector<double>> choleskyDecomposition(
    const std::vector<std::vector<double>>& corr
) {
    int n = static_cast<int>(corr.size());
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                double diag = corr[i][i] - sum;
                if (diag <= 0.0) {
                    throw std::runtime_error(
                        "Cholesky decomposition failed: Matrix is not positive-definite. "
                        "Diagonal element at index " + std::to_string(i) +
                        " is non-positive: " + std::to_string(diag)
                    );
                }
                L[i][j] = std::sqrt(diag);
            } else {
                if (L[j][j] == 0.0) {
                    throw std::runtime_error(
                        "Cholesky decomposition failed: Division by zero at index " +
                        std::to_string(j)
                    );
                }
                L[i][j] = (corr[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

#endif // MATHUTILS_H
