//team39_helper.h
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#ifndef TEAM39_HELPER_H
#define TEAM39_HELPER_H

using namespace Eigen;

inline bool isSymmetric(const SparseMatrix<double>& A, double tol = 1e-12) {
        return (A - SparseMatrix<double>(A.transpose())).norm() <= tol * A.norm();
}

// Builds the (m*n) x (m*n) sparse matrix A that applies the filter H to an
// m x n image stored as a vector (row by row): A * vec(F) = vec(F * H).
inline SparseMatrix<double> matrix_convolution(const MatrixXd& H, int m, int n) {
        // half-size of the kernel (1 for a 3x3 kernel)
        const int half_rows = (H.rows() - 1) / 2;
        const int half_cols = (H.cols() - 1) / 2;

        std::vector<Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(m) * n * H.size());

        // for every output pixel (i,j) ...
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        const int row = i * n + j;   // position of (i,j) in the vector

                        // ... add the weight of each neighbour (r,c) in the kernel window
                        for (int k = 0; k < H.rows(); ++k) {
                                for (int l = 0; l < H.cols(); ++l) {
                                        const int r = i + k - half_rows;
                                        const int c = j + l - half_cols;

                                        // neighbours outside the image count as zero
                                        if (r < 0 || r >= m || c < 0 || c >= n) continue;
                                        if (H(k, l) == 0.0) continue;

                                        triplets.emplace_back(row, r * n + c, H(k, l));
                                }
                        }
                }
        }

        SparseMatrix<double> A(m * n, m * n);
        A.setFromTriplets(triplets.begin(), triplets.end());
        return A;
}

inline double getArrayEuclideanNorm(VectorXd v) {
        return v.norm();
}

inline double getMatrixFrobeniusNorm(MatrixXd M) {
        return M.norm();
}

inline int getNumberOfZeroInMatrix(MatrixXd M) {
        int count = 0;
        for (int i = 0; i < M.rows(); i++) {
                for (int j = 0; j < M.cols(); j++) {
                        if (M(i,j) == 0) {
                                count++;
                        }
                }
        }
        return count;
        
}
#endif