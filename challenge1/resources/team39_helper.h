//team39_helper.h
#include <Eigen/Dense>

#ifndef TEAM39_HELPER_H
#define TEAM39_HELPER_H

using namespaceEigen;

inline bool isSymmetric( MatrixXd M){
        return M.isApprox(M.transpose());
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