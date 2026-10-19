// challenge 2
//
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <cmath>
#include <Eigen/SVD>
// #include <lis.h>

#define STB_IMAGE_IMPLEMENTATION
#include "resources2/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "resources2/stb_image_write.h"

using namespace Eigen;


int main(int argc, char* argv[]) {
	if (argc < 2) {
   	std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
    	return 1;
  }	


	//----------------------------------------------TASK 1-----------------------------------------
	
	const char* image_path = argv[1];
	int width = 0;
    	int height = 0;
    	int channels = 0;

       	unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 1);
       	if (!image_data) {
    		std::cerr << "Error: I could not load the image " << image_path << std::endl;
	    	return 1;
       	}
	std::cout << "Image dimension: " << height << " x " << width << std::endl;

       	//create the image matrix 
       	int m = height;
     	int n = width;
    	Eigen::MatrixXd A(m, n);

    	for (int i = 0; i < m; ++i) {
	  	for (int j = 0; j < n; ++j) {
	     		A(i, j) = static_cast<double>(image_data[i * n + j]);
	 	}
     	}
      	stbi_image_free(image_data);
    
	// Calculating ATA
    	Eigen::MatrixXd ATA = A.transpose() * A;

       	// Frobenius norm of ATA
    	double norm_ATA = ATA.norm();

       	std::cout << "Dimension of A: " << A.rows() << " x " << A.cols() << std::endl;
       	std::cout << "Dimension of A^T*A: " << ATA.rows() << " x " << ATA.cols() << std::endl;
       	std::cout << "Frobenius norm of A^T*A: " << norm_ATA << std::endl;
	
	
	//----------------------------------------------TASK 2-----------------------------------------

	// ATA is symmetric and positive defined
	// as a consequence we use SelfAdjointEigenSolver
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(ATA);
   
	// Check
    	if (eigensolver.info() != Eigen::Success) {
		std::cout << "Error in the computation of eigenvalues" << std::endl;
		return 1;
	}
       	// Take eigenvalues
	Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
	long eigenvalues_size = eigenvalues.size();
	double lambda_max1 = eigenvalues(eigenvalues_size - 1);
	double lambda_max2 = eigenvalues(eigenvalues_size - 2);

	// computation of the two largest singular values of A
	double sigma_1 = std::sqrt(std::max(0.0, lambda_max1));
	double sigma_2 = std::sqrt(std::max(0.0, lambda_max2));
    
	std::cout << "1st largest singular value of A: " << sigma_1 << std::endl;
	std::cout << "2nd largest singular value of A: " << sigma_2 << std::endl;
	
	
	//----------------------------------------------TASK 3-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 4-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 5-----------------------------------------
	
	
	// Singular Value Decomposition of A
	Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::VectorXd singular_values = svd.singularValues();

	// NOTE: Frobenius noem of SIGMA
	double norm_SIGMA = singular_values.norm();
	std::cout << "Frobenius norm of diagonal matrix SIGMA: " << norm_SIGMA << std::endl;
	
	
	//----------------------------------------------TASK 6-----------------------------------------
	
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();

	// k = 30
	Eigen::MatrixXd C30(U.rows(), 30);
	Eigen::MatrixXd D30(V.rows(), 30);

	for (int i = 0; i<30; i++){
		C30.col(i) = U.col(i);
		D30.col(i) = singular_values[i] * V.col(i);
	}

	long nnz_C30 = (C30.array() != 0.0).count();
	long nnz_D30 = (D30.array() != 0.0).count();

	std::cout << "nnz(C with k = 30) = " << nnz_C30 << std::endl;
	std::cout << "nnz(D with k = 30) = " << nnz_D30 << std::endl;

	//k = 90	
	Eigen::MatrixXd C90(U.rows(), 90);
	Eigen::MatrixXd D90(V.rows(), 90);

	for (int i = 0; i < 90; i++) {
	     	C90.col(i) = U.col(i);
	    	D90.col(i) = singular_values[i] * V.col(i);
	}
	long nnz_C90 = (C90.array() != 0.0).count();
	long nnz_D90 = (D90.array() != 0.0).count();

	std::cout << "nnz(C with k = 90) = " << nnz_C90 << std::endl;
	std::cout << "nnz(D with k = 90) = " << nnz_D90 << std::endl;
	
	//----------------------------------------------TASK 7-----------------------------------------
	
	Eigen::MatrixXd A_new_30 = C30 * D30.transpose();
	Eigen::MatrixXd A_new_90 = C90 * D90.transpose();
	
	std::vector<unsigned char> img30(m * n);
	std::vector<unsigned char> img90(m * n);

	// k = 30
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			double val30 = A_new_30(i, j);
			val30 = std::max(0.0, std::min(255.0, val30));
		       	img30[i * n + j] = static_cast<unsigned char>(val30);
	       	}
      	}
	stbi_write_png("output2/deer_task7_k30.png", width, height, 1, img30.data(), width);


	// k = 90
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
		       double val90 = A_new_90(i, j);
		       val90 = std::max(0.0, std::min(255.0, val90));
		       img90[i * n + j] = static_cast<unsigned char>(val90);
		}
	}
	stbi_write_png("output2/deer_task7_k90.png", width, height, 1, img90.data(), width);
    


	
	//----------------------------------------------TASK 8-----------------------------------------
	
	const int dim = 400;
	const int blocks_per_row = 10; 
	const int block_size = dim / blocks_per_row;

	Eigen::MatrixXd A_checker(dim, dim);

	for (int i = 0; i < dim; i++) {
	      	int block_row = i / block_size;
	       	for (int j = 0; j < dim; j++) {
			int block_col = j / block_size;
			if ((block_row + block_col) % 2 == 0) {
				//black block
			      	A_checker(i, j) = 0.0;
			} else {
				// white block
				A_checker(i, j) = 255.0;
			}
	       	}
	}

	// Froenius norm
	double norm_checker = A_checker.norm();
	
	std::cout << "Euclidean norm of checkerboard matrix: " << norm_checker << std::endl;


	//----------------------------------------------TASK 9-----------------------------------------
	

	// NOISY matix
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-60.0, 60.0);
  
	Eigen::MatrixXd noisy(dim, dim);
  
	for (int i = 0; i < dim; i++){
	      	for (int j = 0; j < dim; j++) {
	      		double new_val = A_checker(i,j) + dist(gen);
	      		noisy(i,j) = std::min(std::max(new_val, 0.0), 255.0); 
		}
       	}

	// export the NOISY matrix
	Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_bytes(dim, dim);
       	for (int i = 0; i < dim; i++) {
	       	for (int j = 0; j < dim; j++) {
		      	noisy_bytes(i, j) = static_cast<unsigned char>(std::round(noisy(i, j)));
	       	}
       	}
       	stbi_write_png("output2/noisy_checkerboard.png", dim, dim, 1, noisy_bytes.data(), dim);
	
	
	//----------------------------------------------TASK 10-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 11-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 12-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 13-----------------------------------------







return 0;

}

