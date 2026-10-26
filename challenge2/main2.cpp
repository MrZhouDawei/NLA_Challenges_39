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
#include <lis.h>

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
	
	Eigen::SparseMatrix<double> ATA_sparse = ATA.sparseView();

	saveMarket(ATA_sparse, "output2/ATA.mtx");

	LIS_MATRIX ATA_lis;
	LIS_VECTOR x_lis;
	LIS_ESOLVER esolver;

	lis_initialize(&argc, &argv);

	lis_matrix_create(LIS_COMM_WORLD, &ATA_lis);
	lis_matrix_set_size(ATA_lis, 0, ATA.rows());
	lis_matrix_set_type(ATA_lis, LIS_MATRIX_CSR);

	for (int i = 0; i < ATA.rows(); ++i) {
		for (int j = 0; j < ATA.cols(); ++j) {
			if (ATA(i,j) != 0.0) {
				lis_matrix_set_value(LIS_INS_VALUE,i,j,ATA(i,j),ATA_lis);
			}
		}
	}

	lis_matrix_assemble(ATA_lis);

	lis_vector_duplicate(ATA_lis, &x_lis);


// Use Power Iteration with tolerance 1e-10
	lis_esolver_create(&esolver);

	char options_task3[] = "-e pi -etol 1e-10";
	lis_esolver_set_option(options_task3, esolver);

	LIS_SCALAR eigenvalue_lis;

	lis_esolve(
		ATA_lis,
		x_lis,
		&eigenvalue_lis,
		esolver
	);
	LIS_INT iter_task3;
    LIS_REAL residual_task3;

    lis_esolver_get_iter(esolver, &iter_task3);
    lis_esolver_get_residualnorm(esolver, &residual_task3);

    std::cout << "Iterations without shift = " << iter_task3 << std::endl;
    std::cout << "Residual without shift = " << residual_task3 << std::endl;

	std::cout << "Largest eigenvalue with LIS: " << eigenvalue_lis << std::endl;
	std::cout << "Largest eigenvalue with Eigen: " << lambda_max1 << std::endl;
	
	// ---------------------------------------------- TASK 4 -----------------------------------------

	LIS_ESOLVER esolver_shift;
	LIS_VECTOR x_shift;
	LIS_SCALAR eigenvalue_shift;

	lis_vector_duplicate(ATA_lis, &x_shift);
	lis_esolver_create(&esolver_shift);

	// Shift value
	double mu = 0.5 * lambda_max2;

	std::string opts = "-e pi -etol 1e-10 -shift " + std::to_string(mu);

	std::vector<char> opts_char(opts.begin(), opts.end());
	opts_char.push_back('\0');

	lis_esolver_set_option(opts_char.data(), esolver_shift);

	lis_esolve(
		ATA_lis,
		x_shift,
		&eigenvalue_shift,
		esolver_shift
	);

	LIS_INT iter_shift;
	LIS_REAL residual_shift;

	lis_esolver_get_iter(esolver_shift, &iter_shift);
	lis_esolver_get_residualnorm(esolver_shift, &residual_shift);

	std::cout << "Shift mu = " << mu << std::endl;
	std::cout << "Residual with shift = " << residual_shift << std::endl;
	std::cout << "Iterations with shift = " << iter_shift << std::endl;
	
	
	lis_esolver_destroy(esolver);
	lis_esolver_destroy(esolver_shift);

	lis_vector_destroy(x_lis);
	lis_vector_destroy(x_shift);

	lis_matrix_destroy(ATA_lis);

	lis_finalize();
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


	// Singular Value Decomposition of the NOISY matrix
	Eigen::BDCSVD<Eigen::MatrixXd> svd_noisy(noisy, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::VectorXd singular_values_noisy = svd_noisy.singularValues();

	std::cout << "1st largest singular value of noisy matrix: " << singular_values_noisy[0] << std::endl;
	std::cout << "2nd largest singular value of noisy matrix: " << singular_values_noisy[1] << std::endl;


	//----------------------------------------------TASK 11-----------------------------------------

	Eigen::MatrixXd U_noisy = svd_noisy.matrixU();
	Eigen::MatrixXd V_noisy = svd_noisy.matrixV();

	// k = 6
	Eigen::MatrixXd C6(U_noisy.rows(), 6);
	Eigen::MatrixXd D6(V_noisy.rows(), 6);

	for (int i = 0; i < 6; i++){
		C6.col(i) = U_noisy.col(i);
		D6.col(i) = singular_values_noisy[i] * V_noisy.col(i);
	}

	std::cout << "Size of C with k = 6: " << C6.rows() << " x " << C6.cols() << std::endl;
	std::cout << "Size of D with k = 6: " << D6.rows() << " x " << D6.cols() << std::endl;

	// k = 12
	Eigen::MatrixXd C12(U_noisy.rows(), 12);
	Eigen::MatrixXd D12(V_noisy.rows(), 12);

	for (int i = 0; i < 12; i++){
		C12.col(i) = U_noisy.col(i);
		D12.col(i) = singular_values_noisy[i] * V_noisy.col(i);
	}

	std::cout << "Size of C with k = 12: " << C12.rows() << " x " << C12.cols() << std::endl;
	std::cout << "Size of D with k = 12: " << D12.rows() << " x " << D12.cols() << std::endl;


	//----------------------------------------------TASK 12-----------------------------------------
	// Reconstruct the noisy checkerboard using truncated SVD.
	// k = 6 keeps fewer singular components and gives stronger compression.
	// k = 12 keeps more information and should be closer to the original image.
	
	Eigen::MatrixXd checker_k6 = C6 * D6.transpose();
	Eigen::MatrixXd checker_k12 = C12 * D12.transpose();

	Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor> checker6_bytes(dim, dim);
	Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor> checker12_bytes(dim, dim);
	
	for(int i = 0; i < dim; ++i) {
		for(int j = 0; j < dim; ++j) {
			double val =checker_k6(i, j);
			val = std::max(0.0, std::min(255.0, val));
			checker6_bytes(i, j) = static_cast<unsigned char>(std::round(val));
		}
	}

	for(int i = 0; i < dim; ++i) {
		for(int j = 0; j < dim; ++j) {
			double val = checker_k12(i, j);
			val = std::max(0.0, std::min(255.0, val));
			checker12_bytes(i, j) = static_cast<unsigned char>(std::round(val));
		}
	}

	stbi_write_png(
		"output2/checker_k6.png",
		dim, dim, 1,
		checker6_bytes.data(),
		dim
	);

	stbi_write_png(
		"output2/checker_k12.png",
		dim, dim, 1,
		checker12_bytes.data(),
		dim
	);
	
	//----------------------------------------------TASK 13-----------------------------------------







return 0;

}

