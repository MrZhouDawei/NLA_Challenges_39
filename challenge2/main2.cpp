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
// #include <lis.h>

#define STB_IMAGE_IMPLEMENTATION
#include "resources2/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "resources2/stb_image_write.h"

int main(int argc, char* argv[]) {
	if (argc < 2) {
   	std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
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
	
	
	
	
	
	
	//----------------------------------------------TASK 6-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 7-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 8-----------------------------------------
	
	



	//----------------------------------------------TASK 9-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 10-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 11-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 12-----------------------------------------
	
	
	
	
	
	//----------------------------------------------TASK 13-----------------------------------------







return 0;

}

