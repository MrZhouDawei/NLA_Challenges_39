#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "../resources/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../resources/stb_image_write.h"
#include "../resources/team39_helper.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  const char* input_image_path = argv[1];

  // Load the GREY image
  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path << std::endl;
    return 1;
  }

  // ------------------------------- TASK1 ---------------------------------------
  std::cout << "Image size: " << height << " x " << width << std::endl;


  

  // ORIGINAL matrix
  MatrixXd original(height, width);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      original(i,j) = static_cast<double>(image_data[i* width + j]);
    }
  }
  stbi_image_free(image_data);

  // ------------------------------- TASK2 -----------------------------------------

  // NOISY matix
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-50.0, 50.0);

  MatrixXd noisy(height, width);
  for (int i = 0; i < height; i++){
	  for (int j = 0; j < width; j++) {
		  double val = original(i,j) + dist(gen);
		  noisy (i,j) = std::min(std::max(val, 0.0), 255.0); 
	  }
  }

  // export the NOISY matrix
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_bytes(height, width);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      noisy_bytes(i, j) = static_cast<unsigned char>(std::round(noisy(i, j)));
    }
  }
  stbi_write_png("../output/deer_noisy.png", width, height, 1, noisy_bytes.data(), width);


  // ---------------------------------- TASK3 --------------------------------------
  
  // create v from the original image
  const long size_v = original.cols() * original.rows();
  VectorXd v = original.reshaped<RowMajor>();

  // create w form the noisy image
  const long size_w = noisy.cols() * noisy.rows();
  VectorXd w = noisy.reshaped<RowMajor>();

  // control the size of the vectors
  if (v.size() != size_v || w.size() != size_w) {
        std::cout << "Convertion matrix - vector failed." << std::endl;
	std::cout << "Vector v size: " << v.size() << ", expected size: " << size_v << std::endl;
	std::cout << "Vector w size: " << w.size() << ", expected size: " << size_w << std::endl;
	return 1;
    }
  
  // compute the euclidean norm
  double norm_v = getArrayEuclideanNorm(v);
  std::cout << "Euclidean norm of v: " << norm_v << std::endl;
  // double norm_w = getArrayEuclideanNorm(w);



  // -------------------------------------- TASK4 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK5 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK6 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK7 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK8 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK9 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK10 --------------------------------------
  
  Matrix3d H3;
  H3 << -1.0,  0.0,  1.0,
      -2.0,  0.0,  2.0,
      -1.0,  0.0,  1.0; 

  SparseMatrix<double> A3 = matrix_convolution(H3, height, width);
  std::cout << "Is the matrix A3 symmetric? " << isSymmetric(A3, 1e-12) << std::endl;
  

  // -------------------------------------- TASK11 --------------------------------------
  
  VectorXd matrix_3_product = A3 * v;

  MatrixXd matrix_3(height, width);
  for (int i = 0; i<height; i++){
	  int row = i * width;
	  for (int j = 0; j < width; j++){
		  double val = matrix_3_product[row+j];
		  matrix_3(i,j) = std::min(255.0,std::max(0.0,val));
	  }
  }

  //save image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> matrix_3_bytes(height, width);
  for (int i = 0; i < height; i++) {
	  for (int j = 0; j < width; j++) {
		  matrix_3_bytes(i,j) = static_cast<unsigned char>(std::round(matrix_3(i,j)));
	  }
  }
  stbi_write_png("../output/deer_filtered_3.png", width, height, 1, matrix_3_bytes.data(),width);

  
  // -------------------------------------- TASK12 --------------------------------------
  
  SparseMatrix<double> I_matrix(A3.rows(), A3.cols());
  I_matrix.setIdentity();

  double tol = 1.0e-10;

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DA finire
  
  // -------------------------------------- TASK13 --------------------------------------
  
  /*
  
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> y_matrix(height, width);
  for (int i = 0; i < height; i++) {
	  int row = i * width;
	  for (int j = 0; j < width; j++){
		  double val = y_sol[row+j];
		  double y_matrix_val = std::max(0.0, std::min(255.0, val));
		  y_matrix(i,j) = static_cast<unsigned char>(std::round(y_matrix_val));
	  }
  }
  stbi_write_png("../output/deer_filtered_3_bis.png", width, height, 1, y_matrix.data(), width);

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> y_matrix(height, width);
  for (int i = 0; i < height; i++) {
	  int row = i * width;
	  for (int j = 0; j < width; j++){
		  double val = y_sol[row+j];
		  double y_matrix_val = std::max(0.0, std::min(255.0, val));
		  y_matrix(i,j) = static_cast<unsigned char>(std::round(y_matrix_val));
	  }
  }
  stbi_write_png("../output/deer_filtered_3_bis.png", width, height, 1, y_matrix.data(), width);


  */



  return 0;
}


