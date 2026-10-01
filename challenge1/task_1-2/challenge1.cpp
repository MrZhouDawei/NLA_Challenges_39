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
  const long size_v = original.col() * original.col();
  VectorXd v(size_v);
  for (int i = 0; i < size_v; i++){
	  v(i) = static_cast<double>(original[i]);
  }

  // create w form the noisy image
  const long sze_w = noisy.col() * noisy.row();
  VectorXd w(size_w);
  for (int i = 0; i < size_w; i++){
	  w(i) = static_cast<double>(noisy[i]);
  }

  // control the size of the vectors
  if (v.size() != size_v || w.size() != size_w) {
        std::cout << "Convertion matrix - vector failed." << endl;
	std::cout << "Vector v size: " << v.size() << ", expected size: " << size_v << endl;
	std::cout << "Vector w size: " << w.size() << ", expected size: " << size_w << endl;
    }
  
  // compute the euclidean norm
  double norm_v = getArrayEuclideanNorm(v);
  // double norm_w = getArrayEuclideanNorm(w);



  // -------------------------------------- TASK4 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK5 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK6 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK7 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK8 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK9 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK10 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK11 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK12 --------------------------------------
  
  
  
  
  // -------------------------------------- TASK13 --------------------------------------
  






  return 0;
}










