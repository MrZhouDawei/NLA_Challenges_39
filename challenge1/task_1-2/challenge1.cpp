#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "../resources/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../resources/stb_image_write.h"

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
  std::mt19937 gen;
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
  
  // VectorXd v = 








  return 0;
}










