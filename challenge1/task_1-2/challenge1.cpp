#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "../resources/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../resources/stb_image_write.h"
#include "../resources/team39_helper.h"
#include <lis.h>

using namespace Eigen;
using namespace std;

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
  
  // get A1 using convolution operation
  Matrix3d Hav1;
  Hav1 << 1.0/12.0,  1.0/12.0,  1.0/12.0,
      1.0/12.0,  1.0/3.0,  1.0/12.0,
      1.0/12.0,  1.0/12.0,  1.0/12.0; 
  SparseMatrix<double> A1;

  A1 = matrix_convolution(Hav1,height,width);

  // print A1 non zero entries
  cout << "A1 number of non zero entries: " << A1.nonZeros() << std::endl;
  
  
  // -------------------------------------- TASK5 --------------------------------------
  
  // apply smoothering filter
    VectorXd smoothing = A1*w;

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothing_bytes(height,width);
    for(int i=0; i < height; i++){
      for(int j=0; j < width; j++) {
        double val = smoothing[i * width + j];
        
        val = std::min(255.0, std::max(0.0, val));
        smoothing_bytes(i, j) = static_cast<unsigned char>(std::round(val));
      }
    }
  
  //load_image
  stbi_write_png("../output/deer_smoothing.png",width,height,1,smoothing_bytes.data(),width);

  // -------------------------------------- TASK6 --------------------------------------
  
  // get A2 using convolution operation
  Matrix3d Hsh1;
  Hsh1 <<  0.0, -3.0,  0.0,
          -1.0,  9.0, -3.0,
           0.0, -1.0,  0.0;

  SparseMatrix<double> A2 = matrix_convolution(Hsh1, height, width);
  cout << "A2 number of non zero entries: " << A2.nonZeros() << std::endl;
  std::cout << "Is the matrix A2 symmetric? " << (isSymmetric(A2, 1e-12) ? "Yes" : "No") << std::endl;
  
  
  // -------------------------------------- TASK7 --------------------------------------
  
  // apply sharpening filter to the original image
  VectorXd sharpening = A2 * v;

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpening_bytes(height, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double val = sharpening[i * width + j];
      val = std::min(255.0, std::max(0.0, val));
      sharpening_bytes(i, j) = static_cast<unsigned char>(std::round(val));
    }
  }
  stbi_write_png("../output/deer_sharpening.png", width, height, 1, sharpening_bytes.data(), width);
  
  
  // -------------------------------------- TASK8 --------------------------------------
  
  saveMarketVector(w, "../output/deer_sharpening.mtx");
  saveMarket(A2, "../output/A2.mtx");

  LIS_MATRIX A2_lis;
  LIS_VECTOR w_lis;
  LIS_VECTOR x_lis;
  LIS_SOLVER solver;

  LIS_REAL residual;
  LIS_INT iter;
  //initialize lis
  lis_initialize(&argc, &argv);
  // create lis matrix 
  lis_matrix_create(LIS_COMM_WORLD, &A2_lis);
  lis_matrix_set_size(A2_lis, 0, A2.cols());
  lis_matrix_set_type(A2_lis, LIS_MATRIX_CSR);
  
  // set values to lis matrix
  for (int k = 0; k < A2.outerSize(); ++k) {
      for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it) {
          lis_matrix_set_value( LIS_INS_VALUE, it.row(), it.col(), it.value(), A2_lis);
      }
  }
  lis_matrix_assemble(A2_lis);

 // create lis vector
  lis_vector_create(LIS_COMM_WORLD, &w_lis);
  lis_vector_set_size(w_lis, 0, w.size());
   // set values to lis vector
  for (int i = 0; i < w.size(); ++i) {
      lis_vector_set_value( LIS_INS_VALUE,i, w[i],w_lis);
  }

  // create solver
  lis_solver_create(&solver);
  // Use BiCGSTAB as the iterative solver (good for non-symmetric & sparse matrices)
  // Use ILU as the preconditioner
  // Use ILU(0), so no additional fill-in is allowed
  // Set the convergence tolerance to 1e-12

  // Use ILU (Incomplete LU) as a preconditioner.
  // ILU approximates the LU factorization while preserving sparsity.

  // ILU(0) does not allow additional fill-in.
  // It keeps the same sparsity pattern as the original matrix.

  char options[] = "-i bicgstab -p ilu -ilu_fill 0 -tol 1e-12";
  lis_solver_set_option(options, solver);

  // create x vector  with the same size as w
  lis_vector_duplicate(w_lis, &x_lis);

  // Solve the linear system A2 * x = w using the BiCGSTAB iterative solver with ILU preconditioning
  // The solution will be stored in x_lis 
  // error and iteration count can be retrieved from the solver after the solve operation
  LIS_INT status =lis_solve(A2_lis, w_lis, x_lis, solver);
  if (status != LIS_SUCCESS) {
      std::cerr << "Error: lis_solve failed with status " << status << std::endl;
  }

  // Get the number of iterations and the final residual
  lis_solver_get_iter(solver, &iter);
  lis_solver_get_residualnorm(solver, &residual); 

  cout << "BiCGSTAB Iteration count = " << iter << ", final residual = " << residual << std::endl;


  //parse the solution from lis vector to Eigen vector
  VectorXd mat_uploaded(w.size());
  for (int i = 0; i < w.size(); ++i) {
      LIS_SCALAR value;
      lis_vector_get_value(x_lis, i, &value);
      mat_uploaded[i] = static_cast<double>(value);
  }
 // ------- CLEANUP: Free the allocated resources for the solver, matrix, and vectors
  lis_solver_destroy(solver);
  lis_matrix_destroy(A2_lis);
  lis_vector_destroy(w_lis);
  lis_vector_destroy(x_lis);  

  lis_finalize();
  
  
  
  // -------------------------------------- TASK9 --------------------------------------

   Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sol(height, width);

   for (int i = 0; i < height; ++i) {
	   int row = i * width;
	   for (int j = 0; j < width; ++j) {
		   double val = mat_uploaded(row + j);
		   double mat_val = std::max(0.0, std::min(255.0, val));
		   sol(i, j) = static_cast<unsigned char>(std::round(mat_val));
	   }
   }

   stbi_write_png("../output/deer_taks_9.png", width, height, 1, sol.data(), width);

  
  // -------------------------------------- TASK10 --------------------------------------
  
  Matrix3d H3;
  H3 << -1.0,  0.0,  1.0,
      -2.0,  0.0,  2.0,
      -1.0,  0.0,  1.0; 

  SparseMatrix<double> A3 = matrix_convolution(H3, height, width);
  std::cout << "Is the matrix A3 symmetric? " << (isSymmetric(A3, 1e-12) ? "Yes" : "No") << std::endl;

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
  stbi_write_png("../output/deer_sobel.png", width, height, 1, matrix_3_bytes.data(),width);

  
  // -------------------------------------- TASK12 --------------------------------------
  
   SparseMatrix<double> I_matrix(A3.rows(), A3.cols());
  I_matrix.setIdentity();

  SparseMatrix<double> AA = 4.0 * I_matrix + A3;
  AA.makeCompressed();

  double tol = 1.0e-10;
  int max_iterations = 500;

  BiCGSTAB<SparseMatrix<double>> bicgstab;
  bicgstab.setMaxIterations(max_iterations);
  bicgstab.setTolerance(tol);
  bicgstab.compute(AA);

  // controllo
  if (bicgstab.info() != Success) {
	  std::cout << "BiCGSTAB failed!!!" << std::endl;
	  return 1;
  }

  VectorXd y_solution_12 = bicgstab.solve(w);

  std::cout <<"Iteration count task 12 = " << bicgstab.iterations() <<" and the final residual is = " << bicgstab.error() << std::endl;

  if (bicgstab.info() == NoConvergence) {
    std::cout << "BiCGSTAB doesn't reach the convergence in " << max_iterations << " iterations!" << std::endl;
  }


  
  // -------------------------------------- TASK13 --------------------------------------
  
 Matrix<unsigned char, Dynamic, Dynamic, RowMajor> y_matrix(height, width);
  for (int i = 0; i < height; i++) {
	  int row = i * width;
	  for (int j = 0; j < width; j++){
		  double val = y_solution_12[row+j];
		  double y_matrix_val = std::max(0.0, std::min(255.0, val));
		  y_matrix(i,j) = static_cast<unsigned char>(std::round(y_matrix_val));
	  }
  }
  stbi_write_png("../output/deer_task_13.png", width, height, 1, y_matrix.data(), width);


  return 0;
}

// COMMAND TO COMPILE 

/* mpicxx -DUSE_MPI \
    -I${mkEigenInc} \
    -I${mkLisInc} \
    challenge1.cpp \
    -L${mkLisLib} \
    -llis \
    -o challenge1 */

// COMMAND TO RUN
/*

mpirun -np 1 ./challenge1 ../resources/deer.jpg

*/
