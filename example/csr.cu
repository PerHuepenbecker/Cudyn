#include "../include/cudyn/scheduler.cuh"
#include "../include/cudyn/utils.cuh"

#include "../include/matrix/Matrix.hpp"
#include "../include/matrix/CSR.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <limits>

auto generate_csr_problem(int problem_size, float sparcity, std::mt19937& gen){
    if(problem_size <= 0 || sparcity <= 0 || sparcity > 1) {
        std::cerr << "[cudyn bad argument] - The problem size and sparcity must be greater than 0 and sparcity must be less than or equal to 1" << std::endl;
        exit(1);
    }

    // setup of the random number generator with fixed seed for testing accuracy
    
    std::uniform_real_distribution<> dist_sparcity(0.0,1.0);
    std::uniform_int_distribution<> dist_values(
        0,
        10

        //std::numeric_limits<double>::min(),
        //std::numeric_limits<double>::max()
    );

    std::vector<double> bare_values(problem_size*problem_size);

    for(size_t i = 0; i < problem_size*problem_size; ++i) {
        if(dist_sparcity(gen) > sparcity){
            bare_values[i] = dist_values(gen);
        } else {
            bare_values[i] = 0;
        }
    }

    // Create intermediate matrix
    Matrix<double> tmp_matrix(std::move(bare_values), problem_size);
    // Create CSR matrix
    CSRMatrix<double> csr_matrix(tmp_matrix);

    return csr_matrix;
}

auto generate_multiplication_vector(const CSRMatrix<double>& csr_matrix, std::mt19937& gen){
    auto problem_size = csr_matrix.get_cols_count();
    std::vector<double> bare_values(problem_size);

    std::uniform_real_distribution<> dist_values(
        0,
        10
    );

    for(size_t i = 0; i < problem_size; ++i){
        bare_values[i] = dist_values(gen);
    }

    return bare_values;
}

int main(int argc, char** argv) {
    // argv[1] == problem size
    // argv[2] == sparcity
    // argv[3] == grid_dimensions
    // argv[4] == block_dimensions 
    if(argc < 5) {
        std::cerr << "[cudyn error] - Not enough arguments provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <problem_size> <sparcity> <grid_dimensions> <block_dimensions>" << std::endl;
        return 1;
    }

    int problem_size = std::stoi(argv[1]);
    float sparcity = std::stof(argv[2]);
    int grid_size = std::stoi(argv[3]);
    int block_dimensions = std::stoi(argv[4]);
    if(problem_size <= 0 || sparcity <= 0 || sparcity > 1) {
        std::cerr << "[cudyn error] - The problem size and sparcity must be greater than 0 and sparcity must be less than or equal to 1" << std::endl;
        return 1;
    }

    std::cout << "Starting the example\n" ;
    std::cout << "--------------------\n" ;
    std::cout << "Problem size: " << problem_size << "\n";
    std::cout << "Sparcity: " << sparcity << "\n";
    std::cout << "Grid size: " << grid_size << " blocks\n";
    std::cout << "Threads per block: " << block_dimensions << "\n";


    // Setup the random number generator with fixed seed for testing accuracy
    auto general_seed = 42;
    std::mt19937 gen(general_seed);

    // Confined sparse row example
    //int problem_size = 1024;
    //float sparcity = 0.7;
    auto csr_matrix = generate_csr_problem(problem_size, sparcity, gen);
    std::cout << "Generated CSR matrix with size: " << problem_size << " and sparcity: " << sparcity << std::endl;
    //csr_matrix.display_full();

    // Generate the multiplication vector

    auto multiplication_vector = generate_multiplication_vector(csr_matrix, gen);

    // getting the csr matrix inner data
    auto csr_matrix_data = csr_matrix.get_data();
    auto csr_matrix_row_ptrs = csr_matrix.get_row_ptrs();
    auto csr_matrix_col_indices = csr_matrix.get_col_indices();

    double* csr_matrix_data_d = nullptr;
    size_t* csr_matrix_row_ptrs_d = nullptr;
    size_t* csr_matrix_col_indices_d = nullptr;

    // allocating the csr matrix data on device
    cudaMalloc(&csr_matrix_data_d, csr_matrix_data.size() * sizeof(double));
    cudaMalloc(&csr_matrix_row_ptrs_d, csr_matrix_row_ptrs.size() * sizeof(size_t));
    cudaMalloc(&csr_matrix_col_indices_d, csr_matrix_col_indices.size() * sizeof(size_t));

    // allocating the multiplication vector and result vector
    std::vector<double> result(problem_size, 0.0);
    double* result_d = nullptr;
    double* multiplication_vector_d = nullptr;
    cudaMalloc(&multiplication_vector_d, problem_size * sizeof(double));
    cudaMalloc(&result_d, problem_size * sizeof(double));

    // zeroing the device result vector
    cudaMemset(result_d, 0, problem_size * sizeof(double));

    
    if (cudaGetLastError() != cudaSuccess){
        std::cerr << "[cudyn error] - failed to allocate enough memory on device" << std::endl;
        exit(1);
    }
    // Copying the data to device memory
    cudaMemcpy(multiplication_vector_d, multiplication_vector.data(), problem_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_matrix_data_d, csr_matrix_data.data(), csr_matrix_data.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_matrix_row_ptrs_d, csr_matrix_row_ptrs.data(), csr_matrix_row_ptrs.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_matrix_col_indices_d, csr_matrix_col_indices.data(), csr_matrix_col_indices.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) {
        std::cerr << "[cudyn error] - failed to copy data to device" << std::endl;
        exit(1);
    }

    // lambda kernel definition
    auto csr_multiplication_kernel = [csr_matrix_data_d,
                                      csr_matrix_col_indices_d, 
                                      csr_matrix_row_ptrs_d,
                                      multiplication_vector_d,
                                      result_d, problem_size] __device__ (size_t i) {
                                        //if(i >= problem_size) return;

                                        // TODO: Write logic to handle the multiplication
                                        auto start = csr_matrix_row_ptrs_d[i];
                                        auto end = csr_matrix_row_ptrs_d[i+1];

                                        for(size_t j = start; j < end; ++j){
                                            auto col_index = csr_matrix_col_indices_d[j];
                                            auto value = csr_matrix_data_d[j];
                                            result_d[i] += value * multiplication_vector_d[col_index];
                                        }
                                    };

    // defining the test probem size

    //auto grid_size = 2;
    
    // launching the kernel

    cudyn::scheduler::generic_irregular_kernel<<<grid_size, block_dimensions>>>(problem_size, grid_size, csr_multiplication_kernel);

    if(cudaGetLastError() != cudaSuccess) {
        std::cerr << "[cudyn error] - Failed to launch the kernel\n";
        exit(1); 
    }

    // copying the result back to host memory
    cudaMemcpy(result.data(), result_d, problem_size*sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaGetLastError() != cudaSuccess) {
        std::cerr << "[cudyn error] - Failed to copy the result back to host memory\n";
        exit(1);
    }

    cudaFree(csr_matrix_data_d);
    cudaFree(csr_matrix_row_ptrs_d);
    cudaFree(csr_matrix_col_indices_d);
    cudaFree(multiplication_vector_d);
    cudaFree(result_d);
    

    std::cout << std::endl;
    std::cout << "Control run\n";

    auto res = csr_matrix * multiplication_vector;
    auto counter = 0;
    for(size_t i = 0; i < problem_size; ++i){
        if (res[i] - result[i] > 0.0001){
            std::cout << "Result " << i << " differs: " << res[i] << " != " << result[i] << std::endl;
            counter++;
        }
    }

    std::cout << counter << " results differ\n";

    std::cout << std::endl;

}