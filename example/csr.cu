#include "../include/cudyn/scheduler.cuh"
#include "../include/cudyn/utils.cuh"

#include "../include/matrix/Matrix.hpp"
#include "../include/matrix/CSR.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <limits>

auto generate_csr_problem(int problem_size, float sparcity){
    if(problem_size <= 0 || sparcity <= 0 || sparcity > 1) {
        std::cerr << "[cudyn bad argument] - The problem size and sparcity must be greater than 0 and sparcity must be less than or equal to 1" << std::endl;
        exit(1);
    }

    // setup of the random number generator with fixed seed for testing accuracy
    auto general_seed = 42;
    std::mt19937 gen(general_seed);
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



int main(){
    std::cout << "Starting the example" << std::endl;
    // Confined sparse row example
    int problem_size = 10;
    float sparcity = 0.1;
    auto csr_matrix = generate_csr_problem(problem_size, sparcity);
    std::cout << "Generated CSR matrix with size: " << problem_size << " and sparcity: " << sparcity << std::endl;
    csr_matrix.display_full();
}