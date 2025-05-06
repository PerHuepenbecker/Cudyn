#include "../include/cudyn/scheduler.cuh"
#include "../include/cudyn/utils.cuh"

#include "../include/matrix/Matrix.hpp"
#include "../include/matrix/CSR.hpp"
#include "../include//matrix_market_parser/MatrixMarketCSRParser.h"

#include <iostream>
#include <random>
#include <vector>
#include <limits>


template <typename T>
std::vector<T> generate_basic_multiplication_vector(size_t problem_size, T baseValue = 1, bool random = false, int seed = 42, T lower_bound = 0, T upper_bound = 2 << 20){
    std::vector<T> result (problem_size);
    if(!random){
        std::fill(result.begin(), result.end(), baseValue);
    } else {
        static_assert(std::is_same_v<T, int> || std::is_same_v<T, double>, "Only int and double types are supported for random generation");

        std::mt19937 gen(seed);
        if constexpr (std::is_same_v<T,int>){
            std::uniform_int_distribution<> dist(lower_bound, upper_bound);
            for(auto& el: result){
                el = dist(gen);
            }
        } else {
            std::uniform_real_distribution<> dist (lower_bound, upper_bound);
            for(auto& el:result){
                el = dist(gen);
            }
        } 
    }

    return result;
}
    


int main(int argc, char** argv) {
    if(argc != 2){
        std::cerr << "Missing argument: filename" << std::endl;
        std::cerr << "Usage: <program name> <filename>" << std::endl;
        std::cerr << "Exiting.." << std::endl;
        exit(1);
    }    

    // Extracting the filename from program arguments
    std::string filename = argv[1];

    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);

    auto vec = generate_basic_multiplication_vector<int>(10)



}