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
std::vector<T> generate_multiplication_vector(size_t problem_size, double sparcity){

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


}