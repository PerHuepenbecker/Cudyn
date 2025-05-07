#include "../include/cudyn/scheduler.cuh"
#include "../include/cudyn/utils.cuh"

#include "../include/matrix/Matrix.hpp"
#include "../include/matrix/CSR.hpp"
#include "../include//matrix_market_parser/MatrixMarketCSRParser.h"

#include "../include/cudyn/CudynCSR.cuh"

#include <iostream>
#include <random>
#include <vector>
#include <limits>

    
int main(int argc, char** argv) {
    if(argc != 2){
        std::cerr << "Missing argument: filename" << std::endl;
        std::cerr << "Usage: <program name> <filename>" << std::endl;
        std::cerr << "Exiting.." << std::endl;
        exit(1);
    };

    // Extracting the filename from program arguments
    std::string filename = argv[1];

    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);

    if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL){

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = CudynCSR::Helpers::generate_multiplication_vector<double>(csr_matrix.get_rows_count());

        CudynCSR::Datastructures::DeviceDataSpMV<double> deviceData(csr_matrix, multiplicationVector);

    } else if (fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER || fileDataType == MatrixMarketHeaderTypes::DataType::PATTERN){
        
        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = CudynCSR::Helpers::generate_multiplication_vector<int>(csr_matrix.get_rows_count());

        CudynCSR::Datastructures::DeviceDataSpMV<int> deviceData(csr_matrix, multiplicationVector);
    }        

}