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
std::vector<T> generate_multiplication_vector(size_t problem_size, T baseValue = 1, bool random = false, int seed = 42, T lower_bound = 0, T upper_bound = 2 << 20){
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

template <typename T>
struct DeviceDataCSR {
    T* csr_matrix_data_d = nullptr;
    size_t* csr_matrix_row_ptrs_d = nullptr;
    size_t* csr_matrix_col_indices_d = nullptr;
    size_t rows = 0;
    size_t columns = 0;

    DeviceDataCSR(const CSRMatrix<T>& csrMatrix);
};

template <typename T>
DeviceDataCSR<T>::DeviceDataCSR(const CSRMatrix<T>& csrMatrix){
    auto data = csrMatrix.get_data();
    auto column_pointers = csrMatrix.get_col_indices();
    auto row_pointers = csrMatrix.get_row_ptrs();

    cudaMalloc(&csr_matrix_data_d, data.size() * sizeof(T));
    cudaMalloc(&csr_matrix_col_indices_d, column_pointers.size() * sizeof(size_t));
    cudaMalloc(&csr_matrix_row_ptrs_d, row_pointers.size() * sizeof(size_t));
    
}

template<typename T>
struct DeviceDataSpMV{
    DeviceDataCSR<T> csrData;
    T* multiplicationVector = nullptr;
    T* result = nullptr;
};

template<typename T>
void allocateDeviceMemory(DeviceDataSpMV<T> deviceData){

}

template<typename T>
void prepareCSRDeviceArrays(){}
    


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



    if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL){

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = generate_multiplication_vector<double>(csr_matrix.get_rows_count());






    } else if (fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER || fileDataType == MatrixMarketHeaderTypes::DataType::PATTERN){
        
        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = generate_multiplication_vector<int>(csr_matrix.get_rows_count());


        }
    
    



}