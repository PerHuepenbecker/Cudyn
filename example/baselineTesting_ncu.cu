#include "../include/cudyn/utils.cuh"
#include "../include//cudyn/CudynCSR.cuh"

#include "../include/matrix/Matrix.hpp"
#include "../include/matrix/CSR.hpp"
#include "../include//matrix_market_parser/MatrixMarketCSRParser.h"

#include <iostream>
#include <random>
#include <vector>
#include <limits>

template <typename T>
__global__ void SpMVKernel(
    const T* __restrict__ dataPointer ,
    const size_t* __restrict__ columnIndices ,
    const size_t* __restrict__ rowPointers ,
    const T* __restrict__ multiplicationVector, 
    T* __restrict__ resultVector,
    size_t numRows
){

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < numRows){
        T dotProduct = 0;
        size_t rowStartOffset = rowPointers[row];
        size_t rowEndOffset = rowPointers[row+1];

        for (size_t j = rowStartOffset; j < rowEndOffset; ++j) {
            dotProduct += dataPointer[j] * multiplicationVector[columnIndices[j]];
        }    
        
        resultVector[row] = dotProduct;
    }      
}    



template <typename T>
auto lauchSpMVKernel(Cudyn::CSR::Datastructures::DeviceDataSpMV<T>& deviceData, size_t numBlocks, size_t threadsPerBlock){
    auto data = deviceData.csrData.csrMatrixData_d.get();
    auto columnIndices = deviceData.csrData.csrMatrixColIndices_d.get();
    auto rowPointers = deviceData.csrData.csrMatrixRowPtrs_d.get();

    auto resultVector = deviceData.result.get();
    auto multiplicationVector = deviceData.multiplicationVector.get();

    auto rows = deviceData.csrData.rows;

    std::vector<float> measurements;

    if(numBlocks * threadsPerBlock >= rows){

       

            SpMVKernel<<<numBlocks, threadsPerBlock>>>(data, columnIndices, rowPointers, multiplicationVector, resultVector, rows);
            Cudyn::Utils::errorCheck();

    } else {
        std::cerr << "Warning: Number of threads are not enough" << std::endl;
    }    


    Cudyn::Utils::errorCheck();
}    

int main(int argc, char** argv){

    if(argc < 3){
        std::cerr << "Missing argument: filename" << std::endl;
        std::cerr << "Usage: <program name> <filename>" << std::endl;
        std::cerr << "Exiting.." << std::endl;
        exit(1);
    };

    

    std::string filename = argv[1];

    //int threadsPerBlock = std::stoi(argv[2]);

    std::vector<int> threadCountArguments;

    for(int i = 2; i < argc; ++i){
        int threadCountArg = std::stoi(argv[i]);

        if(threadCountArg <= 0 || threadCountArg > 1024){
        std::cerr << "Error: Bad number of threads per block: " << threadCountArg << " - exiting" << std::endl;
        exit(1);
    }
        threadCountArguments.push_back(threadCountArg);
    }

    
    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);


    if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL){

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<double>(csr_matrix.get_rows_count(), 1, true);

        for(const auto el: threadCountArguments){
            int numBlocks = (csr_matrix.get_rows_count() + el - 1) / el;

            std::cout << "\nLaunching with " << el << "Threads per Block\n"; 
            std::cout << "Launcing with " << numBlocks << " Blocks\n\n";

            Cudyn::CSR::Datastructures::DeviceDataSpMV<double> deviceData(csr_matrix, multiplicationVector);

            lauchSpMVKernel<double>(deviceData, numBlocks, el);

            Cudyn::Utils::errorCheck();
        
        }

    } else if (fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER || fileDataType == MatrixMarketHeaderTypes::DataType::PATTERN){
        
        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<int>(csr_matrix.get_rows_count());

         for(const auto el: threadCountArguments){
            int numBlocks = (csr_matrix.get_rows_count() + el - 1) / el;

            std::cout << "\nLaunching with " << el << "Threads per Block\n"; 
            std::cout << "Launcing with " << numBlocks << " Blocks\n\n";

            Cudyn::CSR::Datastructures::DeviceDataSpMV<int> deviceData(csr_matrix, multiplicationVector);

            lauchSpMVKernel<int>(deviceData, numBlocks, el);

            Cudyn::Utils::errorCheck();
        
            
        }
        
    }

}