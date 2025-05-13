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

    if(numBlocks * threadsPerBlock >= rows){

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        SpMVKernel<<<numBlocks, threadsPerBlock>>>(data, columnIndices, rowPointers, multiplicationVector, resultVector, rows);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Kernel execution time: " << milliseconds << "ms" << std::endl;

    } else {
        std::cerr << "Warning: Number of threads are not enough" << std::endl;
    }


    Cudyn::Utils::errorCheck();
}


int main(int argc, char** argv){

    if(argc != 4){
        std::cerr << "Missing argument: filename" << std::endl;
        std::cerr << "Usage: <program name> <filename>" << std::endl;
        std::cerr << "Exiting.." << std::endl;
        exit(1);
    };

    std::string filename = argv[1];

    int numBlocks = std::stoi(argv[2]);
    int threadsPerBlock = std::stoi(argv[3]);

    if(numBlocks <= 0) {
        std::cerr << "Error: Bad number of blocks: " << numBlocks << " - exiting" << std::endl;
    }
    if(threadsPerBlock <= 0 || threadsPerBlock > 2048){
        std::cerr << "Error: Bad number of threads per block: " << threadsPerBlock << " - exiting" << std::endl;
    }


    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);


    if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL){

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<double>(csr_matrix.get_rows_count(), 1, true);

        Cudyn::CSR::Datastructures::DeviceDataSpMV<double> deviceData(csr_matrix, multiplicationVector);

        lauchSpMVKernel<double>(deviceData, numBlocks, threadsPerBlock);

        Cudyn::Utils::errorCheck();
        
        std::cout << "Cuda Success!" << std::endl;

        auto result = deviceData.getResult();
        size_t zeroCount = 0;
        for(const auto& el :result){
            if (el==0){
                zeroCount++;
            } 
        }

        std::cout << "Output vector dimensions: " << result.size() << std::endl;        
        std::cout << "Contains " << zeroCount << " zeros" << std::endl;

    } else if (fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER || fileDataType == MatrixMarketHeaderTypes::DataType::PATTERN){
        
        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<int>(csr_matrix.get_rows_count());

        Cudyn::CSR::Datastructures::DeviceDataSpMV<int> deviceData(csr_matrix, multiplicationVector);
        lauchSpMVKernel<int>(deviceData, numBlocks, threadsPerBlock);

        Cudyn::Utils::errorCheck();

        std::cout << "Cuda Success!" << std::endl;
        auto result = deviceData.getResult();
        size_t zeroCount = 0;
        for(const auto& el :result){
            if (el==0){
                zeroCount++;
            } 
        }

        std::cout << "Output vector dimensions: " << result.size() << std::endl;        
        std::cout << "Contains " << zeroCount << " zeros" << std::endl;
    }

}