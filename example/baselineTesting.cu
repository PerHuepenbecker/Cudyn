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

template<typename T>
double average(const T& vec){
    double sum = 0;
    
    for(const auto& val: vec){
        sum+= val;
    }

    return vec.empty() ? 0.0 : sum/vec.size();
}

template<typename T>
double standardDeviation(const T& vec) {
    double avg = average(vec);

    double variance = 0;
    for (const auto& el : vec) {
        double diff = el - avg;
        variance += diff * diff;
    }

    return vec.empty() ? 0.0 : std::sqrt(variance / vec.size()); 
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

        for(int i = 0; i < 10; i++){

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            SpMVKernel<<<numBlocks, threadsPerBlock>>>(data, columnIndices, rowPointers, multiplicationVector, resultVector, rows);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            measurements.push_back(milliseconds);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaDeviceSynchronize();
        }    

        std::cout << "\nRuntime profiling results: " << std::endl;
        std::cout << "Average runtime " << average(measurements) << std::endl;
        std::cout << "Standard deviation " << standardDeviation(measurements) << std::endl;
        std::cout << "Minimum runtime " << *std::min_element(measurements.begin(), measurements.end()) << std::endl;
        std::cout << "Maximum runtime " << *std::max_element(measurements.begin(), measurements.end()) << std::endl;
        std::cout << "Number of measurements " << measurements.size() << std::endl;



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