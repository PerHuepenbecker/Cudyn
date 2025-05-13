#include "../include/cudyn/scheduler.cuh"
#include "../include/cudyn/utils.cuh"
#include "../include/cudyn/CudynCSR.cuh"

#include "../include/matrix/Matrix.hpp"
#include "../include/matrix/CSR.hpp"

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

template<typename T>
auto generateCSRProblem(int problem_size, float sparcity, std::mt19937& gen){
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

    std::vector<T> bare_values(problem_size*problem_size);

    for(size_t i = 0; i < problem_size*problem_size; ++i) {
        if(dist_sparcity(gen) > sparcity){
            bare_values[i] = dist_values(gen);
        } else {
            bare_values[i] = 0;
        }
    }

    // Create intermediate matrix
    Matrix<T> tmp_matrix(std::move(bare_values), problem_size);
    // Create CSR matrix
    CSRMatrix<T> csr_matrix(tmp_matrix);

    return csr_matrix;
}


int main(int argc, char** argv) {
    // argv[1] == Kernel type
    // argv[2] == problem size
    // argv[3] == sparcity
    // argv[4] == grid_dimensions
    // argv[5] == block_dimensions 

    if(argc < 5) {
        std::cerr << "[cudyn error] - Not enough arguments provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << "<kernel type> <problem_size> <sparcity> <grid_dimensions_single> <block_dimensions_single> <grid_dimensions_static_dual> <block_dimensions_static_dual>" << std::endl;
        return 1;
    }

    std::string kernelType = argv[1];

    size_t problemSize = std::stoul(argv[2]);
    float sparcity = std::stof(argv[3]);
    size_t gridSize = std::stoul(argv[4]);
    size_t blockDimensions = std::stoul(argv[5]);

    size_t gridSizeB = std::stoul(argv[4]);
    size_t blockDimensionsB = std::stoul(argv[5]);

    if(kernelType == "dual"){
            if(argc < 7){
                std::cout << "Assigning equal grid dimensions" << std::endl;
        } else {
            gridSizeB = std::stoul(argv[6]);
            blockDimensionsB = std::stoul(argv[7]);
        }
    }

    std::cout << "Starting with " << problemSize << " " << sparcity << " " << gridSize << " " << blockDimensions << "\n";

    auto seed = 42;
    std::mt19937 gen(seed);

    auto csrMatrix = generateCSRProblem<double>(problemSize, sparcity, gen);


    auto multVec = Cudyn::CSR::Helpers::generate_multiplication_vector<double>(problemSize);

    Cudyn::CSR::Datastructures::DeviceDataSpMV<double> deviceData (csrMatrix, multVec);

    auto kernelConfig = Cudyn::Utils::GridConfiguration::KernelConfig{.total_tasks = problemSize, .grid_dimensions = gridSize, .block_dimensions = blockDimensions};

    if(kernelType == "dynamic" || kernelType == "dual") {
        std::cout << "Launching dynamic Kernel" << std::endl;
        Cudyn::CSR::Kernel::CudynCSRSpMV<double> dynamicKernelLogic (deviceData);

        Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(kernelConfig, dynamicKernelLogic);
        Cudyn::Utils::errorCheck();

        auto result = deviceData.getResult();
        std::cout << "Dynamic success" << std::endl;
        size_t zeroCount = 0;
        for(const auto el: result){
            if(el == 0){
                zeroCount++;
            }
        }
        std::cout<< "Contains " << zeroCount << " zeros" << std::endl;

    } 
    
    if(kernelType =="static" || kernelType == "dual") {
        std::cout << "Launching static Kernel" << std::endl;

        lauchSpMVKernel<double>(deviceData, gridSizeB, blockDimensionsB);

        Cudyn::Utils::errorCheck();

        auto result = deviceData.getResult();
        std::cout << "Static success" << std::endl;
        size_t zeroCount = 0;
        for(const auto el: result){
            if(el == 0){
                zeroCount++;
            }
        }
        std::cout<< "Contains " << zeroCount << " zeros" << std::endl;
    }

}