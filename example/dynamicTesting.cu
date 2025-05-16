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

    // Extracting the filename from program arguments

    if(argc < 4 || argc > 5){
        std::cerr << "Missing argument: filename" << std::endl;
        std::cerr << "Usage: <program name> <filename> <grid dimensions> <block dimensions> <kernel type>" << std::endl;
        std::cerr << "Exiting.." << std::endl;
        exit(1);
    };

    std::string kernelTypeStr;
    std::string filename = argv[1];
    int numBlocks = std::stoi(argv[2]);
    int threadsPerBlock = std::stoi(argv[3]);
    if(argc == 5){
        kernelTypeStr = argv[4];
    }

    auto getKernelType = [&kernelTypeStr](){
        if(kernelTypeStr == "less-atomic"){
            return Cudyn::Scheduler::KernelType::REDUCED_ATOMICS;
        } else return Cudyn::Scheduler::KernelType::STANDARD;
    };


    auto kernelType = getKernelType();
    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);

    if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL){

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<double>(csr_matrix.get_rows_count());

        Cudyn::CSR::Datastructures::DeviceDataSpMV<double> deviceData(csr_matrix, multiplicationVector);

        Cudyn::CSR::Kernel::CudynCSRSpMV<double> SpMVKernel(deviceData);

        Cudyn::Utils::GridConfiguration::KernelConfig config{.total_tasks = deviceData.csrData.rows, .grid_dimensions = (size_t)numBlocks, .block_dimensions = (size_t)threadsPerBlock};

        switch(kernelType){
                case Cudyn::Scheduler::KernelType::REDUCED_ATOMICS:{
                    Cudyn::Launcher::launch<Cudyn::Scheduler::ReducedAtomicScheduler>(config, SpMVKernel);
                    break;
                } default: {
                    Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(config, SpMVKernel);
                }
            
        }
        Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(config, SpMVKernel);

        Cudyn::Utils::errorCheck();
        
        std::cout << "Cuda Success!" << std::endl;
        std::cout << "Output vector dimensions: " << deviceData.getResult().size() << std::endl;


    } else if (fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER || fileDataType == MatrixMarketHeaderTypes::DataType::PATTERN){
        
        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<int>(csr_matrix.get_rows_count());

        Cudyn::CSR::Datastructures::DeviceDataSpMV<int> deviceData(csr_matrix, multiplicationVector);

        Cudyn::CSR::Kernel::CudynCSRSpMV<int> SpMVKernel(deviceData);

        Cudyn::Utils::GridConfiguration::KernelConfig config{.total_tasks = deviceData.csrData.rows, .grid_dimensions = (size_t)numBlocks, .block_dimensions = (size_t)threadsPerBlock};

        std::cout << "Starting dynamic kernel with " << (static_cast<double>(deviceData.csrData.rows) / numBlocks) / threadsPerBlock << std::endl;

        switch(kernelType){
            case Cudyn::Scheduler::KernelType::REDUCED_ATOMICS:{
                Cudyn::Launcher::launch<Cudyn::Scheduler::ReducedAtomicScheduler>(config, SpMVKernel);
                break;
            } default: {
                Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(config, SpMVKernel);
            }
        }

        Cudyn::Utils::errorCheck();
        
        std::cout << "Cuda Success!" << std::endl;
        std::cout << "Output vector dimensions: " << deviceData.getResult().size() << std::endl;
    }        

}