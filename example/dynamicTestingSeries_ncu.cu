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

    if(argc < 2 || argc > 3){
        std::cerr << "Missing argument: filename" << std::endl;
        std::cerr << "Usage: <program name> <filename> <grid dimensions> <block dimensions> <kernel type>" << std::endl;
        std::cerr << "Exiting.." << std::endl;
        exit(1);
    };

    std::string kernelTypeStr;
    std::string filename = argv[1];
    if(argc == 3){
        kernelTypeStr = argv[2];
    }

    std::vector<size_t> threadsPerBlockArgs{32,64,128,256,384,512,768,1024};
    std::vector<size_t> tasksPerThreadArgs{2,4,8,16,32};



    auto getKernelType = [&kernelTypeStr](){
        if(kernelTypeStr == "--reduced-atomic"){
            return Cudyn::Scheduler::KernelType::REDUCED_ATOMICS;
        } else return Cudyn::Scheduler::KernelType::STANDARD;
    };


    auto kernelType = getKernelType();
    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);

    if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL){

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<double>(csr_matrix.get_rows_count(), 1, true);

        Cudyn::CSR::Datastructures::DeviceDataSpMV<double> deviceData(csr_matrix, multiplicationVector);

        Cudyn::CSR::Kernel::CudynCSRSpMV<double> SpMVKernel(deviceData);

        int total_tasks = deviceData.csrData.rows;

        for(const auto tasksPerThread: tasksPerThreadArgs){


            std::cout << "Testing " << tasksPerThread<< "Tasks per Thread" << std::endl;
            std::cout << std::endl;  

            for(const auto threadsPerBlock: threadsPerBlockArgs){

                std::cout << "Testing " << threadsPerBlock << "ThreadsPerBlock" << std::endl;
                std::cout << std::endl;  
                
                int total_threads = (total_tasks + tasksPerThread - 1) / tasksPerThread;
                int numBlocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
        
                Cudyn::Utils::GridConfiguration::KernelConfig config{.total_tasks = deviceData.csrData.rows, .grid_dimensions = (size_t)numBlocks, .block_dimensions = (size_t)threadsPerBlock};
        
                switch(kernelType){
                        case Cudyn::Scheduler::KernelType::REDUCED_ATOMICS:{
                            Cudyn::Launcher::launch<Cudyn::Scheduler::ReducedAtomicScheduler>(config, SpMVKernel);
                            break;
                        } default: {
                            Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(config, SpMVKernel);                            
                        }
                    
                }
        
                Cudyn::Utils::errorCheck();
            }
        }



    } else if (fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER || fileDataType == MatrixMarketHeaderTypes::DataType::PATTERN){
        
        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();
        auto multiplicationVector = Cudyn::CSR::Helpers::generate_multiplication_vector<int>(csr_matrix.get_rows_count());

        Cudyn::CSR::Datastructures::DeviceDataSpMV<int> deviceData(csr_matrix, multiplicationVector);

        Cudyn::CSR::Kernel::CudynCSRSpMV<int> SpMVKernel(deviceData);

        int total_tasks = deviceData.csrData.rows;

            for(const auto tasksPerThread: tasksPerThreadArgs){

            std::cout << "Testing " << tasksPerThread<< "Tasks per Thread" << std::endl;
            std::cout << std::endl;  

            for(const auto threadsPerBlock: threadsPerBlockArgs){

                std::cout << "Testing " << threadsPerBlock << "ThreadsPerBlock" << std::endl;
                std::cout << std::endl;  
                
                int total_threads = (total_tasks + tasksPerThread - 1) / tasksPerThread;
                int numBlocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
        
                Cudyn::Utils::GridConfiguration::KernelConfig config{.total_tasks = deviceData.csrData.rows, .grid_dimensions = (size_t)numBlocks, .block_dimensions = (size_t)threadsPerBlock};
        
                switch(kernelType){
                        case Cudyn::Scheduler::KernelType::REDUCED_ATOMICS:{
                            Cudyn::Launcher::launch<Cudyn::Scheduler::ReducedAtomicScheduler>(config, SpMVKernel);
                            break;
                        } default: {
                            Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(config, SpMVKernel);                            
                        }
                    
                }
        
                Cudyn::Utils::errorCheck();
            }
        }
    }        

}