#include <iostream>
#include <random>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <unordered_map>

#include<map>
#include <string>
#include <vector>
#include <limits>

#include "../include/cudyn/utils.cuh"
#include "../include/cudyn/scheduler.cuh"


typedef struct{
    unsigned int baseNumber = 0;
    unsigned int steps = 0;
    unsigned int tasksWorked = 0;
} result;


template <typename TaskFunctor>
__global__ void generic_regular_kernel (size_t task_count, TaskFunctor f){

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < task_count){
        f(index);
    }
}



int main(int argc, char** argv){
    std::cout << "Starting..." << std::endl;

    if(argc != 5  || (argv[1] == "--help" || argv[1] == "-h")){
        std::cout << "Usage: <prog> <problem size as 2^x> <percentage of early finishers (0.x)> <block size> <T>"<< std::endl;
        exit(0);
    } 

    unsigned int PROBLEM_SIZE = 0;
    unsigned int tmp =  std::stoi(argv[1]);
    if(tmp == 32){
        PROBLEM_SIZE = (1 << tmp)-1;
    } else {
        PROBLEM_SIZE = (1 << tmp);
    }

    float percentageOneIt = std::stof(argv[2]);
    
    size_t threadsPerBlock = std::stoi(argv[3]);
    size_t tasksPerThread = std::stoi(argv[4]);



    std::cout << "Initialized" << std::endl;

    std::vector<uint64_t> vec_h(PROBLEM_SIZE);
    std::vector<result> results_h(PROBLEM_SIZE);
    std::vector<size_t> tasksWorked_h(PROBLEM_SIZE);
    
    // defines the maximumm number of iterations per thread 

    int maxValue = 1<<25;
    std::mt19937 gen (123456);
    std::uniform_real_distribution <double> dist2 (0 , 1);

    for(auto &el: vec_h){
        el = (dist2(gen) <= percentageOneIt) ? 1: maxValue;
    }

    Cudyn::Utils::Memory::CudaArray<uint64_t> vec_d (vec_h);
    Cudyn::Utils::Memory::CudaArray<result> results_d (results_h);
    Cudyn::Utils::Memory::CudaArray<size_t> tasksWorked_d(tasksWorked_h);
   
    auto data_vec = vec_d.data();
    auto results_vec = results_d.data();
    auto tasksWorked_vec = tasksWorked_d.data();

    auto subtractingLogic = [data_vec,results_vec, tasksWorked_vec] __device__ (size_t i){
        unsigned int steps = 0;
        float value = data_vec[i];
        float initial_value = value;
        while(true){
            
            steps++;

            if(value == 1){
                results_vec[i].baseNumber = initial_value;
                results_vec[i].steps = steps;
                tasksWorked_vec[blockIdx.x * blockDim.x + threadIdx.x] += 1;
                break;
            }

            initial_value += (sinf(initial_value) +1);

            // minimal work done here for each iteration
            //initial_value += 0.000001f;

            value--;
        }
    };

    std::vector<size_t> tasksPerThreadArgs{2,4,8,16,32};
    //std::vector<size_t> threadsPerBlockArgs {32,64,128,256,512,1024};
    
    


            std::fill(tasksWorked_h.begin(), tasksWorked_h.end(), 0);
            tasksWorked_d.upload(tasksWorked_h); 

            size_t total_threads = (PROBLEM_SIZE + tasksPerThread - 1) / tasksPerThread;
            size_t numBlocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;


            std::cout << "Dynamic T= " << tasksPerThread << " : " << threadsPerBlock << " : " << numBlocks << std::endl; 
            Cudyn::Utils::GridConfiguration::KernelConfig kernelConfig{.total_tasks = PROBLEM_SIZE, .grid_dimensions = numBlocks, .block_dimensions = threadsPerBlock};
            Cudyn::Profiling::launchProfiled<Cudyn::Scheduler::StandardScheduler>(kernelConfig, subtractingLogic, 1);

            cudaDeviceSynchronize();

            std::cout << "\n";
    
            // Get data back to host
            tasksWorked_d.download(tasksWorked_h);

            // Simple lamdba to count the distribution of worked tasks by the thread
             auto evaluateTaskSharing = [&tasksWorked_h, numBlocks, threadsPerBlock](){
                
                std::map<size_t, size_t> counts;

                for(int i = 0; i < threadsPerBlock * numBlocks; ++i){
                    size_t tasksWorked = tasksWorked_h.at(i);
                    counts[tasksWorked] += 1;
                }
                
                return std::move(counts);
            
            };

    // calling lambda

    auto countedTasks = evaluateTaskSharing();

    for (const auto& [tasksWorked, threadCount] : countedTasks) {

    double percentage = 100.0 * threadCount / (threadsPerBlock * numBlocks);
    std::cout << "\tCount of threads that worked " << tasksWorked << " tasks: " << threadCount << " (" << percentage << "%)" << std::endl;

}
std::cout << "\n";




    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numBlocks_reg = (PROBLEM_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    generic_regular_kernel<<<numBlocks_reg, threadsPerBlock>>>(PROBLEM_SIZE, subtractingLogic);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Regular mapped: Block size: "<< threadsPerBlock << " : " << milliseconds  << "ms" << std::endl;




    unsigned int num_one = 0;
    unsigned int num_other = 0;

    
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        if(results_h[i].steps == 1){
            num_one++;
    }
        else{
            num_other++;
        }
    }
   
    std::cout << "Number of steps == 1: " << num_one << std::endl;
    std::cout << "Number of steps != 1: " << num_other << std::endl;

    auto percentage_one = ((double)num_one / PROBLEM_SIZE) * 100;
    auto percentage_other = ((double)num_other / PROBLEM_SIZE) * 100;



    std::cout << std::fixed << std::setprecision(2);
    std::cout << "----------------------" << std::endl;
    std::cout << "Percentage of steps == 1: " << percentage_one << "%" << std::endl;
    std::cout << "Percentage of steps != 1: " << percentage_other << "%" << std::endl;
    std::cout << "----------------------" << std::endl;
    

}