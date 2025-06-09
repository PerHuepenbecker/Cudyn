#pragma once

#include "utils.cuh"
#include <iostream>     
#include <vector>       
#include <tuple>        
#include <algorithm> 




namespace Cudyn::Scheduler{


   

    enum class KernelType{STANDARD, REDUCED_ATOMICS, SUGGESTED};

    template <typename TaskFunctor>
    __global__ void genericIrregularKernelLowAtomics(uint64_t total_tasks, uint64_t count_blocks, TaskFunctor f){
        __shared__ uint64_t counter;

        if(threadIdx.x == 0) counter = 0;
        __syncthreads();
        
        uint64_t base_tasks = total_tasks / count_blocks;
        uint64_t remainder = total_tasks % count_blocks;

        uint64_t block_tasks = base_tasks + (blockIdx.x < remainder);

        uint64_t block_start_index = blockIdx.x * base_tasks + min((uint64_t)blockIdx.x, remainder);

        uint64_t thread_id = threadIdx.x;
        uint64_t num_threads = blockDim.x;

        if(thread_id < block_tasks){
            f(block_start_index+thread_id);
        }

        if(threadIdx.x == 0){
            counter = min(num_threads, block_tasks);
        }
        __syncthreads();

        const uint32_t full_warp_mask = 0xFFFFFFFF; 

        while (true) {
            uint64_t warp_claimed_task_start_offset; 
            uint32_t lane_id = threadIdx.x % warpSize; 
        
            if (lane_id == 0) {
                warp_claimed_task_start_offset = atomicAdd((unsigned long long*)&counter, (unsigned long long)warpSize);
            }
        
            warp_claimed_task_start_offset = __shfl_sync(full_warp_mask, warp_claimed_task_start_offset, 0);
                                                    
            if (warp_claimed_task_start_offset >= block_tasks) {
                break;
            }
        
            uint64_t my_task_offset_in_block = warp_claimed_task_start_offset + lane_id;
        

            if (my_task_offset_in_block < block_tasks) {
                f(block_start_index + my_task_offset_in_block);
            }
        }
    }


    template <typename TaskFunctor>
    __global__ void genericIrregularKernel(uint64_t total_tasks, uint64_t count_blocks, TaskFunctor f){

        //block wide counter for dynamic task assignment based on the index of the data thats up for processing
        __shared__ uint64_t counter;

        //central initialization of the counter variable by the first thread of the block
        if(threadIdx.x == 0) counter = 0;
        __syncthreads();

        // naive task splitting of total task count by the number of blocks used 
        
        uint64_t base_tasks = total_tasks / count_blocks;
        uint64_t remainder = total_tasks % count_blocks;

        // Calculate how many extra tasks this block gets
        uint64_t block_tasks = base_tasks + (blockIdx.x < remainder);

        // Compute the starting index for this block
        uint64_t block_start_index = blockIdx.x * base_tasks + min((uint64_t)blockIdx.x, remainder);

        // Definition of the variable t which serves for intra block synchronization
        uint64_t t = 0;

        // Initial assignment of tasks to all threads to avoid synchronization issues right from the start by running directly into the atomic counter. 
        uint64_t thread_id = threadIdx.x;
        uint64_t num_threads = blockDim.x;

        if(thread_id < block_tasks){
            f(block_start_index+thread_id);
        }

        if(threadIdx.x == 0){
            counter = min(num_threads, block_tasks);
        }
        __syncthreads();

    
        // Task distribution and frequent update of the intra block synchronization update variable t
        while(true){
            t = atomicAdd((unsigned long long*)&counter, 1);
            if(t >= block_tasks) break;
            f(block_start_index+t);
            }

        __syncthreads();
    }



    template <typename TaskFunctor>
    __global__ void genericIrregularKernelSuggested(uint64_t total_tasks, uint64_t count_blocks, TaskFunctor f) {
        __shared__ uint64_t counter_shared; 

        uint64_t base_tasks_per_block = total_tasks / count_blocks;
        uint64_t remainder_tasks = total_tasks % count_blocks;
        uint64_t tasks_for_this_block = base_tasks_per_block + (blockIdx.x < remainder_tasks);
        uint64_t block_start_index_global = blockIdx.x * base_tasks_per_block + min((uint64_t)blockIdx.x, remainder_tasks);

        if (threadIdx.x == 0) {
            counter_shared = 0; 
        }
        __syncthreads();

        uint64_t current_task_offset_in_block = atomicAdd((unsigned long long*)&counter_shared, 1);

        bool finished = false;

        while (true) {

            finished = f(block_start_index_global + current_task_offset_in_block);

            if(finished){
                uint64_t current_task_offset_in_block = atomicAdd((unsigned long long*)&counter_shared, 1);
                if (current_task_offset_in_block >= tasks_for_this_block) {
                    break;
                }
            }
    }



    struct StandardScheduler {
        template <typename Functor>
        static void launch(Cudyn::Utils::GridConfiguration::KernelConfig config, Functor f) {
            genericIrregularKernel<<<config.grid_dimensions, config.block_dimensions>>>(
                config.total_tasks, config.grid_dimensions, f);
        }
    };


    struct ReducedAtomicScheduler {
        template <typename Functor>
        static void launch(Cudyn::Utils::GridConfiguration::KernelConfig config, Functor f) {
            genericIrregularKernelLowAtomics<<<config.grid_dimensions, config.block_dimensions>>>(
                config.total_tasks, config.grid_dimensions, f);
        }
    };

    struct SuggestedScheduler {
        template<typename Functor>
        static void launch(Cudyn::Utils::GridConfiguration::KernelConfig config, Functor f) {
            genericIrregularKernelSuggested<<<config.grid_dimensions, config.block_dimensions>>>(
                config.total_tasks, config.grid_dimensions, f);
        }
    };

};

namespace Cudyn::Profiling{

    namespace Details{
        template<typename T>
        double average(const T& vec){
            double sum = 0;
            for(const auto& val: vec){
                sum+= val;
            }
            
            return vec.empty() ? 0.0 : sum/vec.size();}
    
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
    }



    template <typename SchedulingPolicy, typename TaskFunctor>
    __host__ auto launchProfiled(Cudyn::Utils::GridConfiguration::KernelConfig config, TaskFunctor f,size_t iterationsCount = 10, int deviceId = 0) {
        if (!Cudyn::Utils::GridConfiguration::Details::validatGridConfigurationForDevice(config, deviceId)) {
            std::cerr << "Invalid Grid configuration for device" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<float> measurements;

        for(int i = 0; i < iterationsCount; i++){
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            SchedulingPolicy::template launch(config, f);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            measurements.push_back(milliseconds);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaDeviceSynchronize();
        }
        
        double avg = Details::average(measurements);
        double stddev = Details::standardDeviation(measurements);

        std::cout << "Kernel execution time (" << typeid(SchedulingPolicy).name() << "): "
                  << avg << " ms" << std::endl;
                  std::cout << "Standard deviation: " << stddev << " ms" << std::endl;
        std::cout << "Number of measurements: " << measurements.size() << std::endl;
        std::cout << "Minimum time: " << *std::min_element(measurements.begin(), measurements.end()) << " ms" << std::endl;
        std::cout << "Maximum time: " << *std::max_element(measurements.begin(), measurements.end()) << " ms" << std::endl;
        
        return std::make_tuple(avg, stddev);
    }
}

namespace Cudyn::Launcher{


    template <typename SchedulingPolicy, typename TaskFunctor>
    __host__ void launch(Cudyn::Utils::GridConfiguration::KernelConfig config, TaskFunctor f, int deviceId = 0) {
        if (!Cudyn::Utils::GridConfiguration::Details::validatGridConfigurationForDevice(config, deviceId)) {
            std::cerr << "Invalid Grid configuration for device" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        SchedulingPolicy::template launch(config, f);

    }

};

