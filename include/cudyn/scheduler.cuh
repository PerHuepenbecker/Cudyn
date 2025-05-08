#pragma once

#include "utils.cuh"




namespace Cudyn::Scheduler{


    struct StandardScheduler {
        template <typename Functor>
        static void launch(GridConfiguration::KernelConfig config, Functor f) {
            genericIrregularKernel<<<config.grid_dimensions, config.block_dimensions>>>(
                config.total_tasks, config.grid_dimensions, f);
        }
    };


    struct ReducedAtomicScheduler {
        template <typename Functor>
        static void launch(GridConfiguration::KernelConfig config, Functor f) {
            genericIrregularKernelLowAtomics<<<config.grid_dimensions, config.block_dimensions>>>(
                config.total_tasks, config.grid_dimensions, f);
        }
    };


    enum class KernelType{STANDARD, REDUCED_ATOMICS, BATCH};

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

        //block_tasks += additional_task;

        // Initialization of the inter block synchronization variable 

        //uint64_t block_start_index = blockIdx.x * block_tasks + min((unsigned long) blockIdx.x, total_tasks % count_blocks);

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
};

namespace Cudyn::Launcher{


    template <typename SchedulingPolicy, typename TaskFunctor>
    __host__ void launch(GridConfiguration::KernelConfig config, TaskFunctor f, int deviceId = 0) {
        if (!GridConfiguration::Details::validatGridConfigurationForDevice(config, deviceId)) {
            std::cerr << "Invalid Grid configuration for device" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        SchedulingPolicy::template launch(config, f);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Kernel execution time (" << typeid(SchedulingPolicy).name() << "): "
                  << milliseconds << " ms" << std::endl;
    }

    template <typename TaskFunctor>
    __host__ void launch_old(GridConfiguration::KernelConfig kernelConfig, TaskFunctor f, Scheduler::KernelType type = Scheduler::KernelType::STANDARD, int deviceId = 0){

        // TODO: Integrate GridConfigruationChecking functionality

        if(!GridConfiguration::Details::validatGridConfigurationForDevice(kernelConfig, deviceId)){
            std::cerr << "Invalid Grid dimensions for GPU" << std::endl;
            exit(1);
        }

        uint64_t block_tasks = kernelConfig.total_tasks / kernelConfig.grid_dimensions;

        std::cout << "Tasks per block " << block_tasks << std::endl;
        std::cout << "Remainder: " << kernelConfig.total_tasks % kernelConfig.grid_dimensions << std::endl;
        std::cout << "Tasks per thread " << block_tasks / kernelConfig.block_dimensions << std::endl;

        cudaEvent_t start, stop;
        
        if(type == Scheduler::KernelType::STANDARD){
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            Scheduler::genericIrregularKernel<<<kernelConfig.grid_dimensions, kernelConfig.block_dimensions>>>(kernelConfig.total_tasks,kernelConfig.grid_dimensions, f);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Dynamic Kernel execution time Standard: " << milliseconds << "ms" << std::endl;


        } else if(type == Scheduler::KernelType::BATCH){

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start); 

            Scheduler::genericIrregularKernelBatch<<<kernelConfig.grid_dimensions, kernelConfig.block_dimensions>>>(kernelConfig.total_tasks,kernelConfig.grid_dimensions, f);  

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Batched Kernel execution time: " << milliseconds << "ms" << std::endl;
        }
        
        else {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start); 

            Scheduler::genericIrregularKernelLowAtomics<<<kernelConfig.grid_dimensions, kernelConfig.block_dimensions>>>(kernelConfig.total_tasks,kernelConfig.grid_dimensions, f);  

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Dynamic Kernel execution time reduced Atomics: " << milliseconds << "ms" << std::endl;

        }
        
        

    }
};