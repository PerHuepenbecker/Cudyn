#include "utils.cuh"

namespace cudyn::scheduler{

    template <typename TaskFunctor>
    __host__ void launch(KernelConfig, TaskFunctor f){

    }

    template <typename TaskFunctor>
    __global__ void generic_irregular_kernel(uint64_t total_tasks, uint64_t count_blocks, TaskFunctor f){

        //block wide counter for dynamic task assignment based on the index of the data thats up for processing
        __shared__ uint64_t counter;

        //central initialization of the counter variable by the first thread of the block
        if(threadIdx.x == 0) counter = 0;
        __syncthreads();

        // naive task splitting of total task count by the number of blocks used 
        
        uint64_t block_tasks = total_tasks/count_blocks;

        // assigning remaining tasks if M%T != 0 to the first blocks
        uint64_t additional_task = (blockIdx.x < total_tasks%count_blocks);

        block_tasks += additional_task;

        // Initialization of the inter block synchronization variable 

        uint64_t block_start_index = blockIdx.x * block_tasks + min((unsigned long) blockIdx.x, total_tasks % count_blocks);

        // Definition of the variable t which serves for intra block synchronization
        uint64_t t = 0;

        // Initial assignment of tasks to all threads to avoid synchronization issues right from the start by running directly into the atomic counter. 
        uint64_t thread_id = threadIdx.x;
        uint64_t num_threads = blockDim.x;

        if(thread_id < block_tasks){
            f(block_start_index+thread_id);
        }

        if(thread_idx.x == 0){
            counter = min(num_threads, block_tasks);
        }
        __syncthreads();

        // Task distribution and frequent update of the intra block synchronization update variable t
        while(t = atomicAdd((unsigned long long*)&counter, 1) < max_tasks){
            f(block_start_index+t);
        }
    }

    }