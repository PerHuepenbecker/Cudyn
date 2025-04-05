#include <cstdint>
#include <tuple>

template<typename TaskFunctor>
class GenericIrregularWrapper{

    private:
    // TaskFunctor that implements the task that will be executed with dynamic task distribution
    TaskFunctor f;

    std::tuple kernel_arguments;

    // Kernel function that the actual task scheduling. Argument uint64_t defines the complete number of subtasks required to process the 
    // whole data supplied for processing by the TaskFunctor f. The argument uint64_t T defines the number of blocks the whole set of subtasks is 
    // being split by
    
    __global__ void generic_irregular_kernel(uint64_t M, uint64_t T, TaskFunctor f){

        //block wide counter for dynamic task assignment based on the index of the data thats up for processing
        __shared__ uint64_t counter;

        //central initialization of the counter variable by the first thread of the block
        if(threadIdx.x == 0) counter = 0;
        __syncthreads();

        // naive task splitting of total task count by the number of blocks used 
        // TODO: implement handling function for a number of tasks that doesnt split even by the number of blocks
        uint64_t max_tasks = M/T;

        // Initialization of the inter block synchronization variable i. 
        uint64_t i = blockIdx.x * max_tasks;
        // Definition of the variable t which serves for intra block synchronization
        uint64_t t;

        //TODO: implement proper initial assignment of tasks to all threads to avoid synchronization issues right from the start

        // Task distribution and frequent update of the intra block synchronization update variable t
        while(t = atomicAdd((unsigned long long*)&counter, 1) < max_tasks){
            f(i+t);
        }
    }

};