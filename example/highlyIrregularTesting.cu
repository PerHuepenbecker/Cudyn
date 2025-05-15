#include <iostream>
#include <random>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <unordered_map>


#include <string>
#include <vector>
#include <limits>

    


#include "../include/cudyn/scheduler.cuh"

#define THRESHOLD_BASE 22



typedef struct{
    unsigned int block_index;
    unsigned int thread_index;
    unsigned int global_index;
    unsigned int warp_no;
    unsigned int num_tasks_worked;
    unsigned int thread_array[16];
} debugData;

template <typename TaskFunctor>
__global__ void generic_irregular_kernel(uint64_t M, uint64_t T, debugData* debugData, TaskFunctor f){
    
    unsigned int task_counter = 0;
    unsigned int thread_array[16] = {0};

    __shared__ uint64_t counter; 
    if(threadIdx.x == 0) counter = 0;
    __syncthreads();

    uint64_t max_tasks = M/T;

    unsigned int i = blockIdx.x * max_tasks;
    unsigned int t;

    while((t = atomicAdd((unsigned long long*)&counter, 1)) < max_tasks){
        unsigned int index = i + t;
        f(index);
        thread_array[task_counter] = index;
        task_counter++;
    }

    debugData[blockIdx.x * blockDim.x + threadIdx.x].block_index = blockIdx.x;
    debugData[blockIdx.x * blockDim.x + threadIdx.x].thread_index = threadIdx.x;
    debugData[blockIdx.x * blockDim.x + threadIdx.x].global_index = blockIdx.x * blockDim.x + threadIdx.x;
    debugData[blockIdx.x * blockDim.x + threadIdx.x].warp_no = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    debugData[blockIdx.x * blockDim.x + threadIdx.x].num_tasks_worked = task_counter;
    for(int i = 0; i < 16; i++){
        debugData[blockIdx.x * blockDim.x + threadIdx.x].thread_array[i] = thread_array[i];
    }
}

typedef struct{
    unsigned int baseNumber;
    unsigned int steps;
    unsigned int workerThread;
} result;


template <typename TaskFunctor>
__global__ void generic_regular_kernel (TaskFunctor f){

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    f(index);
}


void print_worked_tasks(debugData* debugData, unsigned int global_index) {
    std::cout << "---------------------" << std::endl;
    std::cout << "Printing worked tasks" << std::endl;
    std::cout << "Block index: " << debugData[global_index].block_index << std::endl;
    std::cout << "Thread index: " << debugData[global_index].thread_index << std::endl;
    std::cout << "Global index: " << debugData[global_index].global_index << std::endl;
    std::cout << "Warp number: " << debugData[global_index].warp_no << std::endl;
    std::cout << "Number of tasks worked: " << debugData[global_index].num_tasks_worked << std::endl;
    std::cout << "Tasks worked: " << std::endl;
    for(int i = 0; i < 16; i++){
        std::cout << debugData[global_index].thread_array[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "---------------------" << std::endl;
}

void print_task_distros(debugData* debugData, unsigned int NUM_BLOCKS, unsigned int NUM_THREADS){

    std::cout << "Printing task distros" << std::endl;

    std::unordered_map<unsigned int, unsigned int> workloads;

    for(int i = 0; i < NUM_BLOCKS * NUM_THREADS; i++){
        if(workloads.find(debugData[i].num_tasks_worked) != workloads.end()){
            workloads[debugData[i].num_tasks_worked]++;
        }
        else workloads.insert({debugData[i].num_tasks_worked, 1});
    }

    for (auto const& x : workloads)
    {
        std::cout << "Task distros: " << x.first << " : " << x.second << " times" << std::endl;
    }
}


int main(int argc, char** argv){
    std::cout << "Starting..." << std::endl;

    if(argc != 6){
        std::cerr << "Usage: <prog> <problem_size as 2^x> <blocks_dynamic> <threads_dynamic> <blocks_static> <threads_static>"<< std::endl;
        exit(1);
    }

    unsigned int tmp =  std::stoi(argv[1]);
    unsigned int PROBLEM_SIZE = 1 << tmp;
    unsigned int NUM_BLOCKS = std::stoi(argv[2]);
    unsigned int NUM_THREADS = std::stoi(argv[3]);
    unsigned int REGULAR_KERNEL_BLOCKS = std::stoi(argv[4]);
    unsigned int REGULAR_KERNEL_THREADS = std::stoi(argv[5]);
    

    // Your new threshold calculation logic here...
    // Example:
    double desired_fraction_for_steps_one = 0.75; // Target 25%
    uint64_t max_random_data_value = (1ULL << THRESHOLD_BASE) - 1;
    uint64_t calculated_threshold;
    if (desired_fraction_for_steps_one >= 1.0) { calculated_threshold = 0; }
        else if (desired_fraction_for_steps_one <= 0.0) { calculated_threshold = max_random_data_value; }
        else { calculated_threshold = static_cast<uint64_t>(max_random_data_value * (1.0 - desired_fraction_for_steps_one)); }

    auto threshold = calculated_threshold; // This is the threshold the lambda will capture


    //auto threshold = (1ULL << THRESHOLD_BASE) - ((1ULL << THRESHOLD_BASE) / 2);

    std::mt19937 gen(12345);
    std::uniform_int_distribution<uint64_t> dist(1, (1ULL << THRESHOLD_BASE) -1);

    uint64_t* data_h = (uint64_t*) malloc(sizeof(uint64_t) * PROBLEM_SIZE);

    std::vector<uint64_t> vec_h(PROBLEM_SIZE);


    result* results_h = (result*) malloc(sizeof(result) * PROBLEM_SIZE);

    uint64_t* data_d;
    result* results_d;

    debugData* debugData_h = (debugData*) malloc(sizeof(debugData) * NUM_BLOCKS * NUM_THREADS);
    debugData* debugData_d;

    cudaMalloc(&debugData_d, sizeof(debugData) * NUM_BLOCKS * NUM_THREADS);
    cudaMalloc(&results_d, sizeof(result) * PROBLEM_SIZE);
    cudaMalloc(&data_d, sizeof(uint64_t) * PROBLEM_SIZE);

    int maxValue = 4 * ((1 << 23) -1) ;
    double percentageOneIt = 0.25;
    std::mt19937 gen_2 (123);
    std::uniform_real_distribution <double> dist2 (0 , 1);

    for(auto &el: vec_h){
        el = (dist2(gen_2) <= percentageOneIt) ? 1: maxValue;
    }

    Cudyn::Utils::Memory::CudaArray<uint64_t> vec_d (vec_h);


    cudaMemcpy(data_d, data_h, sizeof(uint64_t) * PROBLEM_SIZE, cudaMemcpyHostToDevice);

    std::cout << "Data generated" << std::endl;
    std::cout << "Starting Kernel" << std::endl;
    std::cout << "Problem Size: " << PROBLEM_SIZE << std::endl;
    std::cout << "Threshold Dimension: " << THRESHOLD_BASE << std::endl;

    auto counting_logic = [data_d, results_d, threshold] __device__ (uint64_t i){
        unsigned int steps = 0;
        unsigned int x = 0;

        while(1){
            steps ++; 
            if (data_d[i] + x > threshold) {
                results_d[i].baseNumber = data_d[i];
                results_d[i].steps = steps;
                results_d[i].workerThread = blockIdx.x*blockDim.x+threadIdx.x;
                break;
            }
            x++;
        }
    };

    auto data_vec = vec_d.data();

    auto subtractingLogic = [data_vec,results_d] __device__ (size_t i){
        unsigned int steps = 0;
        unsigned int initial_value = data_vec[i];
        unsigned int value = initial_value;
        while(true){
            steps++;
            if(value == 1){
                results_d[i].baseNumber = data_vec[i];
                results_d[i].steps = steps;
                results_d[i].workerThread = blockIdx.x*blockDim.x+threadIdx.x;
                break;
            }
            value--;
        }
    };

    Cudyn::Utils::GridConfiguration::KernelConfig kernelConfig{.total_tasks = PROBLEM_SIZE, .grid_dimensions = NUM_BLOCKS, .block_dimensions = NUM_THREADS};
    
    Cudyn::Launcher::launch<Cudyn::Scheduler::StandardScheduler>(kernelConfig, subtractingLogic);

    Cudyn::Launcher::launch<Cudyn::Scheduler::ReducedAtomicScheduler>(kernelConfig, subtractingLogic);

    //generic_irregular_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(PROBLEM_SIZE,NUM_BLOCKS, debugData_d, counting_logic);
    cudaMemcpy(results_h, results_d, sizeof(result) * PROBLEM_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(debugData_h, debugData_d, sizeof(debugData) * NUM_BLOCKS * NUM_THREADS, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    generic_regular_kernel<<<REGULAR_KERNEL_BLOCKS, REGULAR_KERNEL_THREADS>>>(subtractingLogic);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Regular mapped implementation took " << milliseconds << std::endl;

    cudaFree(data_d);
    cudaFree(debugData_d);
    cudaFree(results_d);

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


    for(int i = 0; i < NUM_BLOCKS*NUM_THREADS; i++){
        if(debugData_h[i].num_tasks_worked == 7 || debugData_h[i].num_tasks_worked == 9){
            print_worked_tasks(debugData_h, i);
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "----------------------" << std::endl;
    std::cout << "Percentage of steps == 1: " << percentage_one << "%" << std::endl;
    std::cout << "Percentage of steps != 1: " << percentage_other << "%" << std::endl;
    std::cout << "----------------------" << std::endl;


    free(data_h);
    free(results_h);

}