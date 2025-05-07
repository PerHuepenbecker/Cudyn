#pragma once

#include <cstdint>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

typedef struct {
    const uint64_t total_tasks;
    const uint64_t grid_dimensions;
    const uint64_t block_dimensions;
    
} KernelConfig;

namespace utils {

    void errorCheck(){
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("[CudaDevicePointer] Allocation failed");
        }
    }
    
    // Wrapper for safe RAII handling of memory via smart pointer inspired approach
    template <typename T>
    class CudaDevicePointer {
        
        private:
            // Templated private internal pointer that is wrapped in RAII logic
            T* pointer_ = nullptr;
        
        public:
            CudaDevicePointer() = default;
            explicit CudaDevicePointer(size_t count){
                allocateMemory(count);
            }
            // Automatic handling of memory deallocation on destructor call
            ~CudaDevicePointer(){free();}

            // Make the pointer non copyable by explicitly deleting the copy and copy assignment operators
            CudaDevicePointer(const CudaDevicePointer&) = delete;
            CudaDevicePointer& operator=(const CudaDevicePointer&) = delete;

            // Allow move by defining move and move assignment operaors
            CudaDevicePointer(const CudaDevicePointer&& other) noexcept {
                pointer_ = other.pointer_;
                other.pointer_ = nullptr;
            }

            CudaDevicePointer& operator=(CudaDevicePointer&& other) noexcept {
                if(this != &other) {
                    free();
                    pointer_ = other.pointer_;
                    other.pointer_= nullptr;
                }
                return *this;
            }

            void allocateMemory(size_t count, bool zero = true){
                free();
                cudaMalloc((void**)&pointer_, sizeof(T) * count);
                errorCheck();
                if(zero){
                    cudaMemset(pointer_, 0, count);
                }
            }

            void free(){
                if(pointer_){
                    cudaFree(pointer_);
                    pointer_ = nullptr;
                }
            }

            T* get() const {return pointer_;}

            operator T*() const {return pointer_;}
    };
}

namespace grid_configuration {
    namespace details {

        // Validate if the grid configuration is valid and fits within the device limits of the GPU
        
        __host__ bool validate_grid_configuration_for_device(KernelConfig config, int device_id){
            cudaDeviceProp device_properties;
            cudaError_t error = cudaGetDeviceProperties(&device_properties, device_id);
            if (error != cudaSuccess) {
                std::cerr << "[cudyn error] - Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
                return false;
            }
            if(config.block_dimensions > device_properties.maxThreadsPerBlock){
                std::cerr << "[cudyn error] - The number of threads per block exceeds the maximum allowed for the device" << std::endl;
                return false;
            }
            if(config.grid_dimensions > device_properties.maxGridSize[0]){
                std::cerr << "[cudyn error] - The number of blocks exceeds the maximum allowed for the device" << std::endl;
                return false;
            }
            return true;
        }
    }

    template <uint64_t TotalTasks, uint64_t BlocksCount, uint64_t ThreadsPerBlock>
    __host__ constexpr KernelConfig manual_grid_configuration() {
    
    // Check if the number of total tasks, blocks, and threads per block are greater than 0
    // This is important to ensure that the kernel can be launched with valid parameters
    // and to avoid potential runtime errors or undefined behavior
    // The static_asserts will cause a compile-time error if the conditions are not met

    static_assert(TotalTasks > 0, "[cudyn bad argument] - The number of total tasks must be greater than 0");
    static_assert(BlocksCount > 0, "[cudyn bad argument] - The number of blocks must be greater than 0");
    static_assert(ThreadsPerBlock > 0, "[cudyn bad argument] - The number of threads per block must be greater than 0");

    // Check if the number of threads per block is a multiple of 32
    // This is important for performance reasons, as CUDA architectures are designed to work with warps of 32 threads
    // and having a number of threads per block that is not a multiple of 32 can lead to underutilization of the GPU

    if(ThreadsPerBlock % 32 != 0){
        std::cout << "[cudyn warning] - The number of threads per block should be a multiple of 32" << std::endl;
    }

    KernelConfig config{TotalTasks, BlocksCount, ThreadsPerBlock};

    assert(details::validate_grid_configuration_for_device(config, 0) && "[cudyn error] - The grid configuration is not valid for the device");

    return config;
}
}