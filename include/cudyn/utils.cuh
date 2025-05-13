#pragma once

#include <cstdint>
#include <iostream>
#include <array>
#include <vector>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>



namespace Cudyn::Utils {

    void errorCheck(){
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("[CudaDevicePointer] Allocation failed");
        }
    }
    
    namespace Memory{
            // Wrapper for safe RAII handling of memory via smart pointer inspired approach
    template <typename T>
    class CudaDevicePointer {
        
        private:
            // Templated private internal pointer that is wrapped in RAII logic
            size_t size = 0;
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
            CudaDevicePointer(CudaDevicePointer&& other) noexcept {
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
                size = count;
                if(zero){
                    cudaMemset(pointer_, 0, count);
                }
            }

            void free(){
                if(pointer_){
                    cudaFree(pointer_);
                    pointer_ = nullptr;
                    size = 0;
                }
            }

            T* get() const {return pointer_;}

            operator T*() const {return pointer_;}
    };

    // Wrapper Classs for the RAII handled CudaDevicePointer for abstraction of basic cudaMemcpy() use to copy the data from host to device

    template <typename T>
    class CudaArray{
        private:
        CudaDevicePointer<T> pointer = CudaDevicePointer<T>{};
        size_t allocated_size = 0;
        bool is_empty = true;
      
        public: 
        CudaArray() = default;
        // Empty Constructor to preallocate a certain size of device memory
        explicit CudaArray(size_t size){
            pointer = CudaDevicePointer<T>(size);
            allocated_size = size;
        }
        
        CudaArray(const std::vector<T>& vec){
            size_t size = vec.size();
            pointer = CudaDevicePointer<T>(size);
            allocated_size = size;
            if(size != 0){
                is_empty = false;
            }

            cudaMemcpy(pointer.get(), vec.data(), size * sizeof(T), cudaMemcpyHostToDevice);
            errorCheck();
        }

        template<std::size_t N>
        CudaArray(const std::array<T, N>& arr){
            size_t size = N;
            pointer = CudaDevicePointer<T>(size);
            allocated_size = size;
            
            cudaMemcpy(pointer.get(), arr.data(), N * sizeof(T), cudaMemcpyHostToDevice);
            errorCheck();
            if(size != 0){
                is_empty = false;
            }
        }

        ~CudaArray(){
            // Explicit free
            pointer.free();
        }

        // Deleting copy constructors to prevent unnecessary global GPU memory usage by performing deep copies in the GPU memory

        CudaArray(const CudaArray& other) = delete;
        CudaArray& operator=(const CudaArray& other) = delete;

        //Explicitly allowing move constructors 

        CudaArray(CudaArray&&other) noexcept {
            pointer = other.pointer;
            allocated_size = other.allocated_size;
            is_empty = other.is_empty;
            other.pointer = nullptr;
            other.allocated_size = 0;
            other.is_empty = true;
        }

        CudaArray& operator=(CudaArray&&other) noexcept {
            pointer = other.pointer;
            allocated_size = other.allocated_size;
            is_empty = other.is_empty;
            other.pointer = nullptr;
            other.allocated_size = 0;
            other.is_empty = true;
        }

        // method to "upload" data from the host to the GPU
        void upload(std::vector<T> vec){
            
            if(allocated_size != vec.size()){
                pointer.free();
                pointer.allocateMemory(vec.size());
                allocated_size = vec.size();
            }

            cudaMemcpy(pointer.get(), vec.data(), sizeof(T)*vec.size(), cudaMemcpyHostToDevice);
            if(vec.size > 0) is_empty = false;
            errorCheck();
        }

        template <std::size_t N>
        void upload(std::array<T,N> arr){
                pointer.free();
                pointer.allocateMemory(N);
                allocated_size = N;

            cudaMemcpy(pointer.get(), arr.data(), sizeof(T)*N, cudaMemcpyHostToDevice);
            errorCheck();
            if(N != 0) is_empty = false;
        }

        void download(std::vector<T> &vec){
            
            if(vec.size() != allocated_size){
                std::cerr << "[Utils::Memory] Supplied Container size too small. Reallocating internal memory" << std::endl;
                vec.resize(allocated_size);
            }

            cudaMemcpy(vec.data(), pointer.get(), sizeof(T)*allocated_size, cudaMemcpyDeviceToHost);            
            errorCheck();
        }


        template <std::size_t N>
        void download(std::array<T,N> &arr){
            static_assert(N > 0, "Array of size 0 not supported here");
            if(N != allocated_size){
                throw std::runtime_error("[Memory::CudaArray::download] Array size mismatch. Copy data from GPU not possible");
            }

            cudaMemcpy(arr.data(), pointer.get(), sizeof(T)*N, cudaMemcpyDeviceToHost);            
            errorCheck();
        }

        T* data(){
            return pointer.get();
        }

        bool isEmpty(){
            return is_empty;
        }

        size_t size(){
            return allocated_size;
        }

    };

    

}




namespace GridConfiguration {


    typedef struct {

        const size_t total_tasks;
        const size_t grid_dimensions;
        const size_t block_dimensions;

    } KernelConfig;


    namespace Details {

        // Validate if the grid configuration is valid and fits within the device limits of the GPU
        
        __host__ bool validatGridConfigurationForDevice(KernelConfig config, int deviceId){
            cudaDeviceProp device_properties;
            cudaError_t error = cudaGetDeviceProperties(&device_properties, deviceId);
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
    __host__ constexpr KernelConfig ManualConfigurationCheck(int deviceId = 0) {
    
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

    if(!Details::validatGridConfigurationForDevice(config, deviceId)){

    }

    return config;
}
}
};