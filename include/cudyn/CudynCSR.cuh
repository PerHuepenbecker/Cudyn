#pragma once

#include "../matrix/CSR.hpp"
#include "utils.cuh"

#include <string>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace CudynCSR {
    namespace Helpers {

        template<typename T>
        std::vector<T> generate_multiplication_vector(size_t problem_size, T baseValue = 1, bool random = false, int seed = 42, T lower_bound = 0, T upper_bound = 2 << 20){
            std::vector<T> vec (problem_size);
            if(!random){
                std::fill(vec.begin(), vec.end(), baseValue);
            } else {
                static_assert(std::is_same_v<T, int> || std::is_same_v<T, double>, "Only int and double types are supported for random generation");
                std::mt19937 gen(seed);
                if constexpr (std::is_same_v<T,int>){
                    std::uniform_int_distribution<> dist(lower_bound, upper_bound);
                    for(auto& el: vec){
                        el = dist(gen);
                    }
                } else {
                    std::uniform_real_distribution<> dist (lower_bound, upper_bound);
                    for(auto& el:vec){
                        el = dist(gen);
                    }
                }
                
            }
               return vec;
    }
}

    namespace Datastructures {

        template <typename T>
        struct DeviceDataCSR {

            Utils::CudaDevicePointer<T> csrMatrixData_d;
            Utils::CudaDevicePointer<size_t> csrMatrixRowPtrs_d;
            Utils::CudaDevicePointer<size_t> csrMatrixColIndices_d;
            size_t rows = 0;
            size_t columns = 0;
            
            DeviceDataCSR(const CSRMatrix<T>& csrMatrix){
                auto data = csrMatrix.get_data();
                auto column_pointers = csrMatrix.get_col_indices();
                auto row_pointers = csrMatrix.get_row_ptrs();
                rows = csrMatrix.get_rows_count();
                columns = csrMatrix.get_cols_count();

                // allocation by the device pointer wrappers
                csrMatrixData_d.allocateMemory(data.size());
                csrMatrixColIndices_d.allocateMemory(column_pointers.size());
                csrMatrixRowPtrs_d.allocateMemory(row_pointers.size());

                // copying the contents to device memory
                cudaMemcpy(csrMatrixData_d, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
                cudaMemcpy(csrMatrixRowPtrs_d, row_pointers.data(), row_pointers.size() * sizeof(size_t), cudaMemcpyHostToDevice);
                cudaMemcpy(csrMatrixColIndices_d, column_pointers.data(), column_pointers.size() * sizeof(size_t), cudaMemcpyHostToDevice);


                Utils::errorCheck();  
            };
        };
        
        template<typename T>
        struct DeviceDataSpMV{

            DeviceDataCSR<T> csrData;
            Utils::CudaDevicePointer<T> multiplicationVector;
            Utils::CudaDevicePointer<T> result;

            DeviceDataSpMV(const CSRMatrix<T>& csrMatrix, std::vector<T> multVec)
                :csrData(csrMatrix)
                {
                if(csrMatrix.get_cols_count() != multVec.size()){
                    throw std::runtime_error("Invalid matrix and vector dimensions for SpMV");
                }

            
                multiplicationVector.allocateMemory(multVec.size(), false);
                result.allocateMemory(multVec.size());
                
                Utils::errorCheck();

                cudaMemcpy(multiplicationVector, multVec.data(), sizeof(T)*multVec.size(), cudaMemcpyHostToDevice);
            };

            std::vector<T> getResult(){
                std::vector<T> result_h(csrData.rows, 0);
                cudaMemcpy(result_h.data(), result, sizeof(T) * csrData.rows, cudaMemcpyDeviceToHost);
                return result_h;
            }
        };
    };

    namespace Kernel{

        template<typename T>
        struct CudynCSRSpMV {
            const T* data_d = nullptr;
            const size_t* columnIndices_d = nullptr;
            const size_t* rowPointers_d = nullptr;
            const T* multiplicationVector_d = nullptr;
            T* resultVector_d = nullptr;

            __host__ CudynCSRSpMV(Datastructures::DeviceDataSpMV<T>& deviceData){
                data_d = deviceData.csrData.csrMatrixData_d.get();
                columnIndices_d = deviceData.csrData.csrMatrixColIndices_d.get();
                rowPointers_d = deviceData.csrData.csrMatrixRowPtrs_d.get();

                multiplicationVector_d = deviceData.multiplicationVector.get();
                resultVector_d = deviceData.result.get();
            }

            __device__ void operator()(size_t i) const {
                size_t row_start_offset = rowPointers_d[i];
                size_t row_end_offset = rowPointers_d[i+1];
                
                T dotProduct = 0;

                for (size_t j = row_start_offset; j < row_end_offset; ++j) {
                    size_t col_index = columnIndices_d[j]; 
                    T value = data_d[j];                 
        
                    //printf("Value: %lf\n", value);
                    //printf("Multvec: %lf\n", multiplicationVector_d[col_index]);

                    dotProduct += value * multiplicationVector_d[col_index];
                }

                resultVector_d[i] = dotProduct;
                
            }
        };
    }

}

