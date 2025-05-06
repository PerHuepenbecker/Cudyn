#include "../matrix/CSR.hpp"
#include "utils.cuh"

namespace CSR {
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

    namespace Datastructures {

        template <typename T>
        struct DeviceDataCSR {

            utils::CudaDevicePointer<T> csrMatrixData_d = nullptr;
            utils::CudaDevicePointer<size_t> csrMatrixRowPtrs_d = nullptr;
            utils::CudaDevicePointer<size_t> csrMatrixColIndices_d = nullptr;
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
                cudaMemcpy(csrMatrixRowPtrs_d, row_pointers.data(), row_pointers.size() * sizeof(T), cudaMemcpyHostToDevice);
                cudaMemcpy(csrMatrixColIndices_d, column_pointers.data(), column_pointers.size() * sizeof(T), cudaMemcpyHostToDevice);

                utils::errorCheck();  
            };
        };
        
        template<typename T>
        struct DeviceDataSpMV{

            DeviceDataCSR<T> csrData;
            utils::CudaDevicePointer<T> multiplicationVector = nullptr;
            utils::CudaDevicePointer<T> result = nullptr;

            DeviceDataSpMV(const CSRMatrix<T>& csrMatrix, std::vector<T> multVec){
                if(csrMatrix.get_cols_count() != multVec.size()){
                    throw std::runtime_error("Invalid matrix and vector dimensions for SpMV")
                }
                multiplicationVector.allocateMemory(multVec.size());
                result.allocateMemory(multVec.size());
                
            };

        };
    };
}

