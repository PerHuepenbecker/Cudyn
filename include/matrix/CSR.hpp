
//
// Created by Per Hüpenbecker on 08.03.25.
//
#ifndef CSR_MATRIX_CPP_CSR_MATRIX_H
#define CSR_MATRIX_CPP_CSR_MATRIX_H

#include "Matrix.hpp"

template <typename U>
class CSRMatrix {
    private:
    std::vector<U> data;
    std::vector<size_t> col_indices;
    std::vector<size_t> row_ptrs;
    size_t rows_count;
    size_t cols_count;

    public:
    // Constructor accepting a Matrix object
    CSRMatrix(const Matrix<U>& matrix) {
        rows_count = matrix.get_rows_count();
        cols_count = matrix.get_cols_count();
        size_t row_ptr = 0;
        row_ptrs.push_back(row_ptr);
        auto matrix_data = matrix.get_data();

        for(size_t i = 0; i < rows_count; i++){
            for(size_t j = 0; j < cols_count; j++){

                if(matrix_data[i* cols_count + j] != 0){
                    data.push_back(matrix_data[i*cols_count + j]);
                    col_indices.push_back(j);
                    row_ptr++;
                }
            }
            row_ptrs.push_back(row_ptr);
        }
    }

    CSRMatrix(std::vector<U>data,std::vector<size_t> col_indices, std::vector<size_t> row_ptrs, size_t rows_count, size_t cols_count)
        :data(data), col_indices(col_indices), row_ptrs(row_ptrs), rows_count(rows_count), cols_count(cols_count) {}

    // display function of CSR Matrix
    void display() {
        for (const auto &i: data) {
            std::cout << i << " ";
        }
    }

    // display method for the full matrix including all zero values

    void display_full() {
        for (size_t i = 0; i < rows_count; i++) {
            auto start_index = row_ptrs[i];
            auto end_index = row_ptrs[i + 1];

            size_t k = start_index;

            for (size_t j = 0; j < cols_count; j++) {
                if (k < end_index && j == col_indices[k]) {
                    std::cout << data[k] << " ";
                    k++; // move to the next non-zero element
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << std::endl;
        }
    }

    // Overloaded operator for scalar multiplication of a csr matrix as modification in place

    void operator*(U scalar){
        for (auto& element: data){
            element *= scalar;
        }
    }

    // Overloaded operator for scalar multiplication of a csr matrix that returns a new CSRMatrix object

    CSRMatrix <U> operator*(U scalar) const {
        std::vector <U> scaled_data = data;
        for (auto& element: scaled_data){
            element *= scalar;
        }

        return CSRMatrix(data, col_indices, row_ptrs, rows_count, cols_count);
    }

    // Overloaded operator for matrix multiplication of a csr matrix with a dense
    // matrix that returns a new Matrix object

    friend Matrix<U> operator*(const CSRMatrix<U>& lhs, const Matrix<U>&rhs){

        if(lhs.cols_count != rhs.get_rows_count()){
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        std::vector<U> result(lhs.rows_count * rhs.get_rows_count(), 0);

        // First iterate over all rows in the lhs matrix. Standard for each matrix multiplication

        for(size_t i = 0; i < lhs.rows_count; i++){

            // Loop over the non-zero elements in the lhs csr matrix using the indexing of the row pointers

            for(size_t k = lhs.row_ptrs[i]; k < lhs.row_ptrs[i+1]; k++){
                size_t j = lhs.col_indices[k]; // column index of the non-zero row data
                U value = lhs.data[k]; // actual value of the non-zero lhs element

                // Iterate over the columns of the rhs matrix

                for (size_t l = 0; l < rhs.get_cols_count(); l++){
                    result[i*rhs.get_cols_count()+l] += value * rhs.get_data()[j * rhs.get_cols_count() + l];
                }

            }
        }
        return Matrix<U>(result, rhs.get_cols_count());
    }

    // Overloaded operator for CsMV multiplication of a csr matrix with a dense vector

    friend std::vector<U> operator*(const CSRMatrix<U>& lhs, const std::vector<U>& rhs){

        if(lhs.cols_count != rhs.size()){
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        std::vector<U> result(lhs.rows_count, 0);

        for(size_t i = 0; i < lhs.rows_count; i++){


            for(size_t k = lhs.row_ptrs[i]; k < lhs.row_ptrs[i+1]; k++){
                size_t j = lhs.col_indices[k]; 
                result[i] += lhs.data[k] * rhs[j];
            }
        }
        return result;
    }

    // Getters for the internal data to get them into gpu memory
    // Return const references to avoid copying the data 
    auto & get_data() const {
        return data; 
    }
    auto & get_col_indices() const {
        return col_indices; 
    }
    auto & get_row_ptrs() const {
        return row_ptrs; 
    }

    // Getters for the matrix dimensions
    auto get_rows_count() const {
        return rows_count; 
    }
    auto get_cols_count() const {
        return cols_count; 
    }

};


#endif //CSR_MATRIX_CPP_CSR_MATRIX_H
