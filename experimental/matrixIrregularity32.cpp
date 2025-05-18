#include <iostream>


#include "../include/matrix_market_parser/MatrixMarketCSRParser.h"


#include <iostream>
#include <vector>
#include <limits>

#include <iostream>
#include <vector>
#include <limits>

#include <iostream>
#include <vector>
#include <limits>
#include <iomanip>  // für std::setprecision

template <typename T>
void getAverageNNZSpan32(CSRMatrix<T> matrix) {
    auto rowPointers = matrix.get_row_ptrs();
    auto numRows = rowPointers.size() - 1;

    size_t sumMin = 0;
    size_t sumMax = 0;
    size_t blockCount = 0;

    // Neue Buckets: 8 Intervalle
    std::vector<size_t> spanBuckets(8, 0);

    for (size_t i = 0; i < numRows; i += 32) {
        size_t blockEnd = std::min(i + 32, numRows);
        size_t minNNZ = std::numeric_limits<size_t>::max();
        size_t maxNNZ = std::numeric_limits<size_t>::min();

        for (size_t row = i; row < blockEnd; row++) {
            size_t nnz = rowPointers[row + 1] - rowPointers[row];
            minNNZ = std::min(minNNZ, nnz);
            maxNNZ = std::max(maxNNZ, nnz);
        }

        size_t span = maxNNZ - minNNZ;
        sumMin += minNNZ;
        sumMax += maxNNZ;
        blockCount++;

        // Neue Bucketing-Logik
        if (span < 10)
            spanBuckets[0]++;
        else if (span < 50)
            spanBuckets[1]++;
        else if (span < 100)
            spanBuckets[2]++;
        else if (span < 500)
            spanBuckets[3]++;
        else if (span < 1000)
            spanBuckets[4]++;
        else if (span < 2000)
            spanBuckets[5]++;
        else if (span < 5000)
            spanBuckets[6]++;
        else
            spanBuckets[7]++;
    }

    double averageSpan = (sumMax - sumMin) / static_cast<double>(blockCount);

    std::cout << "Average row length span over 32 rows: " << averageSpan << std::endl;

    std::cout << "\nSpan distribution (as % of blocks):\n";
    std::cout << std::fixed << std::setprecision(2);  // 2 Dezimalstellen

    std::cout << "  [   0 –    9]: " << (spanBuckets[0] * 100.0 / blockCount) << "%\n";
    std::cout << "  [  10 –   49]: " << (spanBuckets[1] * 100.0 / blockCount) << "%\n";
    std::cout << "  [  50 –   99]: " << (spanBuckets[2] * 100.0 / blockCount) << "%\n";
    std::cout << "  [ 100 –  499]: " << (spanBuckets[3] * 100.0 / blockCount) << "%\n";
    std::cout << "  [ 500 –  999]: " << (spanBuckets[4] * 100.0 / blockCount) << "%\n";
    std::cout << "  [1000 – 1999]: " << (spanBuckets[5] * 100.0 / blockCount) << "%\n";
    std::cout << "  [2000 – 4999]: " << (spanBuckets[6] * 100.0 / blockCount) << "%\n";
    std::cout << "  [5000+     ]: " << (spanBuckets[7] * 100.0 / blockCount) << "%\n";
}


int main(int argc, char** argv){
    if(argc != 2){
        std::cout << "Usage" << std::endl;
    }

    std::string filename = argv[1];

    auto fileDataType = MatrixMarketCSRParserBase::peekHeader(filename);

    if(fileDataType == MatrixMarketHeaderTypes::DataType::INTEGER){

        MatrixMarketCSRParser<int> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();

        getAverageNNZSpan32(csr_matrix);


    } else if(fileDataType == MatrixMarketHeaderTypes::DataType::REAL) {

        MatrixMarketCSRParser<double> parser(filename);
        auto csr_matrix = parser.exportCSRMatrix();

        getAverageNNZSpan32(csr_matrix);


    }


    

}
