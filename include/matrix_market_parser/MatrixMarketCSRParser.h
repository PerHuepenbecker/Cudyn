//
// Created by Per Hüpenbecker on 30.04.25.
//

#ifndef MATRIXMARKETPARSER_MATRIXMARKETCSRPARSER_H
#define MATRIXMARKETPARSER_MATRIXMARKETCSRPARSER_H


#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <variant>
#include <memory>
#include <iostream>
#include <functional>

#include "../matrix/CSR.hpp"

namespace MatrixMarketHeaderTypes{
    enum class ObjectType{MATRIX, VECTOR, UNKNOWN};
// complex and pattern left out here and will be evaluated as unsupported
    enum class DataType{INTEGER, REAL, PATTERN,UNSUPPORTED, UNKNOWN};
    enum class FormatType {COORDINATE, ARRAY, UNKNOWN};
// Skew symmetric and hermitian left out here for now
    enum class SymmetryType {GENERAL, SYMMETRIC, UNSUPPORTED, UNKNOWN};
}



template<typename T>
class MatrixMarketCSRParser {


public:


    struct MatrixMarketHeader {
        MatrixMarketHeaderTypes::ObjectType objectType = MatrixMarketHeaderTypes::ObjectType::UNKNOWN;
        MatrixMarketHeaderTypes::FormatType formatType = MatrixMarketHeaderTypes::FormatType::UNKNOWN;
        MatrixMarketHeaderTypes::DataType dataType = MatrixMarketHeaderTypes::DataType::UNKNOWN;
        MatrixMarketHeaderTypes::SymmetryType symmetryType = MatrixMarketHeaderTypes::SymmetryType::UNKNOWN;

        size_t rows = 0;
        size_t columns = 0;
        size_t nonZeroElements = 0;
        // Even if not an actual matrix market header argument this variable is placed here because it is dependent on
        // symmetry type and header supplied nnz
        size_t unfoldedNonZeroElements = 0;
    };

    struct CSRArrays{
        std::vector<T> data_array;
        //std::variant<std::unique_ptr<std::vector<int>>, std::unique_ptr<std::vector<double>>> data_array;
        std::vector<size_t> column_pointers;
        std::vector<size_t> row_pointers;
    };

    void displayHeaderData();
    void displayCSRArrays();
    CSRMatrix<T> exportCSRMatrix();


    MatrixMarketCSRParser(const std::string& filename);
    ~MatrixMarketCSRParser();

private:

    std::ifstream file;
    std::streampos dataStartPosition;

    MatrixMarketHeaderTypes::DataType detectedType;
    MatrixMarketHeader header;

    // Complex variant type for handling of multiple data types here
    // Using unique pointers so avoid deep copying in the data parsing process
    CSRArrays csrArrays;

    // File handling methods
    void openFile(const std::string& filename);
    void closeFile();

    void parseHeader();
    bool parseHeaderArguments(const std::string& line);
    bool parseSizeArguments(const std::string& line);
    bool validateHeaderArguments();

    bool parseData();

    // Header argument parsing mathods
    MatrixMarketHeaderTypes::ObjectType parseObjectType(const std::string& objectString);
    MatrixMarketHeaderTypes::FormatType parseFormatType(const std::string& formatString);
    MatrixMarketHeaderTypes::DataType parseDataType(const std::string& dataTypeString);
    MatrixMarketHeaderTypes::SymmetryType parseSymmetryType (const std::string& symmetryTypeString);

    // Helper method for case insensitive header input evaluation
    std::string toLower(std::string str);

};

// Implementations

template<typename T>
MatrixMarketCSRParser<T>::MatrixMarketCSRParser(const std::string &filename) {
    openFile(filename);
    parseHeader();
    if(!parseData()){
        throw std::runtime_error("DataParsing not possible");
    }
}
template<typename T>
void MatrixMarketCSRParser<T>::openFile(const std::string &filename) {
    file.open(filename);
    if(!file.is_open()){
        std::stringstream ss;
        ss << "Could not open file: " << filename;
        throw std::runtime_error(ss.str());
    }
}
template<typename T>
void MatrixMarketCSRParser<T>::closeFile(){
    if(file.is_open()){
        file.close();
    }
}
template<typename T>
bool MatrixMarketCSRParser<T>::parseHeaderArguments(const std::string& line){

    std::istringstream iss(line.substr(14));
    std::string objectTypeString, formatString, dataTypeString, symmetryString;

    if(!(iss >> objectTypeString))
        return false;
    if(!(iss >> formatString))
        return false;
    if(!(iss >> dataTypeString))
        return false;
    if(!(iss >> symmetryString))
        symmetryString = "general";

    header.objectType = parseObjectType(objectTypeString);
    header.dataType = parseDataType(dataTypeString);
    header.formatType = parseFormatType(formatString);
    header.symmetryType = parseSymmetryType(symmetryString);

    return true;
}

template<typename T>
bool MatrixMarketCSRParser<T>::validateHeaderArguments(){
    if(header.objectType == MatrixMarketHeaderTypes::ObjectType::UNKNOWN) {
        std::cerr << "[MatrixMarketParser] File contains unknown object type\n";
        return false;
    }
    if(header.dataType == MatrixMarketHeaderTypes::DataType::UNSUPPORTED || header.dataType == MatrixMarketHeaderTypes::DataType::UNKNOWN){
        std::cerr << "[MatrixMarketParser] File contains unsupported or unknown data type\n";
        return false;
    }
    if(header.symmetryType == MatrixMarketHeaderTypes::SymmetryType::UNSUPPORTED || header.symmetryType == MatrixMarketHeaderTypes::SymmetryType::UNKNOWN){
        std::cerr << "[MatrixMarketParser] File contains unsupported or unknown symmetry type\n";
        return false;
    }
    if(header.formatType == MatrixMarketHeaderTypes::FormatType::ARRAY){
        std::cerr<< "[MatrixMarketParser] Array format currently unsupported.\n";
        return false;
    }

    if(header.formatType == MatrixMarketHeaderTypes::FormatType::UNKNOWN) {
        std::cerr << "[MatrixMarketParser] File contains invalid format type\n";
        return false;
    }

    return true;
}

template<typename T>
void MatrixMarketCSRParser<T>::parseHeader() {

    std::string line;
    std::getline(file, line);

    if(line.substr(0,14) != "%%MatrixMarket"){
        throw std::runtime_error("File seems to be malformed. Missing %%MatrixMarket");
    }

    // Parsing the header arguments
    parseHeaderArguments(line);
    // Check if all parsed header arguments are valid and not corrupted
    if(!validateHeaderArguments()){
        throw std::runtime_error("[MatrixMarketParser] File header contains invalid data. Exiting.");
    }

    bool foundSizeArguments = false;

    // Looping over possible comment lines after the header line until the size arguments line is found

    while(std::getline(file, line) && !foundSizeArguments){
        // detect and skip comment lines
        if(line.at(0) == '%') continue;
        // if the read line is not a comment line it has to be a size line
        dataStartPosition = file.tellg();
        if(!parseSizeArguments(line)){
            throw std::runtime_error("MatrixMarket file contains invalid size argument line");
        }
        foundSizeArguments = true;
    }
}

template<typename T>
std::string MatrixMarketCSRParser<T>::toLower(std::string str){
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){return std::tolower(c);});
    return str;
}

template<typename T>
MatrixMarketHeaderTypes::ObjectType MatrixMarketCSRParser<T>::parseObjectType(const std::string &objectString) {
    auto compStr = toLower(objectString);

    if(compStr == "matrix") return MatrixMarketHeaderTypes::ObjectType::MATRIX;
    if(compStr == "vector") return MatrixMarketHeaderTypes::ObjectType::VECTOR;
    return MatrixMarketHeaderTypes::ObjectType::UNKNOWN;
}

template<typename T>
MatrixMarketHeaderTypes::FormatType MatrixMarketCSRParser<T>::parseFormatType(const std::string &formatString) {
    auto compStr = toLower(formatString);

    if(compStr == "coordinate") return MatrixMarketHeaderTypes::FormatType::COORDINATE;
    if(compStr == "array") return MatrixMarketHeaderTypes::FormatType::ARRAY;
    return MatrixMarketHeaderTypes::FormatType::UNKNOWN;

}

template <typename T>
MatrixMarketHeaderTypes::DataType MatrixMarketCSRParser<T>::parseDataType(const std::string &dataTypeString) {
    auto compStr = toLower(dataTypeString);

    if(compStr == "real") return MatrixMarketHeaderTypes::DataType::REAL;
    if(compStr == "integer") return MatrixMarketHeaderTypes::DataType::INTEGER;
    if(compStr == "pattern") return MatrixMarketHeaderTypes::DataType::PATTERN;
    if(compStr == "complex") return MatrixMarketHeaderTypes::DataType::UNSUPPORTED;

    return MatrixMarketHeaderTypes::DataType::UNKNOWN;

}

template <typename T>
MatrixMarketHeaderTypes::SymmetryType MatrixMarketCSRParser<T>::parseSymmetryType(const std::string &symmetryTypeString) {
    auto compStr = toLower(symmetryTypeString);

    if(compStr == "general") return MatrixMarketHeaderTypes::SymmetryType::GENERAL;
    if(compStr == "symmetric") return MatrixMarketHeaderTypes::SymmetryType::SYMMETRIC;
    if(compStr == "hermetian" || compStr == "skew-symmetric") return MatrixMarketHeaderTypes::SymmetryType::UNSUPPORTED;
    return MatrixMarketHeaderTypes::SymmetryType::UNKNOWN;
}

template <typename T>
bool MatrixMarketCSRParser<T>::parseSizeArguments(const std::string &line) {
    std::istringstream iss(line);

    if(header.formatType == MatrixMarketHeaderTypes::FormatType::COORDINATE){
        if(!(iss >> header.rows >> header.columns >> header.nonZeroElements)){
            return false;
        }
    } else if(header.formatType == MatrixMarketHeaderTypes::FormatType::ARRAY){
        if(!(iss >> header.rows >> header.columns)){
            return false;
        }
        header.nonZeroElements = header.rows * header.columns;
    } else {
        return false;
    }

    return true;
}

template<typename T>
MatrixMarketCSRParser<T>::~MatrixMarketCSRParser() {
    closeFile();
}
template<typename T>
void MatrixMarketCSRParser<T>::displayHeaderData() {
    switch (header.objectType) {
        case MatrixMarketHeaderTypes::ObjectType::MATRIX:
            std::cout << "Matrix\n";
            break;
        default:
            std::cout << "Vector\n";
    }
    std::cout << "Format: ";
    switch(header.formatType){
        case MatrixMarketHeaderTypes::FormatType::COORDINATE:
            std::cout << "coordinate\n";
            break;
        default:
            std::cout << "array\n";
            break;
    }
    std::cout << "Symmetry type: ";
    switch(header.symmetryType){
        case MatrixMarketHeaderTypes::SymmetryType::GENERAL:
            std::cout<< "general\n";
            break;
        default:
            std::cout << "symmetric\n";
    }

    std::cout << "Data type: ";
    switch(header.dataType){
        case MatrixMarketHeaderTypes::DataType::REAL:
            std::cout << "float\n";
            break;
        case MatrixMarketHeaderTypes::DataType::INTEGER:
            std::cout << "integer\n";
            break;
        default:
            std::cout << "pattern\n";
            break;
    }

    std::cout << "-----------------" << std::endl;
    std::cout << "Rows: " << header.rows << " Columns: " << header.columns << " NNZ: " << header.nonZeroElements << std::endl;
    std::cout << "Unfolded non zero elements: " << header.unfoldedNonZeroElements << "\n";

    std::cout << "Checking for invalid zero values in data.." << std::endl;


}

template<typename T>
bool MatrixMarketCSRParser<T>::parseData() {

    // Declaring a counter variable to check if the read lines match up with the size data from the header
    size_t nonZeroCounter = 0;
    size_t line_counter = 0;
    // Temporary input line string
    std::string line;
    // declaring a row wise counter for later construction of the row pointer array in csr format
    std::vector<int> rowElementCounter(header.rows);

    file.clear();
    file.seekg(dataStartPosition);

    // Counting the elements per row in the first pass
    while(std::getline(file, line)){
        line_counter++;
        std::istringstream iss(line);
        int row = 0;
        int column = 0;

        // since row and column is index data it must be > 0
        if(!(iss >> row >> column) || (row-=1) < 0 || (column -=1) < 0){
            std::stringstream ss;
            ss << "[MatrixMarketParser] Error parsing data from file. Error in data line " << nonZeroCounter +1;
            throw std::runtime_error(ss.str());
        }

        // Handling symmetry
        // Count once if its on the diagonal

        if(row == column){
            nonZeroCounter++;
            rowElementCounter.at(row) ++;
        } else {
            nonZeroCounter++;
            rowElementCounter.at(row) ++;
            if(header.symmetryType == MatrixMarketHeaderTypes::SymmetryType::SYMMETRIC){
                nonZeroCounter++;
                rowElementCounter.at(column)++;
            }
        }
    }

    header.unfoldedNonZeroElements = nonZeroCounter;

    // Checking if the number of declared rows match up with the number of read rows form the file
    if(nonZeroCounter != header.nonZeroElements && header.symmetryType == MatrixMarketHeaderTypes::SymmetryType::GENERAL){
        std::stringstream ss;
        ss << "[MatrixMarketParser] Header declared number of non zeros and processed non zero elements don't add up.\n";
        ss << "                     Counted: " << nonZeroCounter << " | Declared: " << header.nonZeroElements << "\n";
        throw std::runtime_error(ss.str());
    }


    // Constructing the row pointer array from the counted elements fer row for csr matrix format
    std::vector<size_t> rowPointers(header.rows+1);

    // Zero indexing for first element
    rowPointers.at(0) = 0;
    // Assigning the indices
    for (size_t i = 0; i < header.rows; ++i) {

        rowPointers.at(i+1) = rowPointers.at(i) + rowElementCounter.at(i);
    }
    // Resetting the input file stream for second pass
    file.clear();
    file.seekg(dataStartPosition);

    // acquiring a copy of the row pointer vector for handling the insertion indices in the data and column arrays of the csr format
    std::vector<size_t> insertionPointers = rowPointers;


    csrArrays.row_pointers = std::move(rowPointers);
    csrArrays.data_array = std::vector<T>(header.unfoldedNonZeroElements);
    csrArrays.column_pointers.resize(header.unfoldedNonZeroElements);

    size_t row, column;
    T value;

    // Lambdas as a logic wrapper to handle Integer or Pattern data which is basically Integer data with only value 1

    auto readRegularData = [&row, &column, &value](std::istringstream &iss){
        if(!(iss>> row >> column >> value)) return false;
        return true;
    };

    auto readPatternData = [&row, &column, &value](std::istringstream &iss){
        if(!(iss >> row >> column)) return false;
        value = 1;
        return true;
    };

    std::function regular(readRegularData);
    std::function pattern(readPatternData);
    std::function reader = (header.dataType == MatrixMarketHeaderTypes::DataType::PATTERN) ? pattern : regular;

    for(size_t i = 0; i < header.nonZeroElements; ++i){

        if(!std::getline(file, line)) return false;
        std::istringstream iss(line);
        if(!(reader(iss))) return false;

        // Handle 1 indexing of MatrixMarket format
        row -=1;
        column -=1;

        auto insertion_index = insertionPointers.at(row);
        csrArrays.data_array.at(insertion_index) = value;
        csrArrays.column_pointers.at(insertion_index) = column;
        insertionPointers.at(row)++;

        //Handle possible symmetry
        if(row != column && header.symmetryType != MatrixMarketHeaderTypes::SymmetryType::GENERAL){
            auto insertion_index_symmetric = insertionPointers.at(column);
            csrArrays.data_array.at(insertion_index_symmetric) = value;
            csrArrays.column_pointers.at(insertion_index_symmetric) = row;
            insertionPointers.at(column)++;
        }
    }
    return true;
}


template<typename T>
void MatrixMarketCSRParser<T>::displayCSRArrays() {
    std::cout << "Data:    ";
    for(const auto& element: csrArrays.data_array){
        std::cout << element << " ";
    }

    std::cout << "\n\n";

    std::cout << "Columns: ";
    for(const auto& el: csrArrays.column_pointers){
        std::cout << el << " ";
    }
    std::cout << "\n\n";

    std::cout << "Rows:    ";
    for(const auto& el: csrArrays.row_pointers){
        std::cout << el << " ";
    }
    std::cout << "\n\n";
}

// Method for exporting the internal CSR data as a proper CSR Matrix
template <typename T>
CSRMatrix<T> MatrixMarketCSRParser<T>::exportCSRMatrix(){
    return CSRMatrix(std::move(csrArrays.data_array), std::move(csrArrays.column_pointers), std::move(csrArrays.row_pointers), header.rows, header.columns);
}


namespace MatrixMarketCSRParserBase{

        // Helper function to determine the correct template instantiation while using the parser
        // Code and minor logic duplication is accepted in contrast to way more complex handling
        // if using other approaches to allow some runtime flexibility

        auto peekHeader(const std::string& filename){
            std::ifstream file;
            file.open(filename);
            if(!file.is_open()){
                std::stringstream ss;
                ss << "[PeekHeader] Could not open file: " << filename;
                throw std::runtime_error(ss.str());
            }

            std::string line;
            std::getline(file, line);

            if(line.substr(0,14) != "%%MatrixMarket"){
                throw std::runtime_error("File seems to be malformed. Missing %%MatrixMarket");
            }

            // Parsing the header arguments

            std::istringstream iss(line.substr(14));
            std::string dataTypeString;

            if(!(iss >> dataTypeString))
                return MatrixMarketHeaderTypes::DataType::UNKNOWN;
            if(!(iss >> dataTypeString))
                return MatrixMarketHeaderTypes::DataType::UNKNOWN;
            if(!(iss >> dataTypeString))
                return MatrixMarketHeaderTypes::DataType::UNKNOWN;

            MatrixMarketHeaderTypes::DataType type = MatrixMarketHeaderTypes::DataType::UNKNOWN;

            if(dataTypeString == "real") type = MatrixMarketHeaderTypes::DataType::REAL;
            // Since pattern also can use internal integer
            if(dataTypeString == "integer" || dataTypeString == "pattern") type = MatrixMarketHeaderTypes::DataType::INTEGER;
            return type;
        }
}


#endif //MATRIXMARKETPARSER_MATRIXMARKETCSRPARSER_H
