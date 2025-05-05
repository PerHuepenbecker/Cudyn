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
#include <vector>
#include <iostream>

class MatrixMarketCSRParser {

public:

    enum class ObjectType{MATRIX, VECTOR, UNKNOWN};
    // complex and pattern left out here and will be evaluated as unsupported
    enum class DataType{INTEGER, REAL, UNSUPPORTED, UNKNOWN};
    enum class FormatType {COORDINATE, ARRAY, UNKNOWN};
    // Skew symmetric and hermitian left out here
    enum class SymmetryType {GENERAL, SYMMETRIC, UNSUPPORTED, UNKNOWN};

    struct MatrixMarketHeader {
        ObjectType objectType = ObjectType::UNKNOWN;
        FormatType formatType = FormatType::UNKNOWN;
        DataType dataType = DataType::UNKNOWN;
        SymmetryType symmetryType = SymmetryType::UNKNOWN;

        size_t rows = 0;
        size_t columns = 0;
        size_t nonZeroElements = 0;
    };

    void displayHeaderData();
    void displayCSRArrays();

    MatrixMarketCSRParser(const std::string& filename);
    ~MatrixMarketCSRParser();

private:

    std::ifstream file;
    std::streampos dataStartPosition;

    size_t unfoldedNonZeroElements = 0;

    DataType detectedType;
    MatrixMarketHeader header;

    // Complex variant type for handling of multiple data types here
    // Using unique pointers so avoid deep copying in the data parsing process
    std::variant<std::unique_ptr<std::vector<int>>, std::unique_ptr<std::vector<double>>> data_array;
    std::vector<size_t> column_pointers;
    std::vector<size_t> row_pointers;


    // File handling methods
    void openFile(const std::string& filename);
    void closeFile();

    void parseHeader();
    bool parseHeaderArguments(const std::string& line);
    bool parseSizeArguments(const std::string& line);
    bool validateHeaderArguments();

    bool parseData();

    // Header argument parsing mathods
    ObjectType parseObjectType(const std::string& objectString);
    FormatType parseFormatType(const std::string& formatString);
    DataType parseDataType(const std::string& dataTypeString);
    SymmetryType parseSymmetryType (const std::string& symmetryTypeString);

    // Helper method for case insensitive header input evaluation
    std::string toLower(std::string str);

};

// Implementations



#endif //MATRIXMARKETPARSER_MATRIXMARKETCSRPARSER_H
