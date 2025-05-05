//
// Created by Per Hüpenbecker on 30.04.25.
//


#include "MatrixMarketCSRParser.h"

MatrixMarketCSRParser::MatrixMarketCSRParser(const std::string &filename) {
    openFile(filename);
    parseHeader();
    if(!parseData()){
        throw std::runtime_error("DataParsing not possible");
    }
}

void MatrixMarketCSRParser::openFile(const std::string &filename) {
    file.open(filename);
    if(!file.is_open()){
        std::stringstream ss;
        ss << "Could not open file: " << filename;
        throw std::runtime_error(ss.str());
    }
}

void MatrixMarketCSRParser::closeFile(){
    if(file.is_open()){
        file.close();
    }
}

bool MatrixMarketCSRParser::parseHeaderArguments(const std::string& line){

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


bool MatrixMarketCSRParser::validateHeaderArguments(){
    if(header.objectType == ObjectType::UNKNOWN) {
        std::cerr << "[MatrixMarketParser] File contains unknown object type\n";
        return false;
    }
    if(header.dataType == DataType::UNSUPPORTED || header.dataType == DataType::UNKNOWN){
        std::cerr << "[MatrixMarketParser] File contains unsupported or unknown data type\n";
        return false;
    }
    if(header.symmetryType == SymmetryType::UNSUPPORTED || header.symmetryType == SymmetryType::UNKNOWN){
        std::cerr << "[MatrixMarketParser] File contains unsupported or unknown symmetry type\n";
        return false;
    }
    if(header.formatType == FormatType::ARRAY){
        std::cerr<< "[MatrixMarketParser] Array format currently unsupported.\n";
        return false;
    }

    if(header.formatType == FormatType::UNKNOWN) {
        std::cerr << "[MatrixMarketParser] File contains invalid format type\n";
        return false;
    }

    return true;
}

void MatrixMarketCSRParser::parseHeader() {

    std::string line;
    std::getline(file, line);

    if(line.substr(0,14) != "%%MatrixMarket"){
        std::cout << line.substr(0,14) << std::endl;
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

std::string MatrixMarketCSRParser::toLower(std::string str){
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){return std::tolower(c);});
    return str;
}

MatrixMarketCSRParser::ObjectType MatrixMarketCSRParser::parseObjectType(const std::string &objectString) {
    auto compStr = toLower(objectString);

    if(compStr == "matrix") return ObjectType::MATRIX;
    if(compStr == "vector") return ObjectType::VECTOR;
    return ObjectType::UNKNOWN;
}

MatrixMarketCSRParser::FormatType MatrixMarketCSRParser::parseFormatType(const std::string &formatString) {
    auto compStr = toLower(formatString);

    if(compStr == "coordinate") return FormatType::COORDINATE;
    if(compStr == "array") return FormatType::ARRAY;
    return FormatType::UNKNOWN;

}

MatrixMarketCSRParser::DataType MatrixMarketCSRParser::parseDataType(const std::string &dataTypeString) {
    auto compStr = toLower(dataTypeString);

    if(compStr == "real") return DataType::REAL;
    if(compStr == "integer") return DataType::INTEGER;
    if(compStr == "pattern" || compStr == "complex") return DataType::UNSUPPORTED;
    return DataType::UNKNOWN;

}

MatrixMarketCSRParser::SymmetryType MatrixMarketCSRParser::parseSymmetryType(const std::string &symmetryTypeString) {
    auto compStr = toLower(symmetryTypeString);

    if(compStr == "general") return SymmetryType::GENERAL;
    if(compStr == "symmetric") return SymmetryType::SYMMETRIC;
    if(compStr == "hermetian" || compStr == "skew-symmetric") return SymmetryType::UNSUPPORTED;
    return SymmetryType::UNKNOWN;
}

bool MatrixMarketCSRParser::parseSizeArguments(const std::string &line) {
    std::istringstream iss(line);

    if(header.formatType == FormatType::COORDINATE){
        if(!(iss >> header.rows >> header.columns >> header.nonZeroElements)){
            return false;
        }
    } else if(header.formatType == FormatType::ARRAY){
        if(!(iss >> header.rows >> header.columns)){
            return false;
        }
        header.nonZeroElements = header.rows * header.columns;
    } else {
        return false;
    }

    return true;
}

MatrixMarketCSRParser::~MatrixMarketCSRParser() {
    closeFile();
}

void MatrixMarketCSRParser::displayHeaderData() {
    switch (header.objectType) {
        case ObjectType::MATRIX:
            std::cout << "Matrix\n";
            break;
        default:
            std::cout << "Vector\n";
    }
    std::cout << "Format: ";
    switch(header.formatType){
        case FormatType::COORDINATE:
            std::cout << "coordinate\n";
            break;
        default:
            std::cout << "array\n";
            break;
    }
    std::cout << "Symmetry type: ";
    switch(header.symmetryType){
        case SymmetryType::GENERAL:
            std::cout<< "general\n";
            break;
        default:
            std::cout << "symmetric\n";
    }

    std::cout << "Data type: ";
    switch(header.dataType){
        case DataType::REAL:
            std::cout << "float\n";
            break;
        case DataType::INTEGER:
            std::cout << "integer\n";
            break;
        default:
            std::cout << "double\n";
            break;
    }

    std::cout << "-----------------" << std::endl;
    std::cout << "Rows: " << header.rows << " Columns: " << header.columns << " NNZ: " << header.nonZeroElements << std::endl;
    std::cout << "Unfolded non zero elements: " << unfoldedNonZeroElements << "\n";

    std::cout << "Checking for invalid zero values in data.." << std::endl;


}

bool MatrixMarketCSRParser::parseData() {

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
            if(header.symmetryType == SymmetryType::SYMMETRIC){
                nonZeroCounter++;
                rowElementCounter.at(column)++;
            }
        }
    }

    unfoldedNonZeroElements = nonZeroCounter;

    // Checking if the number of declared rows match up with the number of read rows form the file
    if(nonZeroCounter != header.nonZeroElements && header.symmetryType == SymmetryType::GENERAL){
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


    row_pointers = rowPointers;


    if (header.dataType == DataType::INTEGER){
        auto intData = std::make_unique<std::vector<int>>(unfoldedNonZeroElements);
        column_pointers.resize(unfoldedNonZeroElements);

        size_t row, column;
        int value;

        for(size_t i = 0; i < header.nonZeroElements; ++i){

            if(!std::getline(file, line)) return false;
            std::istringstream iss(line);
            if(!(iss>> row >> column >> value)) return false;

            // Handle 1 indexing of MatrixMarket format
            row -=1;
            column -=1;

            auto insertion_index = insertionPointers.at(row);
            intData->at(insertion_index) = value;
            column_pointers.at(insertion_index) = column;
            insertionPointers.at(row)++;

            //Handle possible symmetry
            if(row != column && header.symmetryType != SymmetryType::GENERAL){
                auto insertion_index_symmetric = insertionPointers.at(column);
                intData->at(insertion_index_symmetric) = value;
                column_pointers.at(insertion_index_symmetric) = row;
                insertionPointers.at(column)++;
            }
        }
        data_array = std::move(intData);
    } else if (header.dataType == DataType::REAL){
        auto doubleData = std::make_unique<std::vector<double>>(unfoldedNonZeroElements);
        column_pointers.resize(unfoldedNonZeroElements);

        size_t row, column;
        double value;


        for(size_t i = 0; i < header.nonZeroElements; ++i){

            if(!std::getline(file, line)) {
                return false;
            };
            std::istringstream iss(line);
            if(!(iss >> row >> column >> value)) {
                return false;
            }

            // Handle 1 indexing of MatrixMarket format
            row -=1;
            column -=1;

            auto insertion_index = insertionPointers.at(row);
            doubleData->at(insertion_index) = value;
            column_pointers.at(insertion_index) = column;
            insertionPointers.at(row)++;

            //Handle possible symmetry
            if(row != column && header.symmetryType != SymmetryType::GENERAL){
                auto insertion_index_symmetric = insertionPointers.at(column);
                doubleData->at(insertion_index_symmetric) = value;
                column_pointers.at(insertion_index_symmetric) = row;
                insertionPointers.at(column)++;
            }
        }
        data_array = std::move(doubleData);
    }
    return true;
}

void MatrixMarketCSRParser::displayCSRArrays() {
    std::cout << "Data:    ";
    if (auto int_ptr = std::get_if<std::unique_ptr<std::vector<int>>>(&data_array)) {
        for (int i : **int_ptr) {
            std::cout << i << " ";
        }
    } else if (auto dbl_ptr = std::get_if<std::unique_ptr<std::vector<double>>>(&data_array)) {
        for (double d : **dbl_ptr) {
            std::cout << d << " ";
        }
    }

    std::cout << "\n\n";

    std::cout << "Columns: ";
    for(const auto& el: column_pointers){
        std::cout << el << " ";
    }
    std::cout << "\n\n";

    std::cout << "Rows:    ";
    for(const auto& el: row_pointers){
        std::cout << el << " ";
    }
    std::cout << "\n\n";

}