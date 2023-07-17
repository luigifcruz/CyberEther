#ifndef JETSTREAM_HELPER_HH
#define JETSTREAM_HELPER_HH

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace Jetstream::Helper {

inline std::vector<char> LoadFile(const std::string& filename) {
    auto filesize = std::filesystem::file_size(filename);
    std::vector<char> data(filesize);
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        JST_FATAL("[YAML] Can't open configuration file.");
        JST_CHECK_THROW(Result::ERROR);
    }

    file.read(data.data(), filesize);

    if (!file) {
        JST_FATAL("[YAML] Can't open configuration file.");
        JST_CHECK_THROW(Result::ERROR);
    }

    return data;
}

inline std::vector<std::string> SplitString(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0;
    size_t lastPos = 0;
    while ((pos = str.find(delimiter, lastPos)) != std::string::npos) {
        result.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos + delimiter.length();
    }
    result.push_back(str.substr(lastPos));
    return result;
}

}  // namespace Jetstream::Helper

#endif