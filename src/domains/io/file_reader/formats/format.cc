#include "format.hh"

namespace Jetstream::Modules {

namespace {

struct Entry {
    FileReaderFormat::Option option;
    std::unique_ptr<FileReaderFormat> (*create)();
};

template<typename T>
std::unique_ptr<FileReaderFormat> Make() {
    return std::make_unique<T>();
}

const std::vector<Entry> kEntries = {
    {{"raw", "Raw I/Q"}, Make<FileReaderFormat::Raw>},
    {{"sigmf", "SigMF"}, Make<FileReaderFormat::Sigmf>},
    {{"wav", "WAV"}, Make<FileReaderFormat::Wav>},
};

}  // namespace

const std::vector<FileReaderFormat::Option>& FileReaderFormat::Options() {
    static const std::vector<Option> options = [] {
        std::vector<Option> result;
        for (const auto& entry : kEntries) {
            result.push_back(entry.option);
        }
        return result;
    }();
    return options;
}

std::unique_ptr<FileReaderFormat> FileReaderFormat::Create(const std::string& name) {
    for (const auto& entry : kEntries) {
        if (entry.option.value == name) {
            return entry.create();
        }
    }
    return nullptr;
}

bool FileReaderFormat::describesSamples() const {
    return true;
}

}  // namespace Jetstream::Modules
