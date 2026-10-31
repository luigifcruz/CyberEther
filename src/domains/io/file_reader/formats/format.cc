#include "format.hh"

#include "jetstream/platform.hh"

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
    {{"auto", "Auto"}, nullptr},
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

std::unique_ptr<FileReaderFormat> FileReaderFormat::Create(const std::string& name,
                                                           const std::string& filepath) {
    for (const auto& entry : kEntries) {
        if (entry.option.value == name && entry.create) {
            return entry.create();
        }
    }

    if (name != "auto") {
        return nullptr;
    }

    const auto path = Platform::PathFromUtf8(filepath);
    for (const auto& entry : kEntries) {
        if (!entry.create) {
            continue;
        }
        auto format = entry.create();
        if (format->matches(path)) {
            return format;
        }
    }
    return std::make_unique<Raw>();
}

bool FileReaderFormat::matches(const std::filesystem::path&) const {
    return false;
}

bool FileReaderFormat::describesSamples() const {
    return true;
}

}  // namespace Jetstream::Modules
