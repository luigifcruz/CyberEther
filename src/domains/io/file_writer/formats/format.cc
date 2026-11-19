#include "format.hh"

#include <any>

namespace Jetstream::Modules {

namespace {

struct Entry {
    FileWriterFormat::Option option;
    std::unique_ptr<FileWriterFormat> (*create)();
};

template<typename T>
std::unique_ptr<FileWriterFormat> Make() {
    return std::make_unique<T>();
}

const std::vector<Entry> kEntries = {
    {{"raw", "Raw I/Q"}, Make<FileWriterFormat::Raw>},
    {{"sigmf", "SigMF"}, Make<FileWriterFormat::Sigmf>},
    {{"wav", "WAV"}, Make<FileWriterFormat::Wav>},
};

}  // namespace

const std::vector<FileWriterFormat::Option>& FileWriterFormat::Options() {
    static const std::vector<Option> options = [] {
        std::vector<Option> result;
        for (const auto& entry : kEntries) {
            result.push_back(entry.option);
        }
        return result;
    }();
    return options;
}

std::unique_ptr<FileWriterFormat> FileWriterFormat::Create(const std::string& name) {
    for (const auto& entry : kEntries) {
        if (entry.option.value == name) {
            return entry.create();
        }
    }
    return nullptr;
}

std::filesystem::path FileWriterFormat::resolve(const std::filesystem::path& path) {
    return path;
}

std::vector<std::filesystem::path> FileWriterFormat::companions() const {
    return {};
}

Result FileWriterFormat::check(const Tensor&) const {
    return Result::SUCCESS;
}

Result FileWriterFormat::start(std::ofstream&, const Tensor&) {
    return Result::SUCCESS;
}

Result FileWriterFormat::track(const Tensor&, U64) {
    return Result::SUCCESS;
}

Result FileWriterFormat::finish(std::ofstream&, U64) {
    return Result::SUCCESS;
}

std::optional<F64> FileWriterFormat::NumericAttribute(const Tensor& tensor, const std::string& key) {
    if (!tensor.hasAttribute(key)) {
        return std::nullopt;
    }
    const auto value = tensor.attribute(key);
    if (const auto* number = std::any_cast<F32>(&value)) {
        return *number;
    }
    if (const auto* number = std::any_cast<F64>(&value)) {
        return *number;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Modules
