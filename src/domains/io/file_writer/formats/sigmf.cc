#include "format.hh"

#include <chrono>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "jetstream/fmt/chrono.h"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Modules {

namespace {

const std::unordered_map<DataType, std::string> kSigmfDataTypes = {
    {DataType::CF32, "cf32_le"},
    {DataType::F32, "rf32_le"},
    {DataType::CI16, "ci16_le"},
    {DataType::I16, "ri16_le"},
    {DataType::CU16, "cu16_le"},
    {DataType::U16, "ru16_le"},
    {DataType::CI8, "ci8"},
    {DataType::I8, "ri8"},
    {DataType::CU8, "cu8"},
    {DataType::U8, "ru8"},
};

std::string CurrentDatetime() {
    const auto now = std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now());
    return jst::fmt::format("{:%Y-%m-%dT%H:%M:%S}Z", now);
}

}  // namespace

std::filesystem::path FileWriterFormat::Sigmf::resolve(const std::filesystem::path& path) {
    auto base = path;
    const auto extension = path.extension();
    if (extension == ".sigmf-meta" || extension == ".sigmf-data") {
        base = path.parent_path() / path.stem();
    }

    metaPath = base;
    metaPath += ".sigmf-meta";

    auto dataPath = base;
    dataPath += ".sigmf-data";
    return dataPath;
}

std::vector<std::filesystem::path> FileWriterFormat::Sigmf::companions() const {
    return {metaPath};
}

Result FileWriterFormat::Sigmf::check(const Tensor& input) const {
    if (!kSigmfDataTypes.contains(input.dtype())) {
        JST_ERROR("[MODULE_FILE_WRITER] SigMF does not support the input data type.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result FileWriterFormat::Sigmf::start(std::ofstream&, const Tensor& input) {
    datatype = kSigmfDataTypes.at(input.dtype());
    sampleRate = NumericAttribute(input, "sampleRate");
    captures.push_back({
        .sampleStart = 0,
        .frequency = NumericAttribute(input, "frequency"),
        .datetime = CurrentDatetime(),
    });

    return write();
}

Result FileWriterFormat::Sigmf::track(const Tensor& input, const U64 sampleIndex) {
    if (captures.empty()) {
        return Result::SUCCESS;
    }

    bool changed = false;
    if (!sampleRate) {
        sampleRate = NumericAttribute(input, "sampleRate");
        changed = sampleRate.has_value();
    }

    const auto frequency = NumericAttribute(input, "frequency");
    if (frequency != captures.back().frequency) {
        changed = true;
        if (captures.back().sampleStart == sampleIndex) {
            captures.back().frequency = frequency;
        } else {
            captures.push_back({
                .sampleStart = sampleIndex,
                .frequency = frequency,
                .datetime = CurrentDatetime(),
            });
        }
    }

    return changed ? write() : Result::SUCCESS;
}

Result FileWriterFormat::Sigmf::finish(std::ofstream&, U64) {
    return datatype.empty() ? Result::SUCCESS : write();
}

Result FileWriterFormat::Sigmf::write() const {
    nlohmann::json global = {
        {"core:datatype", datatype},
        {"core:version", "1.2.0"},
        {"core:recorder", "CyberEther"},
    };
    if (sampleRate) {
        global["core:sample_rate"] = *sampleRate;
    }

    nlohmann::json entries = nlohmann::json::array();
    for (const auto& capture : captures) {
        nlohmann::json entry = {
            {"core:sample_start", capture.sampleStart},
            {"core:datetime", capture.datetime},
        };
        if (capture.frequency) {
            entry["core:frequency"] = *capture.frequency;
        }
        entries.push_back(std::move(entry));
    }

    const nlohmann::json json = {
        {"global", std::move(global)},
        {"captures", std::move(entries)},
        {"annotations", nlohmann::json::array()},
    };

    const std::string meta = Platform::PathToUtf8(metaPath);
    if (Platform::WriteFileAtomic(meta, json.dump(2) + "\n") != Result::SUCCESS) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to write SigMF metadata '{}'.", meta);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
