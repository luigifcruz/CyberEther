#include "module_impl.hh"

#include <any>
#include <chrono>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "jetstream/fmt/chrono.h"
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

std::optional<F64> NumericAttribute(const Tensor& tensor, const std::string& key) {
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

std::string CurrentDatetime() {
    const auto now = std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now());
    return jst::fmt::format("{:%Y-%m-%dT%H:%M:%S}Z", now);
}

}  // namespace

void FileWriterImpl::resolveSigmfPaths() {
    const auto extension = filePath.extension();
    auto base = filePath;
    if (extension == ".sigmf-meta" || extension == ".sigmf-data") {
        base = filePath.parent_path() / filePath.stem();
    }

    filePath = base;
    filePath += ".sigmf-data";
    metadata.metaPath = base;
    metadata.metaPath += ".sigmf-meta";
}

Result FileWriterImpl::startSigmf(const Tensor& input) {
    const auto type = kSigmfDataTypes.find(input.dtype());
    if (type == kSigmfDataTypes.end()) {
        JST_ERROR("[MODULE_FILE_WRITER] SigMF does not support the input data type.");
        return Result::ERROR;
    }

    metadata.datatype = type->second;
    metadata.sampleRate = NumericAttribute(input, "sampleRate");
    metadata.captures.push_back({
        .sampleStart = 0,
        .frequency = NumericAttribute(input, "frequency"),
        .datetime = CurrentDatetime(),
    });

    return writeSigmf();
}

Result FileWriterImpl::trackSigmfCapture(const Tensor& input, const U64 sampleIndex) {
    if (metadata.captures.empty()) {
        return Result::SUCCESS;
    }

    bool changed = false;
    if (!metadata.sampleRate) {
        metadata.sampleRate = NumericAttribute(input, "sampleRate");
        changed = metadata.sampleRate.has_value();
    }

    const auto frequency = NumericAttribute(input, "frequency");
    if (frequency != metadata.captures.back().frequency) {
        changed = true;
        if (metadata.captures.back().sampleStart == sampleIndex) {
            metadata.captures.back().frequency = frequency;
        } else {
            metadata.captures.push_back({
                .sampleStart = sampleIndex,
                .frequency = frequency,
                .datetime = CurrentDatetime(),
            });
        }
    }

    return changed ? writeSigmf() : Result::SUCCESS;
}

Result FileWriterImpl::writeSigmf() const {
    nlohmann::json global = {
        {"core:datatype", metadata.datatype},
        {"core:version", "1.2.0"},
        {"core:recorder", "CyberEther"},
    };
    if (metadata.sampleRate) {
        global["core:sample_rate"] = *metadata.sampleRate;
    }

    nlohmann::json captures = nlohmann::json::array();
    for (const auto& capture : metadata.captures) {
        nlohmann::json entry = {
            {"core:sample_start", capture.sampleStart},
            {"core:datetime", capture.datetime},
        };
        if (capture.frequency) {
            entry["core:frequency"] = *capture.frequency;
        }
        captures.push_back(std::move(entry));
    }

    const nlohmann::json json = {
        {"global", std::move(global)},
        {"captures", std::move(captures)},
        {"annotations", nlohmann::json::array()},
    };

    const std::string meta = Platform::PathToUtf8(metadata.metaPath);
    if (Platform::WriteFileAtomic(meta, json.dump(2) + "\n") != Result::SUCCESS) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to write SigMF metadata '{}'.", meta);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
