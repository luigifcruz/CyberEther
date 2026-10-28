#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "jetstream/platform.hh"

namespace Jetstream::Modules {

namespace {

const std::unordered_map<std::string_view, DataType> kSigmfDataTypes = {
    {"cf64_le", DataType::CF64},
    {"rf64_le", DataType::F64},
    {"cf32_le", DataType::CF32},
    {"rf32_le", DataType::F32},
    {"ci16_le", DataType::CI16},
    {"ri16_le", DataType::I16},
    {"cu16_le", DataType::CU16},
    {"ru16_le", DataType::U16},
    {"ci8", DataType::CI8},
    {"ri8", DataType::I8},
    {"cu8", DataType::CU8},
    {"ru8", DataType::U8},
};

}  // namespace

Result FileReaderImpl::loadSigmf(Recording& result) const {
    const auto path = Platform::PathFromUtf8(filepath);
    const auto extension = path.extension();

    if (extension == ".sigmf") {
        JST_ERROR("[MODULE_FILE_READER] SigMF archives are not supported. Extract '{}' and "
                  "pick its .sigmf-meta file.", filepath);
        return Result::ERROR;
    }

    auto base = path;
    if (extension == ".sigmf-meta" || extension == ".sigmf-data") {
        base = path.parent_path() / path.stem();
    }

    auto metaPath = base;
    metaPath += ".sigmf-meta";
    const std::string meta = Platform::PathToUtf8(metaPath);

    std::ifstream file(metaPath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        JST_ERROR("[MODULE_FILE_READER] Failed to open SigMF metadata '{}'.", meta);
        return Result::ERROR;
    }

    const auto json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        JST_ERROR("[MODULE_FILE_READER] SigMF metadata '{}' is not valid JSON.", meta);
        return Result::ERROR;
    }

    const auto global = json.find("global");
    if (global == json.end() || !global->is_object()) {
        JST_ERROR("[MODULE_FILE_READER] SigMF metadata '{}' has no global object.", meta);
        return Result::ERROR;
    }

    const auto datatype = global->find("core:datatype");
    if (datatype == global->end() || !datatype->is_string()) {
        JST_ERROR("[MODULE_FILE_READER] SigMF metadata '{}' has no core:datatype.", meta);
        return Result::ERROR;
    }

    const auto& datatypeName = datatype->get_ref<const std::string&>();
    const auto type = kSigmfDataTypes.find(datatypeName);
    if (type == kSigmfDataTypes.end()) {
        JST_ERROR("[MODULE_FILE_READER] SigMF data type '{}' is not supported.", datatypeName);
        return Result::ERROR;
    }

    Recording recording;
    recording.dataType = type->second;

    const auto channels = global->find("core:num_channels");
    if (channels != global->end() &&
        (!channels->is_number_unsigned() || channels->get<U64>() != 1)) {
        JST_ERROR("[MODULE_FILE_READER] SigMF recordings with more than one channel are "
                  "not supported.");
        return Result::ERROR;
    }

    const auto metadataOnly = global->find("core:metadata_only");
    if (metadataOnly != global->end() && metadataOnly->is_boolean() && metadataOnly->get<bool>()) {
        JST_ERROR("[MODULE_FILE_READER] SigMF recording '{}' holds metadata only.", meta);
        return Result::ERROR;
    }

    const auto sampleRate = global->find("core:sample_rate");
    if (sampleRate != global->end()) {
        if (!sampleRate->is_number() || !std::isfinite(sampleRate->get<F64>()) ||
            sampleRate->get<F64>() <= 0.0) {
            JST_ERROR("[MODULE_FILE_READER] SigMF metadata '{}' has an invalid "
                      "core:sample_rate.", meta);
            return Result::ERROR;
        }
        recording.sampleRate = sampleRate->get<F64>();
    }

    const auto trailing = global->find("core:trailing_bytes");
    if (trailing != global->end() && trailing->is_number_unsigned()) {
        recording.trailingBytes = trailing->get<U64>();
    }

    const auto dataset = global->find("core:dataset");
    if (dataset != global->end() && dataset->is_string()) {
        recording.dataPath = metaPath.parent_path() /
                             Platform::PathFromUtf8(dataset->get<std::string>());
    } else {
        recording.dataPath = base;
        recording.dataPath += ".sigmf-data";
    }

    const auto captures = json.find("captures");
    if (captures != json.end() && captures->is_array()) {
        for (const auto& entry : *captures) {
            if (!entry.is_object()) {
                continue;
            }

            Recording::Capture capture;
            const auto start = entry.find("core:sample_start");
            if (start != entry.end() && start->is_number_unsigned()) {
                capture.sampleStart = start->get<U64>();
            }

            const auto frequency = entry.find("core:frequency");
            if (frequency != entry.end() && frequency->is_number() &&
                std::isfinite(frequency->get<F64>())) {
                capture.frequency = frequency->get<F64>();
            }

            const auto header = entry.find("core:header_bytes");
            if (header != entry.end() && header->is_number_unsigned() && header->get<U64>() > 0) {
                if (!recording.captures.empty()) {
                    JST_ERROR("[MODULE_FILE_READER] SigMF header bytes are only supported on "
                              "the first capture.");
                    return Result::ERROR;
                }
                recording.headerBytes = header->get<U64>();
            }

            recording.captures.push_back(capture);
        }
    }

    std::stable_sort(recording.captures.begin(), recording.captures.end(),
                     [](const auto& a, const auto& b) { return a.sampleStart < b.sampleStart; });

    JST_INFO("[MODULE_FILE_READER] Loaded SigMF metadata '{}' ({}).", meta, datatypeName);

    result = std::move(recording);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
