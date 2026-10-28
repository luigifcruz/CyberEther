#include "format.hh"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Modules {

namespace {

constexpr U16 kWavePcm = 1;
constexpr U16 kWaveFloat = 3;
constexpr U16 kWaveExtensible = 0xFFFE;
constexpr U32 kUnknownSize = 0xFFFFFFFF;

template<typename T>
T LittleEndian(const U8* bytes) {
    T value = 0;
    for (U64 i = 0; i < sizeof(T); ++i) {
        value |= static_cast<T>(bytes[i]) << (8 * i);
    }
    return value;
}

std::optional<DataType> WaveDataType(const U16 format, const U16 channels, const U16 bits) {
    const bool complex = channels == 2;
    if (format == kWavePcm && bits == 8) {
        return complex ? DataType::CU8 : DataType::U8;
    }
    if (format == kWavePcm && bits == 16) {
        return complex ? DataType::CI16 : DataType::I16;
    }
    if (format == kWaveFloat && bits == 32) {
        return complex ? DataType::CF32 : DataType::F32;
    }
    if (format == kWaveFloat && bits == 64) {
        return complex ? DataType::CF64 : DataType::F64;
    }
    return std::nullopt;
}

}  // namespace

Result FileReaderFormat::Wav::load(const std::string& filepath, Recording& result) const {
    const auto path = Platform::PathFromUtf8(filepath);

    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        JST_ERROR("[MODULE_FILE_READER] WAV needs a regular file, and '{}' is not one.", filepath);
        return Result::ERROR;
    }

    const U64 size = std::filesystem::file_size(path, ec);
    if (ec) {
        JST_ERROR("[MODULE_FILE_READER] Failed to get file size for '{}'.", filepath);
        return Result::ERROR;
    }

    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        JST_ERROR("[MODULE_FILE_READER] Failed to open '{}' for reading.", filepath);
        return Result::ERROR;
    }

    std::array<U8, 12> riff{};
    file.read(reinterpret_cast<char*>(riff.data()), riff.size());
    const bool rf64 = std::memcmp(riff.data(), "RF64", 4) == 0 ||
                      std::memcmp(riff.data(), "BW64", 4) == 0;
    if (!file || (std::memcmp(riff.data(), "RIFF", 4) != 0 && !rf64) ||
        std::memcmp(riff.data() + 8, "WAVE", 4) != 0) {
        JST_ERROR("[MODULE_FILE_READER] File '{}' is not a WAV file.", filepath);
        return Result::ERROR;
    }

    Recording recording;
    recording.dataPath = path;

    std::optional<U64> longDataSize;
    bool hasFormat = false;
    U64 offset = riff.size();

    while (offset + 8 <= size) {
        std::array<U8, 8> header{};
        file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        file.read(reinterpret_cast<char*>(header.data()), header.size());
        if (!file) {
            break;
        }

        const U64 chunkStart = offset + header.size();
        const U64 chunkSize = LittleEndian<U32>(header.data() + 4);

        if (std::memcmp(header.data(), "data", 4) == 0) {
            if (!hasFormat) {
                break;
            }

            U64 dataSize = chunkSize;
            if (rf64 && chunkSize == kUnknownSize && longDataSize) {
                dataSize = *longDataSize;
            }

            const U64 available = size - chunkStart;
            if (dataSize == 0 || dataSize == kUnknownSize || dataSize > available) {
                dataSize = available;
            }

            recording.headerBytes = chunkStart;
            recording.trailingBytes = available - dataSize;

            JST_INFO("[MODULE_FILE_READER] Loaded WAV header '{}' ({}).",
                     filepath, DataTypeToName(recording.dataType));

            result = std::move(recording);
            return Result::SUCCESS;
        }

        std::vector<U8> body(std::min<U64>(chunkSize, 64));
        file.read(reinterpret_cast<char*>(body.data()), static_cast<std::streamsize>(body.size()));
        if (!file) {
            break;
        }

        if (std::memcmp(header.data(), "ds64", 4) == 0 && body.size() >= 16) {
            longDataSize = LittleEndian<U64>(body.data() + 8);
        }

        if (std::memcmp(header.data(), "fmt ", 4) == 0 && body.size() >= 16) {
            U16 format = LittleEndian<U16>(body.data());
            const U16 channels = LittleEndian<U16>(body.data() + 2);
            const U32 sampleRate = LittleEndian<U32>(body.data() + 4);
            const U16 bits = LittleEndian<U16>(body.data() + 14);
            if (format == kWaveExtensible && body.size() >= 26) {
                format = LittleEndian<U16>(body.data() + 24);
            }

            const auto type = WaveDataType(format, channels, bits);
            if (!type || (channels != 1 && channels != 2)) {
                JST_ERROR("[MODULE_FILE_READER] WAV file '{}' has {} channels of {}-bit format "
                          "{}, which is not supported.", filepath, channels, bits, format);
                return Result::ERROR;
            }

            recording.dataType = *type;
            if (sampleRate > 0) {
                recording.sampleRate = sampleRate;
            }
            hasFormat = true;
        }

        if (std::memcmp(header.data(), "auxi", 4) == 0 && body.size() >= 36) {
            const U32 frequency = LittleEndian<U32>(body.data() + 32);
            if (frequency > 0) {
                recording.captures = {{.sampleStart = 0, .frequency = frequency}};
            }
        }

        offset = chunkStart + chunkSize + (chunkSize & 1);
    }

    JST_ERROR("[MODULE_FILE_READER] WAV file '{}' has no {} chunk.", filepath,
              hasFormat ? "data" : "format");
    return Result::ERROR;
}

}  // namespace Jetstream::Modules
