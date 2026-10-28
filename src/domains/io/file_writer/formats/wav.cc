#include "format.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "jetstream/logger.hh"

namespace Jetstream::Modules {

namespace {

constexpr U16 kWavePcm = 1;
constexpr U16 kWaveFloat = 3;
constexpr U32 kUnknownSize = 0xFFFFFFFF;
constexpr U32 kDs64Bytes = 28;
constexpr U32 kAuxiBytes = 68;

template<typename T>
void PutLittleEndian(std::string& out, const T value) {
    for (U64 i = 0; i < sizeof(T); ++i) {
        out.push_back(static_cast<char>((static_cast<U64>(value) >> (8 * i)) & 0xFF));
    }
}

void PutChunk(std::string& out, const char* id, const U32 size) {
    out.append(id, 4);
    PutLittleEndian<U32>(out, size);
}

void PutSystemTime(std::string& out, const std::chrono::system_clock::time_point time) {
    using namespace std::chrono;
    const auto day = floor<days>(time);
    const year_month_day date{day};
    const hh_mm_ss clock{floor<milliseconds>(time - day)};
    PutLittleEndian<U16>(out, static_cast<U16>(static_cast<int>(date.year())));
    PutLittleEndian<U16>(out, static_cast<U16>(static_cast<unsigned>(date.month())));
    PutLittleEndian<U16>(out, static_cast<U16>(weekday{day}.c_encoding()));
    PutLittleEndian<U16>(out, static_cast<U16>(static_cast<unsigned>(date.day())));
    PutLittleEndian<U16>(out, static_cast<U16>(clock.hours().count()));
    PutLittleEndian<U16>(out, static_cast<U16>(clock.minutes().count()));
    PutLittleEndian<U16>(out, static_cast<U16>(clock.seconds().count()));
    PutLittleEndian<U16>(out, static_cast<U16>(clock.subseconds().count()));
}

std::optional<U32> FitsU32(const std::optional<F64>& value) {
    if (!value || !std::isfinite(*value) || *value <= 0.0 ||
        *value > static_cast<F64>(std::numeric_limits<U32>::max())) {
        return std::nullopt;
    }
    return static_cast<U32>(std::llround(*value));
}

std::optional<U16> WaveFormat(const DataType dtype) {
    switch (dtype) {
        case DataType::CF32:
        case DataType::F32:
            return kWaveFloat;
        case DataType::CI16:
        case DataType::I16:
        case DataType::CU8:
        case DataType::U8:
            return kWavePcm;
        default:
            return std::nullopt;
    }
}

}  // namespace

Result FileWriterFormat::Wav::check(const Tensor& input) const {
    if (!WaveFormat(input.dtype())) {
        JST_ERROR("[MODULE_FILE_WRITER] WAV does not support the input data type.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result FileWriterFormat::Wav::start(std::ofstream& file, const Tensor& input) {
    const DataType dtype = input.dtype();
    format = WaveFormat(dtype).value_or(kWavePcm);
    channels = dtype == DataType::CF32 || dtype == DataType::CI16 || dtype == DataType::CU8 ? 2 : 1;
    bits = static_cast<U16>(8 * input.elementSize() / channels);
    sampleRate = NumericAttribute(input, "sampleRate");
    frequency = FitsU32(NumericAttribute(input, "frequency"));
    startTime = std::chrono::system_clock::now();

    const std::string placeholder = header(std::nullopt);
    file.write(placeholder.data(), static_cast<std::streamsize>(placeholder.size()));
    if (!file.good()) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to write the WAV header.");
        return Result::ERROR;
    }

    started = true;
    return Result::SUCCESS;
}

Result FileWriterFormat::Wav::track(const Tensor& input, U64) {
    if (!sampleRate) {
        sampleRate = NumericAttribute(input, "sampleRate");
    }
    return Result::SUCCESS;
}

Result FileWriterFormat::Wav::finish(std::ofstream& file, const U64 dataBytes) {
    if (!started) {
        return Result::SUCCESS;
    }

    if (dataBytes & 1) {
        file.put(0);
    }

    if (!FitsU32(sampleRate)) {
        JST_WARN("[MODULE_FILE_WRITER] Input has no sample rate, so the WAV header stores 0 Hz.");
    }

    const std::string complete = header(dataBytes);
    file.seekp(0, std::ios::beg);
    file.write(complete.data(), static_cast<std::streamsize>(complete.size()));
    if (!file.good()) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to finish the WAV header.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

std::string FileWriterFormat::Wav::header(const std::optional<U64> dataBytes) const {
    const U16 blockAlign = channels * bits / 8;
    const U32 fmtBytes = format == kWaveFloat ? 18 : 16;
    const U32 rate = FitsU32(sampleRate).value_or(0);

    const bool floating = format == kWaveFloat;
    const U64 headerBytes = 12 + 8 + kDs64Bytes + 8 + fmtBytes + (floating ? 8 + 4 : 0) +
                            (frequency ? 8 + kAuxiBytes : 0) + 8;
    const U64 riffBytes = dataBytes ? headerBytes - 8 + *dataBytes + (*dataBytes & 1) : 0;
    const bool large = dataBytes && riffBytes > kUnknownSize - 1;

    std::string out;
    out.reserve(headerBytes);

    PutChunk(out, large ? "RF64" : "RIFF", dataBytes && !large ? riffBytes : kUnknownSize);
    out.append("WAVE", 4);

    PutChunk(out, large ? "ds64" : "JUNK", kDs64Bytes);
    PutLittleEndian<U64>(out, large ? riffBytes : 0);
    PutLittleEndian<U64>(out, large ? *dataBytes : 0);
    PutLittleEndian<U64>(out, large ? *dataBytes / blockAlign : 0);
    PutLittleEndian<U32>(out, 0);

    PutChunk(out, "fmt ", fmtBytes);
    PutLittleEndian<U16>(out, format);
    PutLittleEndian<U16>(out, channels);
    PutLittleEndian<U32>(out, rate);
    PutLittleEndian<U32>(out, static_cast<U32>(std::min<U64>(static_cast<U64>(rate) * blockAlign, kUnknownSize)));
    PutLittleEndian<U16>(out, blockAlign);
    PutLittleEndian<U16>(out, bits);
    if (fmtBytes == 18) {
        PutLittleEndian<U16>(out, 0);
    }

    if (floating) {
        PutChunk(out, "fact", 4);
        PutLittleEndian<U32>(out, dataBytes && !large ? static_cast<U32>(*dataBytes / blockAlign) : kUnknownSize);
    }

    if (frequency) {
        PutChunk(out, "auxi", kAuxiBytes);
        PutSystemTime(out, startTime);
        if (dataBytes) {
            PutSystemTime(out, std::chrono::system_clock::now());
        } else {
            out.append(16, '\0');
        }
        PutLittleEndian<U32>(out, *frequency);
        PutLittleEndian<U32>(out, rate);
        out.append(kAuxiBytes - 40, '\0');
    }

    PutChunk(out, "data", dataBytes && !large ? static_cast<U32>(*dataBytes) : kUnknownSize);

    return out;
}

}  // namespace Jetstream::Modules
