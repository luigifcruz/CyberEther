#include "module_impl.hh"

#include <cstring>

#include "jetstream/memory/axis.hh"
#include "jetstream/platform.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/tools/numeric.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kStreamPollMilliseconds = 200;

}  // namespace

Result FileReaderImpl::validate() {
    const auto& config = *candidate();
    validatedDataType = DataType::None;
    validatedOutputSizeBytes = 0;

    if (config.fileFormat != "raw" && config.fileFormat != "sigmf") {
        JST_ERROR("[MODULE_FILE_READER] Invalid file format '{}'.", config.fileFormat);
        return Result::ERROR;
    }

    const DataType dataType = NameToDataType(config.dataType);
    switch (dataType) {
        case DataType::CF64:
        case DataType::F64:
        case DataType::CF32:
        case DataType::F32:
        case DataType::CI8:
        case DataType::I8:
        case DataType::CU8:
        case DataType::U8:
        case DataType::CI16:
        case DataType::I16:
        case DataType::CU16:
        case DataType::U16:
            break;
        default:
            JST_ERROR("[MODULE_FILE_READER] Invalid data type '{}'.", config.dataType);
            return Result::ERROR;
    }

    if (config.batchSize == 0) {
        JST_ERROR("[MODULE_FILE_READER] Batch size cannot be zero.");
        return Result::ERROR;
    }

    const DataType sizingType = config.fileFormat == "sigmf" ? DataType::CF64 : dataType;
    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(config.batchSize,
                                 static_cast<U64>(DataTypeSize(sizingType)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_FILE_READER] Output byte size exceeds the supported "
                  "range.");
        return Result::ERROR;
    }

    validatedDataType = dataType;
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result FileReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result FileReaderImpl::create() {
    livePlaying = playing;
    liveRealTime = realTime;
    recording = {};
    recording.dataPath = Platform::PathFromUtf8(filepath);
    recording.dataType = validatedDataType;
    publishedFrequency.reset();

    Result loaded = Result::SUCCESS;
    if (fileFormat == "sigmf" && !filepath.empty()) {
        Recording sigmf;
        loaded = loadSigmf(sigmf);
        if (loaded == Result::SUCCESS) {
            recording = std::move(sigmf);
        }
    }

    JST_CHECK(buffer.create(device(), recording.dataType, {batchSize}));
    JST_CHECK(SetSignalAxes(buffer, {
        .sample = Index{0},
    }));

    outputs()["signal"].produced(name(), "signal", buffer);
    fileSize.publish(0);
    currentPosition.publish(0);
    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (filepath.empty()) {
        JST_ERROR("[MODULE_FILE_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    if (loaded != Result::SUCCESS) {
        return Result::INCOMPLETE;
    }

    if (recording.sampleRate) {
        JST_CHECK(buffer.setAttribute("sampleRate", static_cast<F32>(*recording.sampleRate)));
    } else if (fileFormat == "sigmf" && realTime) {
        JST_WARN("[MODULE_FILE_READER] Recording has no sample rate, so Real Time is ignored.");
    }
    pacedSamples = 0;
    JST_CHECK(publishCapture(0));

    filePath = recording.dataPath;
    const std::string dataPath = Platform::PathToUtf8(filePath);

    std::error_code ec;
    const auto status = std::filesystem::status(filePath, ec);
    if (!std::filesystem::exists(status)) {
        JST_ERROR("[MODULE_FILE_READER] File '{}' does not exist.", dataPath);
        return Result::INCOMPLETE;
    }

    if (std::filesystem::is_directory(status)) {
        JST_ERROR("[MODULE_FILE_READER] Path '{}' is a directory.", dataPath);
        return Result::INCOMPLETE;
    }

    if (!std::filesystem::is_regular_file(status)) {
        if (recording.headerBytes > 0 || recording.trailingBytes > 0) {
            JST_ERROR("[MODULE_FILE_READER] Header and trailing bytes are not supported when '{}' "
                      "is a stream.", dataPath);
            return Result::INCOMPLETE;
        }
        return createStream();
    }

    const U64 inputFileSize = std::filesystem::file_size(filePath, ec);
    if (ec) {
        JST_ERROR("[MODULE_FILE_READER] Failed to get file size for '{}'.", dataPath);
        return Result::INCOMPLETE;
    }
    if (inputFileSize < recording.headerBytes ||
        inputFileSize - recording.headerBytes < recording.trailingBytes) {
        JST_ERROR("[MODULE_FILE_READER] File '{}' is shorter than its {} header and {} trailing "
                  "bytes.", dataPath, recording.headerBytes, recording.trailingBytes);
        return Result::INCOMPLETE;
    }
    fileSize.publish(inputFileSize - recording.headerBytes - recording.trailingBytes);

    dataFile.open(filePath, std::ios::in | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_ERROR("[MODULE_FILE_READER] Failed to open '{}' for reading.", dataPath);
        return Result::INCOMPLETE;
    }
    dataFile.seekg(static_cast<std::streamoff>(recording.headerBytes), std::ios::beg);

    JST_INFO("[MODULE_FILE_READER] Opened '{}' ({} bytes).", dataPath, fileSize.get());

    return Result::SUCCESS;
}

Result FileReaderImpl::createStream() {
    std::string error;
    const std::string dataPath = Platform::PathToUtf8(filePath);
    if (stream.open(dataPath, error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_FILE_READER] Failed to open '{}' for reading ({}).", dataPath, error);
        return Result::INCOMPLETE;
    }

    streamFilled = 0;
    streamTotal = 0;
    streaming = true;
    starved = false;
    ended = false;
    errored = false;
    receiving = true;
    streamThread.start([this]() { streamLoop(); });

    JST_INFO("[MODULE_FILE_READER] Opened stream '{}'.", dataPath);

    return Result::SUCCESS;
}

Result FileReaderImpl::readStream() {
    if (errored) {
        return Result::ERROR;
    }

    if (ended) {
        return Result::YIELD;
    }

    auto* data = static_cast<U8*>(buffer.data());
    const U64 size = buffer.sizeBytes();

    while (streamFilled < size) {
        U64 received = 0;
        const Result result = stream.read(data + streamFilled, size - streamFilled, received);

        if (result == Result::SUCCESS) {
            streamFilled += received;
            streamTotal += received;
            continue;
        }

        if (result == Result::TIMEOUT) {
            {
                std::lock_guard lock(streamMutex);
                starved = true;
            }
            streamCondition.notify_one();
            return Result::YIELD;
        }

        if (result == Result::INCOMPLETE) {
            ended = true;
            JST_INFO("[MODULE_FILE_READER] Reached the end of stream '{}'.", Platform::PathToUtf8(filePath));
            if (streamFilled == 0) {
                return Result::YIELD;
            }
            std::memset(data + streamFilled, 0, size - streamFilled);
            break;
        }

        JST_ERROR("[MODULE_FILE_READER] Failed to read from stream '{}'.", Platform::PathToUtf8(filePath));
        errored = true;
        return Result::ERROR;
    }

    JST_CHECK(publishCapture((streamTotal - streamFilled) / buffer.elementSize()));
    currentPosition.publish(streamTotal);
    updateThroughput(streamFilled);
    streamFilled = 0;

    return Result::SUCCESS;
}

void FileReaderImpl::streamLoop() {
    while (receiving) {
        {
            std::unique_lock lock(streamMutex);
            streamCondition.wait(lock, [this]() { return starved || !receiving; });
        }

        if (!receiving) {
            return;
        }

        if (stream.wait(kStreamPollMilliseconds) == Result::TIMEOUT) {
            continue;
        }

        starved = false;
        Scheduler::Context::NotifyPendingCompute();
    }
}

Result FileReaderImpl::destroy() {
    {
        std::lock_guard lock(streamMutex);
        receiving = false;
    }
    streamCondition.notify_all();
    if (streamThread.joinable()) {
        streamThread.join();
    }
    stream.close();
    streaming = false;

    if (dataFile.is_open()) {
        dataFile.close();
    }

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

Result FileReaderImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.filepath == filepath &&
        config.fileFormat == fileFormat &&
        config.dataType == dataType &&
        config.batchSize == batchSize) {
        loop = config.loop;
        playing = config.playing;
        realTime = config.realTime;
        livePlaying = playing;
        liveRealTime = realTime;
        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

Result FileReaderImpl::publishCapture(const U64 sampleIndex) {
    if (recording.captures.empty()) {
        return Result::SUCCESS;
    }

    F64 frequency = 0.0;
    for (const auto& capture : recording.captures) {
        if (capture.sampleStart > sampleIndex) {
            break;
        }
        frequency = capture.frequency.value_or(0.0);
    }

    if (publishedFrequency == frequency) {
        return Result::SUCCESS;
    }

    JST_CHECK(buffer.setAttribute("frequency", static_cast<F32>(frequency)));
    publishedFrequency = frequency;

    return Result::SUCCESS;
}

bool FileReaderImpl::isStream() const {
    return streaming;
}

U64 FileReaderImpl::getCurrentPosition() const {
    return currentPosition.get();
}

U64 FileReaderImpl::getFileSize() const {
    return fileSize.get();
}

F32 FileReaderImpl::getThroughput() const {
    return throughputMBs.get();
}

void FileReaderImpl::updateThroughput(const U64 deltaBytes) {
    constexpr double kMeasurementPeriodSeconds = 0.10;
    constexpr double kEmaAlpha = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsedSeconds < kMeasurementPeriodSeconds) {
        return;
    }

    const double instantThroughput = static_cast<double>(bytesSinceLastMeasurement) /
                                     1e6 /
                                     elapsedSeconds;
    const double smoothedThroughput = kEmaAlpha * instantThroughput +
                                      (1.0 - kEmaAlpha) * throughputMBs.get();
    throughputMBs.publish(static_cast<F32>(smoothedThroughput));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
