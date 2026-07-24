#include "module_impl.hh"

#include "jetstream/platform.hh"
#include "jetstream/tools/numeric.hh"

namespace Jetstream::Modules {

Result FileReaderImpl::validate() {
    const auto& config = *candidate();
    validatedDataType = DataType::None;
    validatedOutputSizeBytes = 0;

    if (config.fileFormat != "raw") {
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

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(config.batchSize,
                                 static_cast<U64>(DataTypeSize(dataType)),
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
    JST_CHECK(buffer.create(device(), validatedDataType, {batchSize}));

    outputs()["signal"].produced(name(), "signal", buffer);
    fileSize.publish(0);
    currentPosition.publish(0);
    currentBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (filepath.empty()) {
        JST_WARN("[MODULE_FILE_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    filePath = Platform::PathFromUtf8(filepath);

    if (!std::filesystem::exists(filePath)) {
        JST_WARN("[MODULE_FILE_READER] File '{}' does not exist.", filepath);
        return Result::INCOMPLETE;
    }

    std::error_code ec;
    const U64 inputFileSize = std::filesystem::file_size(filePath, ec);
    if (ec) {
        JST_WARN("[MODULE_FILE_READER] Failed to get file size for '{}'.", filepath);
        return Result::INCOMPLETE;
    }
    fileSize.publish(inputFileSize);

    dataFile.open(filePath, std::ios::in | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_WARN("[MODULE_FILE_READER] Failed to open '{}' for reading.", filepath);
        return Result::INCOMPLETE;
    }

    JST_INFO("[MODULE_FILE_READER] Opened '{}' ({} bytes).", filepath, fileSize.get());

    return Result::SUCCESS;
}

Result FileReaderImpl::destroy() {
    if (dataFile.is_open()) {
        dataFile.close();
    }

    currentBandwidth.publish(0.0f);
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
        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

U64 FileReaderImpl::getCurrentPosition() const {
    return currentPosition.get();
}

U64 FileReaderImpl::getFileSize() const {
    return fileSize.get();
}

F32 FileReaderImpl::getCurrentBandwidth() const {
    return currentBandwidth.get();
}

void FileReaderImpl::updateBandwidth(const U64 deltaBytes) {
    constexpr double kBandwidthMeasurementPeriodSeconds = 0.10;
    constexpr double kBandwidthEmaAlpha = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsedSeconds < kBandwidthMeasurementPeriodSeconds) {
        return;
    }

    const double instantBandwidth = static_cast<double>(bytesSinceLastMeasurement) /
                                    static_cast<double>(JST_MB) /
                                    elapsedSeconds;
    const double smoothedBandwidth = kBandwidthEmaAlpha * instantBandwidth +
                                     (1.0 - kBandwidthEmaAlpha) * currentBandwidth.get();
    currentBandwidth.publish(static_cast<F32>(smoothedBandwidth));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
