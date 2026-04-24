#include "module_impl.hh"

namespace Jetstream::Modules {

Result FileWriterImpl::validate() {
    const auto& config = *candidate();

    if (config.fileFormat != "raw") {
        JST_ERROR("[MODULE_FILE_WRITER] Invalid file format '{}'.",
                  config.fileFormat);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FileWriterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result FileWriterImpl::create() {
    bytesWritten.publish(0);
    currentBandwidth.publish(0.0f);
    filePath.clear();
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (!filepath.empty()) {
        filePath = std::filesystem::u8path(filepath);

        std::error_code ec;
        if (std::filesystem::exists(filePath, ec) && !ec) {
            bytesWritten.publish(std::filesystem::file_size(filePath, ec));
            if (ec) {
                bytesWritten.publish(0);
            }
        }
    }

    if (!recording) {
        return Result::SUCCESS;
    }

    if (filepath.empty()) {
        JST_WARN("[MODULE_FILE_WRITER] File path is empty.");
        return Result::INCOMPLETE;
    }

    auto parentPath = filePath.parent_path();
    if (!parentPath.empty() && !std::filesystem::exists(parentPath)) {
        JST_WARN("[MODULE_FILE_WRITER] Parent directory for '{}' does not exist.", filepath);
        return Result::INCOMPLETE;
    }

    if (std::filesystem::exists(filePath) && !overwrite) {
        JST_ERROR("[MODULE_FILE_WRITER] File '{}' already exists.", filepath);
        return Result::ERROR;
    }

    dataFile.open(filePath, std::ios::out | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to open '{}' for writing.", filepath);
        return Result::ERROR;
    }

    bytesWritten.publish(0);

    JST_INFO("[MODULE_FILE_WRITER] Opened '{}' for writing.", filepath);

    return Result::SUCCESS;
}

Result FileWriterImpl::destroy() {
    if (dataFile.is_open()) {
        dataFile.close();
        JST_INFO("[MODULE_FILE_WRITER] Closed '{}' ({} bytes written).", filepath, bytesWritten.get());
    }

    currentBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

U64 FileWriterImpl::getBytesWritten() const {
    return bytesWritten.get();
}

F32 FileWriterImpl::getCurrentBandwidth() const {
    return currentBandwidth.get();
}

void FileWriterImpl::updateBandwidth(const U64 deltaBytes) {
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
