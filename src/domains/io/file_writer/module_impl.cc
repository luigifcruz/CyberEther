#include "module_impl.hh"

#include <algorithm>

#include "jetstream/platform.hh"

namespace Jetstream::Modules {

Result FileWriterImpl::validate() {
    const auto& config = *candidate();

    if (!FileWriterFormat::Create(config.fileFormat)) {
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
    throughputMBs.publish(0.0f);
    filePath.clear();
    format = FileWriterFormat::Create(fileFormat);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (!filepath.empty()) {
        filePath = format->resolve(Platform::PathFromUtf8(filepath));

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

    const std::string dataPath = Platform::PathToUtf8(filePath);

    auto parentPath = filePath.parent_path();
    if (!parentPath.empty() && !std::filesystem::exists(parentPath)) {
        JST_WARN("[MODULE_FILE_WRITER] Parent directory for '{}' does not exist.", dataPath);
        return Result::INCOMPLETE;
    }

    const auto companions = format->companions();
    const bool companionExists = std::any_of(companions.begin(), companions.end(), [](const auto& path) {
        return std::filesystem::exists(path);
    });
    if ((std::filesystem::exists(filePath) || companionExists) && !overwrite) {
        JST_ERROR("[MODULE_FILE_WRITER] File '{}' already exists.", dataPath);
        return Result::ERROR;
    }

    JST_CHECK(format->check(inputs().at("buffer").tensor));

    dataFile.open(filePath, std::ios::out | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to open '{}' for writing.", dataPath);
        return Result::ERROR;
    }

    bytesWritten.publish(0);

    JST_CHECK(format->start(dataFile, inputs().at("buffer").tensor));

    JST_INFO("[MODULE_FILE_WRITER] Opened '{}' for writing.", dataPath);

    return Result::SUCCESS;
}

Result FileWriterImpl::destroy() {
    if (dataFile.is_open()) {
        format->finish(dataFile, bytesWritten.get());
        dataFile.close();
        JST_INFO("[MODULE_FILE_WRITER] Closed '{}' ({} bytes written).",
                 Platform::PathToUtf8(filePath), bytesWritten.get());
    }

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

U64 FileWriterImpl::getBytesWritten() const {
    return bytesWritten.get();
}

F32 FileWriterImpl::getThroughput() const {
    return throughputMBs.get();
}

void FileWriterImpl::updateThroughput(const U64 deltaBytes) {
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
