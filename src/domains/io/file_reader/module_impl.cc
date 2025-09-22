#include "module_impl.hh"

namespace Jetstream::Modules {

Result FileReaderImpl::validate() {
    const auto& config = *candidate();

    if (config.fileFormat != "raw") {
        JST_ERROR("[MODULE_FILE_READER] Invalid file format '{}'.", config.fileFormat);
        return Result::ERROR;
    }

    if (config.dataType != "CF32" &&
        config.dataType != "F32" &&
        config.dataType != "I16" &&
        config.dataType != "I8" &&
        config.dataType != "U8") {
        JST_ERROR("[MODULE_FILE_READER] Invalid data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    if (config.batchSize == 0) {
        JST_ERROR("[MODULE_FILE_READER] Batch size cannot be zero.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FileReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result FileReaderImpl::create() {
    JST_CHECK(buffer.create(device(), NameToDataType(dataType), {batchSize}));

    outputs()["signal"] = {name(), "signal", buffer};

    if (filepath.empty()) {
        JST_WARN("[MODULE_FILE_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    filePath = std::filesystem::path(filepath);

    if (!std::filesystem::exists(filePath)) {
        JST_WARN("[MODULE_FILE_READER] File '{}' does not exist.", filePath.string());
        return Result::INCOMPLETE;
    }

    std::error_code ec;
    fileSize = std::filesystem::file_size(filePath, ec);
    if (ec) {
        JST_WARN("[MODULE_FILE_READER] Failed to get file size for '{}'.", filePath.string());
        return Result::INCOMPLETE;
    }

    dataFile.open(filePath, std::ios::in | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_WARN("[MODULE_FILE_READER] Failed to open '{}' for reading.", filePath.string());
        return Result::INCOMPLETE;
    }

    currentPosition = 0;

    JST_INFO("[MODULE_FILE_READER] Opened '{}' ({} bytes).", filePath.string(), fileSize);

    return Result::SUCCESS;
}

Result FileReaderImpl::destroy() {
    if (dataFile.is_open()) {
        dataFile.close();
    }

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

const U64& FileReaderImpl::getCurrentPosition() const {
    return currentPosition;
}

const U64& FileReaderImpl::getFileSize() const {
    return fileSize;
}

}  // namespace Jetstream::Modules
