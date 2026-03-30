#include "module_impl.hh"

namespace Jetstream::Modules {

Result FileReaderImpl::validate() {
    const auto& config = *candidate();

    if (config.fileFormat != "raw") {
        JST_ERROR("[MODULE_FILE_READER] Invalid file format '{}'.", config.fileFormat);
        return Result::ERROR;
    }

    const bool isFloatType = config.dataType == "CF32" ||
                             config.dataType == "F32";
    const bool isI8Type = config.dataType == "CI8" ||
                          config.dataType == "I8";
    const bool isU8Type = config.dataType == "CU8" ||
                          config.dataType == "U8";
    const bool isI16Type = config.dataType == "CI16" ||
                           config.dataType == "I16";
    const bool isU16Type = config.dataType == "CU16" ||
                           config.dataType == "U16";

    if (!isFloatType && !isI8Type && !isU8Type && !isI16Type && !isU16Type) {
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

    outputs()["signal"].produced(name(), "signal", buffer);
    fileSize.publish(0);
    currentPosition.publish(0);

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
    const U64 inputFileSize = std::filesystem::file_size(filePath, ec);
    if (ec) {
        JST_WARN("[MODULE_FILE_READER] Failed to get file size for '{}'.", filePath.string());
        return Result::INCOMPLETE;
    }
    fileSize.publish(inputFileSize);

    dataFile.open(filePath, std::ios::in | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_WARN("[MODULE_FILE_READER] Failed to open '{}' for reading.", filePath.string());
        return Result::INCOMPLETE;
    }

    JST_INFO("[MODULE_FILE_READER] Opened '{}' ({} bytes).", filePath.string(), fileSize.get());

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

U64 FileReaderImpl::getCurrentPosition() const {
    return currentPosition.get();
}

U64 FileReaderImpl::getFileSize() const {
    return fileSize.get();
}

}  // namespace Jetstream::Modules
