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
    bytesWritten = 0;
    filePath.clear();

    if (!filepath.empty()) {
        filePath = std::filesystem::path(filepath);

        std::error_code ec;
        if (std::filesystem::exists(filePath, ec) && !ec) {
            bytesWritten = std::filesystem::file_size(filePath, ec);
            if (ec) {
                bytesWritten = 0;
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
        JST_WARN("[MODULE_FILE_WRITER] Parent directory '{}' does not exist.", parentPath.string());
        return Result::INCOMPLETE;
    }

    if (std::filesystem::exists(filePath) && !overwrite) {
        JST_ERROR("[MODULE_FILE_WRITER] File '{}' already exists.",
                  filePath.string());
        return Result::ERROR;
    }

    dataFile.open(filePath, std::ios::out | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_ERROR("[MODULE_FILE_WRITER] Failed to open '{}' for writing.",
                  filePath.string());
        return Result::ERROR;
    }

    bytesWritten = 0;

    JST_INFO("[MODULE_FILE_WRITER] Opened '{}' for writing.", filePath.string());

    return Result::SUCCESS;
}

Result FileWriterImpl::destroy() {
    if (dataFile.is_open()) {
        dataFile.close();
        JST_INFO("[MODULE_FILE_WRITER] Closed '{}' ({} bytes written).",
                 filePath.string(),
                 bytesWritten);
    }

    return Result::SUCCESS;
}

U64 FileWriterImpl::getBytesWritten() const {
    return bytesWritten;
}

}  // namespace Jetstream::Modules
