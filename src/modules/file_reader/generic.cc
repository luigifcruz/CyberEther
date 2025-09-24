#include <regex>
#include <filesystem>
#include <fstream>

#include "jetstream/modules/file_reader.hh"

// TODO: Correctly separate Device implementations.

namespace Jetstream {

template<Device D, typename T>
struct FileReader<D, T>::GImpl {
    std::ifstream dataFile;
    std::filesystem::path filePath;
    U64 fileSize;
    U64 currentPosition;

    Result startPlaying(FileReader<D, T>& m);
    Result stopPlaying();

    Result underlyingStartPlaying();
    Result underlyingStopPlaying();
};

template<Device D, typename T>
Result FileReader<D, T>::create() {
    JST_DEBUG("Initializing File Reader module.");
    JST_INIT_IO();

    // Initialize state
    gimpl->fileSize = 0;
    gimpl->currentPosition = 0;

    // Check playback status.
    if (!config.playing) {
        JST_ERROR("Not playing.");
        return Result::ERROR;
    }

    // Start playback
    const auto& res = gimpl->startPlaying(*this);

    if (res != Result::SUCCESS) {
        config.playing = false;
        return res;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::destroy() {
    JST_DEBUG("Destroying File Reader module.");

    JST_CHECK(gimpl->stopPlaying());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::playing(const bool& playing) {
    if (playing) {
        if (!config.playing) {
            JST_CHECK(gimpl->startPlaying(*this));
            config.playing = true;
        }
    } else {
        if (config.playing) {
            config.playing = false;
            JST_CHECK(gimpl->stopPlaying());
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
U64 FileReader<D, T>::getFileSize() const {
    return gimpl->fileSize;
}

template<Device D, typename T>
U64 FileReader<D, T>::getCurrentPosition() const {
    return gimpl->currentPosition;
}

template<Device D, typename T>
Result FileReader<D, T>::GImpl::startPlaying(FileReader<D, T>& m) {
    // Initialize output buffer
    JST_CHECK(m.output.buffer.create(D, mem2::TypeToDataType<T>(), m.config.shape));
    if (m.output.buffer.size() == 0) {
        JST_ERROR("Buffer shape is zero.");
        return Result::ERROR;
    }

    // Check file format type.
    if (m.config.fileFormat != FileFormatType::Raw) {
        JST_ERROR("File format '{}' is not supported.", m.config.fileFormat);
        return Result::ERROR;
    }

    // Check if the provided filepath is valid.
    if (m.config.filepath.empty()) {
        JST_ERROR("File path is empty.");
        return Result::ERROR;
    }

    if (!std::regex_match(m.config.filepath, std::regex("^[a-zA-Z0-9_./-]*$"))) {
        JST_ERROR("File path '{}' contains invalid characters.", m.config.filepath);
        return Result::ERROR;
    }

    filePath = std::filesystem::path(m.config.filepath);

    // Check if file exists
    if (!std::filesystem::exists(filePath)) {
        JST_ERROR("File '{}' does not exist.", filePath.string());
        return Result::ERROR;
    }

    // Get file size
    std::error_code ec;
    fileSize = std::filesystem::file_size(filePath, ec);
    if (ec) {
        JST_ERROR("Failed to get file size for '{}'.", filePath.string());
        return Result::ERROR;
    }

    // Open the raw data file.
    dataFile.open(filePath, std::ios::in | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_ERROR("Failed to open file '{}' for reading.", filePath.string());
        return Result::ERROR;
    }

    // Start underlying recording.
    JST_CHECK(underlyingStartPlaying());

    currentPosition = 0;

    JST_INFO("Opened file '{}' for reading. Size: {} bytes.", filePath.string(), fileSize);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::GImpl::stopPlaying() {
    // Stop Playing.
    JST_INFO("Stopping playing.");

    if (dataFile.is_open()) {
        dataFile.close();
    }

    // Stop underlying playback.
    JST_CHECK(underlyingStopPlaying());

    return Result::SUCCESS;
}

template<Device D, typename T>
void FileReader<D, T>::info() const {
    JST_DEBUG("  File Format: Raw Binary");
    JST_DEBUG("  Filepath: {}", config.filepath);
    JST_DEBUG("  Playing: {}", config.playing);
    JST_DEBUG("  Loop: {}", config.loop);
    JST_DEBUG("  Shape: {}", config.shape);
}

}  // namespace Jetstream
