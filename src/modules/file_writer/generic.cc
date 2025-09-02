#include <regex>
#include <filesystem>
#include <fstream>


#include "jetstream/modules/file_writer.hh"

namespace Jetstream {

template<Device D, typename T>
struct FileWriter<D, T>::GImpl {
    std::fstream dataFile;
    std::filesystem::path filePath;

    Result startRecording(FileWriter<D, T>& m);
    Result stopRecording();

    Result underlyingStartRecording();
    Result underlyingStopRecording();
};

template<Device D, typename T>
Result FileWriter<D, T>::create() {
    JST_DEBUG("Initializing File Writer module.");
    JST_INIT_IO();

    // Save Config.

    if (config.recording) {
        const auto& res = gimpl->startRecording(*this);

        if (res != Result::SUCCESS) {
            config.recording = false;
            return res;
        }

        return Result::SUCCESS;
    }

    JST_ERROR("Recording was not initiated.");
    return Result::ERROR;
}

template<Device D, typename T>
Result FileWriter<D, T>::destroy() {
    JST_DEBUG("Destroying File Writer module.");

    JST_CHECK(gimpl->stopRecording());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::recording(const bool& recording) {
    if (recording) {
        if (!config.recording) {
            JST_CHECK(gimpl->startRecording(*this));
            config.recording = true;
        }
    } else {
        if (config.recording) {
            config.recording = false;
            JST_CHECK(gimpl->stopRecording());
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::GImpl::startRecording(FileWriter<D, T>& m) {
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

    // Check if parent directory exists
    auto parentPath = filePath.parent_path();
    if (!parentPath.empty() && !std::filesystem::exists(parentPath)) {
        JST_ERROR("Parent directory '{}' does not exist.", parentPath.string());
        return Result::ERROR;
    }

    // Check if file already exists
    if (std::filesystem::exists(filePath) && !m.config.overwrite) {
        JST_ERROR("File '{}' already exists.", filePath.string());
        return Result::ERROR;
    }

    // Open the raw data file.
    dataFile.open(filePath, std::ios::out | std::ios::binary);
    if (!dataFile.is_open()) {
        JST_ERROR("Failed to open file '{}' for writing.", filePath.string());
        return Result::ERROR;
    }

    // Start underlying recording.
    JST_CHECK(underlyingStartRecording());

    // Start Recording.
    JST_INFO("Starting raw binary recording to '{}'.", filePath.string());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::GImpl::stopRecording() {
    // Stop Recording.
    JST_INFO("Stopping recording.");

    // Close the data file.
    if (dataFile.is_open()) {
        dataFile.close();
    }

    // Stop underlying recording.
    JST_CHECK(underlyingStopRecording());

    return Result::SUCCESS;
}

template<Device D, typename T>
void FileWriter<D, T>::info() const {
    JST_DEBUG("  File Format: Raw Binary");
    JST_DEBUG("  Filepath: {}", config.filepath);
    JST_DEBUG("  Name: {}", config.name);
    JST_DEBUG("  Description: {}", config.description);
    JST_DEBUG("  Author: {}", config.author);
    JST_DEBUG("  Sample Rate: {:.2f} MHz", config.sampleRate / JST_MHZ);
    JST_DEBUG("  Center Frequency: {:.2f} MHz", config.centerFrequency / JST_MHZ);
    JST_DEBUG("  Overwrite: {}", config.overwrite);
}

}  // namespace Jetstream
