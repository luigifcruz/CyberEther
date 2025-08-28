#include <regex>
#include <filesystem>
#include <fstream>
#include <chrono>

#include "jetstream/modules/file_writer.hh"

namespace Jetstream {

template<Device D, typename T>
struct FileWriter<D, T>::GImpl {
    std::fstream dataFile;
    std::fstream metaFile;

    std::filesystem::path dataFilePath;
    std::filesystem::path metaFilePath;
    std::filesystem::path tmpFolderPath;

    std::filesystem::path dirname;
    std::filesystem::path basename;

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
    std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();

    // Check file format type.

    if (m.config.fileFormat != FileFormatType::SigMF) {
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

    if (!std::regex_match(m.config.name, std::regex("^[a-zA-Z0-9_.-]*$"))) {
        JST_ERROR("Name '{}' contains invalid characters.", m.config.name);
        return Result::ERROR;
    }

    dirname = std::filesystem::path(m.config.filepath);

    if (!std::filesystem::is_directory(dirname)) {
        JST_ERROR("File path '{}' is not a directory.", dirname.string());
        return Result::ERROR;
    }

    if (!std::filesystem::exists(dirname)) {
        JST_ERROR("Directory '{}' does not exist.", dirname.string());
        return Result::ERROR;
    }

    // Generate basename with datetime.

    if (m.config.name.empty()) {
        basename = jst::fmt::format("{:%Y-%m-%dT%H-%M-%S}Z.sigmf", t);
    } else {
        basename = jst::fmt::format("{:%Y-%m-%dT%H-%M-%S}Z_{}.sigmf", t, m.config.name);
    }

    const auto& filepath = dirname / basename.stem();
    if (std::filesystem::exists(filepath) && !m.config.overwrite) {
        JST_ERROR("Folder '{}' already exists.", filepath.string());
        return Result::ERROR;
    }

    // Create the temporary SigMF folder.

    tmpFolderPath = dirname / basename.stem();
    JST_TRACE("[FILE_WRITER] Using temporary folder: '{}'", tmpFolderPath.string());

    if (std::filesystem::exists(tmpFolderPath)) {
        if (!m.config.overwrite) {
            JST_ERROR("Temporary folder '{}' already exists.", tmpFolderPath.string());
            return Result::ERROR;
        }

        if (!std::filesystem::remove_all(tmpFolderPath)) {
            JST_ERROR("Failed to remove temporary folder '{}'.", tmpFolderPath.string());
            return Result::ERROR;
        }
    }

    if (!std::filesystem::create_directory(tmpFolderPath)) {
        JST_ERROR("Failed to create temporary folder '{}'.", tmpFolderPath.string());
        return Result::ERROR;
    }

    // Create the SigMF files.

    dataFilePath = tmpFolderPath / basename.replace_extension(".sigmf-data");
    metaFilePath = tmpFolderPath / basename.replace_extension(".sigmf-meta");

    // Open the SigMF files.

    dataFile.open(dataFilePath, std::ios::out | std::ios::binary);
    metaFile.open(metaFilePath, std::ios::out);

    // Write the SigMF core metadata.

    // TODO: Replace with JSON library.
    metaFile << "{\n";
    metaFile << "  \"global\": {\n";
    metaFile << "    \"core:author\": \"" << m.config.author << "\",\n";
    metaFile << "    \"core:datatype\": \"cf32_le\",\n";
    metaFile << "    \"core:description\": \"" << m.config.description << "\",\n";
    metaFile << "    \"core:recorder\": \"CyberEther v" << JETSTREAM_VERSION_STR << "\",\n";
    metaFile << "    \"core:sample_rate\": " << jst::fmt::format("{}", m.config.sampleRate) << ",\n";
    metaFile << "    \"core:version\": \"v1.0.0\"\n";
    metaFile << "  },\n";
    metaFile << "  \"captures\": [\n";
    metaFile << "    {\n";
    metaFile << "      \"core:frequency\": " << jst::fmt::format("{}", m.config.centerFrequency) << ",\n";
    metaFile << "      \"core:sample_start\": 0,\n";
    metaFile << "      \"core:datetime\": \"" << jst::fmt::format("{:%Y-%m-%dT%H:%M:%SZ}", t) << "\"\n";
    metaFile << "    }\n";
    metaFile << "  ],\n";
    metaFile << "  \"annotations\": []\n";
    metaFile << "}\n";

    // Start underlying recording.

    JST_CHECK(underlyingStartRecording());

    // Start Recording.

    JST_INFO("Starting recording.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::GImpl::stopRecording() {
    // Stop Recording.

    JST_INFO("Stopping recording.");

    // Close the SigMF files.

    dataFile.close();
    metaFile.close();

    // Stop underlying recording.

    JST_CHECK(underlyingStopRecording());

    // Pack temporary SigMF files into a single file.

    // TODO: Implement.

    // Remove the temporary SigMF folder.

    // TODO: Implement.

    return Result::SUCCESS;
}

template<Device D, typename T>
void FileWriter<D, T>::info() const {
    JST_DEBUG("  File Format: {}", config.fileFormat);
    JST_DEBUG("  Name: {}", config.name);
    JST_DEBUG("  Filepath: {}", config.filepath);
    JST_DEBUG("  Description: {}", config.description);
    JST_DEBUG("  Author: {}", config.author);
    JST_DEBUG("  Sample Rate: {:.2f} MHz", config.sampleRate / JST_MHZ);
    JST_DEBUG("  Center Frequency: {:.2f} MHz", config.centerFrequency / JST_MHZ);
    JST_DEBUG("  Overwrite: {}", config.overwrite);
}

}  // namespace Jetstream
