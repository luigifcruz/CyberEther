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
};

template<Device D, typename T>
Result FileWriter<D, T>::create() {
    JST_DEBUG("Initializing File Writer module.");
    JST_INIT_IO();

    if (config.recording) {
        const auto& res = startRecording();

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

    JST_CHECK(stopRecording());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::recording(const bool& recording) {
    if (recording) {
        if (!config.recording) {
            JST_CHECK(startRecording());
            config.recording = true;
        }
    } else {
        if (config.recording) {
            config.recording = false;
            JST_CHECK(stopRecording());
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::startRecording() {
    std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();

    // Check file format type.

    if (config.fileFormat != FileFormatType::SigMF) {
        JST_ERROR("File format '{}' is not supported.", config.fileFormat);
        return Result::ERROR;
    }

    // Check if the provided filepath is valid.

    if (config.filepath.empty()) {
        JST_ERROR("File path is empty.");
        return Result::ERROR;
    }

    if (!std::regex_match(config.filepath, std::regex("^[a-zA-Z0-9_./-]*$"))) {
        JST_ERROR("File path '{}' contains invalid characters.", config.filepath);
        return Result::ERROR;
    }

    if (!std::regex_match(config.name, std::regex("^[a-zA-Z0-9_.-]*$"))) {
        JST_ERROR("Name '{}' contains invalid characters.", config.name);
        return Result::ERROR;
    }

    gimpl->dirname = std::filesystem::path(config.filepath);

    if (!std::filesystem::is_directory(gimpl->dirname)) {
        JST_ERROR("File path '{}' is not a directory.", gimpl->dirname.string());
        return Result::ERROR;
    }

    if (!std::filesystem::exists(gimpl->dirname)) {
        JST_ERROR("Directory '{}' does not exist.", gimpl->dirname.string());
        return Result::ERROR;
    }

    // Generate basename with datetime.

    if (config.name.empty()) {
        gimpl->basename = jst::fmt::format("{:%Y-%m-%dT%H-%M-%S}Z.sigmf", t);
    } else {
        gimpl->basename = jst::fmt::format("{:%Y-%m-%dT%H-%M-%S}Z_{}.sigmf", t, config.name);
    }

    const auto& filepath = gimpl->dirname / gimpl->basename.stem();
    if (std::filesystem::exists(filepath) && !config.overwrite) {
        JST_ERROR("Folder '{}' already exists.", filepath.string());
        return Result::ERROR;
    }

    // Create the temporary SigMF folder.

    gimpl->tmpFolderPath = gimpl->dirname / gimpl->basename.stem();
    JST_TRACE("[FILE_WRITER] Using temporary folder: '{}'", gimpl->tmpFolderPath.string());

    if (std::filesystem::exists(gimpl->tmpFolderPath)) {
        if (!config.overwrite) {
            JST_ERROR("Temporary folder '{}' already exists.", gimpl->tmpFolderPath.string());
            return Result::ERROR;
        }

        if (!std::filesystem::remove_all(gimpl->tmpFolderPath)) {
            JST_ERROR("Failed to remove temporary folder '{}'.", gimpl->tmpFolderPath.string());
            return Result::ERROR;
        }
    }

    if (!std::filesystem::create_directory(gimpl->tmpFolderPath)) {
        JST_ERROR("Failed to create temporary folder '{}'.", gimpl->tmpFolderPath.string());
        return Result::ERROR;
    }

    // Create the SigMF files.

    gimpl->dataFilePath = gimpl->tmpFolderPath / gimpl->basename.replace_extension(".sigmf-data");
    gimpl->metaFilePath = gimpl->tmpFolderPath / gimpl->basename.replace_extension(".sigmf-meta");

    // Open the SigMF files.

    gimpl->dataFile.open(gimpl->dataFilePath, std::ios::out | std::ios::binary);
    gimpl->metaFile.open(gimpl->metaFilePath, std::ios::out);

    // Write the SigMF core metadata.

    // TODO: Replace with JSON library.
    gimpl->metaFile << "{\n";
    gimpl->metaFile << "  \"global\": {\n";
    gimpl->metaFile << "    \"core:author\": \"" << config.author << "\",\n";
    gimpl->metaFile << "    \"core:datatype\": \"cf32_le\",\n";
    gimpl->metaFile << "    \"core:description\": \"" << config.description << "\",\n";
    gimpl->metaFile << "    \"core:recorder\": \"CyberEther v" << JETSTREAM_VERSION_STR << "\",\n";
    gimpl->metaFile << "    \"core:sample_rate\": " << jst::fmt::format("{}", config.sampleRate) << ",\n";
    gimpl->metaFile << "    \"core:version\": \"v1.0.0\"\n";
    gimpl->metaFile << "  },\n";
    gimpl->metaFile << "  \"captures\": [\n";
    gimpl->metaFile << "    {\n";
    gimpl->metaFile << "      \"core:frequency\": " << jst::fmt::format("{}", config.centerFrequency) << ",\n";
    gimpl->metaFile << "      \"core:sample_start\": 0,\n";
    gimpl->metaFile << "      \"core:datetime\": \"" << jst::fmt::format("{:%Y-%m-%dT%H:%M:%SZ}", t) << "\"\n";
    gimpl->metaFile << "    }\n";
    gimpl->metaFile << "  ],\n";
    gimpl->metaFile << "  \"annotations\": []\n";
    gimpl->metaFile << "}\n";

    // Start underlying recording.

    JST_CHECK(underlyingStartRecording());

    // Start Recording.

    JST_INFO("Starting recording.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::stopRecording() {
    // Stop Recording.

    JST_INFO("Stopping recording.");

    // Close the SigMF files.

    gimpl->dataFile.close();
    gimpl->metaFile.close();

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
