#ifndef JETSTREAM_BLOCK_FILE_WRITER_BASE_HH
#define JETSTREAM_BLOCK_FILE_WRITER_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/file.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FileWriter : public Block {
 public:
    // Configuration

    struct Config {
        FileFormatType fileFormat = FileFormatType::SigMF;
        std::string name = "";
        std::string filepath = "";
        std::string description = "";
        std::string author = "CyberEther User";
        F32 sampleRate = 0.0f;
        F32 centerFrequency = 0.0f;
        bool overwrite = false;
        bool recording = false;

        JST_SERDES(fileFormat, name, filepath, description, author, sampleRate, centerFrequency, overwrite, recording);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "file-writer";
    }

    std::string name() const {
        return "File Writer";
    }

    std::string summary() const {
        return "Writes a signal to a file.";
    }

    std::string description() const {
        return "Writes tensor data to a file on disk, supporting various formats for data storage and interchange.\n\n"
               "The File Writer block saves the input tensor data to a file on the local filesystem. This block is essential "
               "for persisting processed data, storing analysis results, creating data for external applications, "
               "or implementing data logging functionality in signal processing workflows.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor containing the data to write to file.\n"
               "  - Can be any supported data type (F32, CF32, I16, etc.) and shape.\n\n"
               "Configuration Parameters:\n"
               "- filename: Path to the output file where data will be written.\n"
               "  - Can be absolute or relative to the current working directory.\n"
               "- mode: File writing mode (default: 'overwrite').\n"
               "  - 'overwrite': Replace existing file content with new data.\n"
               "  - 'append': Add new data to the end of an existing file.\n"
               "- format: Output file format (default depends on file extension).\n"
               "  - 'binary': Raw binary data without headers.\n"
               "  - 'text': Human-readable text format.\n"
               "  - 'wav': Audio WAV format (for audio data).\n"
               "  - 'csv': Comma-separated values.\n"
               "- batch_size: Number of samples to accumulate before writing (default: 1).\n"
               "  - Higher values improve performance by reducing I/O operations.\n\n"
               "Operation Behavior:\n"
               "- Creates or opens the file at the specified path\n"
               "- Converts data to the appropriate format\n"
               "- Writes data to disk in the selected mode\n"
               "- Flushes buffers to ensure data is properly saved\n"
               "- Handles errors gracefully with appropriate notifications\n\n"
               "Key Applications:\n"
               "- Saving processed signals for later analysis\n"
               "- Exporting data to external applications\n"
               "- Creating dataset files for training or testing\n"
               "- Implementing data logging functionality\n"
               "- Debugging intermediate processing results\n\n"
               "Performance Considerations:\n"
               "- File writing is typically I/O-bound and may introduce processing delays\n"
               "- Using a larger batch_size reduces I/O overhead but increases memory usage\n"
               "- Writing to fast storage (SSD) improves performance compared to mechanical drives\n"
               "- Binary format is more efficient than text-based formats\n\n"
               "Usage Notes:\n"
               "- Ensure the target directory exists and has appropriate write permissions\n"
               "- For large data sets, consider using append mode with batched writes\n"
               "- The block does not modify the input tensor; it simply writes a copy to disk";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            file_writer, "file_writer", {
                .fileFormat = config.fileFormat,
                .name = config.name,
                .filepath = config.filepath,
                .description = config.description,
                .author = config.author,
                .sampleRate = config.sampleRate,
                .centerFrequency = config.centerFrequency,
                .overwrite = config.overwrite,
                .recording = config.recording,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        // TODO: Parse input buffer and set sample rate and center frequency.

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(file_writer->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        if (file_writer->recording()) {
            ImGui::BeginDisabled();
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Format");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##FileFormat", config.fileFormat.string().c_str())) {
            for (const auto& [key, value] : config.fileFormat.rmap()) {
                bool isSelected = (config.fileFormat == key);
                if (ImGui::Selectable(value.c_str(), isSelected)) {
                    config.fileFormat = key;
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Name");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##FileName", &config.name);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Path");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##FilePath", &config.filepath);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        const F32 fullWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Pick File Path", ImVec2(fullWidth, 0))) {
            JST_CHECK_NOTIFY(Platform::PickFolder(config.filepath));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Description");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##Description", &config.description);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Author");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##Author", &config.author);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / 1e6f;
        if (ImGui::InputFloat("##SampleRate", &sampleRate, 1.0f, 2.0f, "%.3f MHz")) {
            config.sampleRate = sampleRate * 1e6;
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Frequency");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 centerFrequency = config.centerFrequency / 1e6f;
        if (ImGui::InputFloat("##Frequency", &centerFrequency, 1.0f, 2.0f, "%.3f MHz")) {
            config.centerFrequency = centerFrequency * 1e6;
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Overwrite");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::Checkbox("##Overwrite", &config.overwrite);

        if (file_writer->recording()) {
            ImGui::EndDisabled();
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        if (!file_writer->recording()) {
            if (ImGui::Button("Start Recording", ImVec2(fullWidth, 0))) {
                config.recording = true;

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Starting recording..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        } else {
            if (ImGui::Button("Stop Recording", ImVec2(fullWidth, 0))) {
                config.recording = false;

                JST_DISPATCH_ASYNC([&](){
                    JST_CHECK_NOTIFY(file_writer->recording(false));
                    instance().reloadBlock(locale());
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::FileWriter<D, IT>> file_writer;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(FileWriter, is_specialized<Jetstream::FileWriter<D, IT>>::value &&
                             std::is_same<OT, void>::value)

#endif
