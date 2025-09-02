#ifndef JETSTREAM_BLOCK_FILE_WRITER_BASE_HH
#define JETSTREAM_BLOCK_FILE_WRITER_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/file_writer.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FileWriter : public Block {
 public:
    // Configuration

    struct Config {
        FileFormatType fileFormat = FileFormatType::Raw;
        std::string filepath = "";
        std::string name = "";
        std::string description = "";
        std::string author = "CyberEther User";
        F32 sampleRate = 0.0f;
        F32 centerFrequency = 0.0f;
        bool overwrite = false;
        bool recording = false;

        JST_SERDES(fileFormat, filepath, name, description, author, sampleRate, centerFrequency, overwrite, recording);
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
        // TODO: Add decent block description describing internals and I/O.
        return "Writes a signal to a file.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            file_writer, "file_writer", {
                .fileFormat = config.fileFormat,
                .filepath = config.filepath,
                .name = config.name,
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
        ImGui::TextUnformatted("File");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##FilePath", &config.filepath);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        const F32 fullWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Pick File Path", ImVec2(fullWidth, 0))) {
            JST_CHECK_NOTIFY(Platform::SaveFile(config.filepath));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Name");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##FileName", &config.name);

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
