#ifndef JETSTREAM_BLOCK_FILE_READER_BASE_HH
#define JETSTREAM_BLOCK_FILE_READER_BASE_HH

#include <regex>

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/file_reader.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FileReader : public Block {
 public:
    // Configuration

    struct Config {
        FileFormatType fileFormat = FileFormatType::Raw;
        std::string filepath = "";
        bool playing = false;
        bool loop = true;
        std::string shape = "[8192]";

        JST_SERDES(fileFormat, filepath, playing, loop, shape);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "file-reader";
    }

    std::string name() const {
        return "File Reader";
    }

    std::string summary() const {
        return "Reads a signal from a file.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Reads a signal from a file.";
    }

    // Constructor

    Result create() {
        std::vector<U64> parsedShape;

        std::regex re(R"(\d+)");
        std::sregex_iterator next(config.shape.begin(), config.shape.end(), re);
        std::sregex_iterator end;

        while (next != end) {
            std::smatch match = *next;
            parsedShape.push_back(std::stoull(match.str()));
            next++;
        }

        JST_CHECK(instance().addModule(
            file_reader, "file_reader", {
                .fileFormat = config.fileFormat,
                .filepath = config.filepath,
                .playing = config.playing,
                .loop = config.loop,
                .shape = parsedShape,
            }, {
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, file_reader->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(file_reader->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        if (file_reader->playing()) {
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
            JST_CHECK_NOTIFY(Platform::PickFile(config.filepath));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Output Shape");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##Shape", &config.shape);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Loop");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::Checkbox("##Loop", &config.loop);

        if (file_reader->playing()) {
            ImGui::EndDisabled();
        }

        // File info display
        if (!config.filepath.empty() && file_reader) {
            U64 fileSize = file_reader->getFileSize();
            U64 currentPos = file_reader->getCurrentPosition();

            if (fileSize > 0) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Position");
                ImGui::TableSetColumnIndex(1);
                F32 progress = static_cast<F32>(currentPos) / static_cast<F32>(fileSize);
                F32 fileSizeGb = static_cast<F32>(fileSize) / (1024.0f * 1024.0f * 1024.0f);
                const auto progressOverlay = jst::fmt::format("{:.1f}% ({:.2f} GB)", progress * 100.0f, fileSizeGb);
                ImGui::SetNextItemWidth(-1);
                ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f), progressOverlay.c_str());
            }
        }

        // Play/Pause controls
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        if (!file_reader->playing()) {
            if (ImGui::Button("Start Playing", ImVec2(fullWidth, 0))) {
                config.playing = true;

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Starting playback..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        } else {
            if (ImGui::Button("Stop Playing", ImVec2(fullWidth, 0))) {
                config.playing = false;

                JST_DISPATCH_ASYNC([&](){
                    JST_CHECK_NOTIFY(file_reader->playing(false));
                    instance().reloadBlock(locale());
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::FileReader<D, IT>> file_reader;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(FileReader, is_specialized<Jetstream::FileReader<D, IT>>::value &&
                             std::is_same<OT, void>::value)

#endif
