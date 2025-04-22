#ifndef JETSTREAM_BLOCK_AUDIO_BASE_HH
#define JETSTREAM_BLOCK_AUDIO_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/audio.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Audio : public Block {
 public:
    using AudioModule = Jetstream::Audio<D, IT>;

    // Configuration

    struct Config {
        std::string deviceName = "Default";
        F32 inSampleRate = 48e3;
        F32 outSampleRate = 48e3;

        JST_SERDES(deviceName, inSampleRate, outSampleRate);
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
        return "audio";
    }

    std::string name() const {
        return "Audio";
    }

    std::string summary() const {
        return "Plays input on the speaker.";
    }

    std::string description() const {
        return "Plays audio data through the system's audio output device (speakers or headphones).\n\n"
               "The Audio block takes audio samples as input and routes them to the computer's audio output. "
               "It automatically handles resampling to match the system's audio device sample rate and manages "
               "the audio playback buffer for smooth, glitch-free audio output.\n\n"
               "Inputs:\n"
               "- buffer: Real-valued tensor (F32 type) containing audio samples.\n"
               "  - For mono audio: 1D tensor with shape [samples]\n"
               "  - For stereo audio: 2D tensor with shape [2, samples]\n\n"
               "Configuration Parameters:\n"
               "- None (uses system default audio device)\n\n"
               "Technical Details:\n"
               "- Built on the cross-platform Miniaudio library\n"
               "- Automatically resamples input to match device sample rate\n"
               "- Uses ring buffer to handle timing differences between processing and playback\n"
               "- Supports both mono and stereo audio format\n"
               "- Low-latency playback when system supports it\n\n"
               "Common Applications:\n"
               "- Real-time audio monitoring\n"
               "- Playback of processed audio signals\n"
               "- SDR audio output\n"
               "- Audio testing and verification\n\n"
               "Tips for Use:\n"
               "- Input values should be normalized to the [-1.0, 1.0] range\n"
               "- For best results, ensure your processing chain can keep up with real-time audio rates\n"
               "- Check system volume settings if no audio is heard";
    }

    // Constructor

    Result create() {
        // Populate internal state.

        availableDeviceList = AudioModule::ListAvailableDevices();

        // Starting audio module.

        JST_CHECK(instance().addModule(
            audio, "audio", {
                .deviceName = config.deviceName,
                .inSampleRate = config.inSampleRate,
                .outSampleRate = config.outSampleRate,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));
        JST_CHECK(Block::LinkOutput("buffer", output.buffer, audio->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(audio->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 inSampleRate = config.inSampleRate / 1e6f;
        if (ImGui::InputFloat("##in-sample-rate", &inSampleRate, 0.1f, 0.2f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.inSampleRate = inSampleRate * 1e6;

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Device List");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        static const char* noDeviceMessage = "No device found";
        if (ImGui::BeginCombo("##DeviceList", availableDeviceList.empty() ? noDeviceMessage : audio->getDeviceName().c_str())) {
            for (const auto& device : availableDeviceList) {
                bool isSelected = (config.deviceName == device);
                if (ImGui::Selectable(device.c_str(), isSelected)) {
                    config.deviceName = device;

                    JST_DISPATCH_ASYNC([&](){
                        ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                        JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                    });
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();

                }
            }
            ImGui::EndCombo();
        }
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        const F32 fullWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Reload Device List", ImVec2(fullWidth, 0))) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading device list..." });
                availableDeviceList = AudioModule::ListAvailableDevices();
                JST_CHECK_NOTIFY(Result::SUCCESS);
            });
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<AudioModule> audio;

    typename AudioModule::DeviceList availableDeviceList;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Audio, is_specialized<Jetstream::Audio<D, IT>>::value &&
                        std::is_same<OT, void>::value)

#endif
