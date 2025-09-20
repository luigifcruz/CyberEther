#ifndef JETSTREAM_BLOCK_SIGNAL_GENERATOR_BASE_HH
#define JETSTREAM_BLOCK_SIGNAL_GENERATOR_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/signal_generator.hh"
#include "jetstream/render/macros.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class SignalGenerator : public Block {
 public:
    // Configuration

    struct Config {
        SignalType signalType = SignalType::Cosine;
        F64 sampleRate = 2000000.0;
        F64 frequency = 1000000.0;
        F64 amplitude = 1.0;
        F64 phase = 0.0;
        F64 dcOffset = 0.0;
        F64 noiseVariance = 1.0;
        F64 chirpStartFreq = 1000000.0;
        F64 chirpEndFreq = 2000000.0;
        F64 chirpDuration = 1.0;
        U64 bufferSize = 8192;

        JST_SERDES(signalType, sampleRate, frequency, amplitude, phase, dcOffset,
                   noiseVariance, chirpStartFreq, chirpEndFreq, chirpDuration, bufferSize);
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
        return "signal_generator";
    }

    std::string name() const {
        return "Signal Generator";
    }

    std::string summary() const {
        return "Generates various types of signals including sine, square, triangle, sawtooth, noise, and chirp.";
    }

    std::string description() const {
        return "The Signal Generator block creates synthetic signals for testing and simulation purposes. "
               "It supports multiple waveform types with configurable parameters for frequency, amplitude, "
               "phase, and other signal-specific properties.\n\n"

               "## Arguments:\n"
               "- **Signal Type**: The type of signal to generate (Sine, Cosine, Square, Triangle, Sawtooth, Noise, DC, Chirp).\n"
               "- **Sample Rate**: The sampling frequency in MHz.\n"
               "- **Frequency**: The fundamental frequency of the generated signal in MHz.\n"
               "- **Amplitude**: The amplitude scaling factor for the signal.\n"
               "- **Phase**: Phase offset in radians.\n"
               "- **DC Offset**: DC bias added to the signal.\n"
               "- **Noise Variance**: Variance of Gaussian noise (for noise signal type).\n"
               "- **Chirp Start/End Freq**: Start and end frequencies for chirp signals in MHz.\n"
               "- **Chirp Duration**: Duration of one chirp sweep in seconds.\n"
               "- **Buffer Size**: Number of samples to generate per processing cycle.\n\n"

               "## Useful For:\n"
               "- Creating test signals for system verification and debugging.\n"
               "- Generating reference waveforms for signal processing algorithms.\n"
               "- Producing noise sources for statistical analysis.\n"
               "- Creating chirp signals for frequency response measurements.\n\n"

               "## Examples:\n"
               "- Sine wave generation:\n"
               "  Config: SignalType=Sine, Frequency=100MHz, SampleRate=2MHz, BufferSize=1024\n"
               "  Output: CF32[1024] or F32[1024] → 100MHz sine wave\n"
               "- Complex noise generation:\n"
               "  Config: SignalType=Noise, NoiseVariance=0.1, BufferSize=8192\n"
               "  Output: CF32[8192] → Complex Gaussian noise\n\n"

               "## Implementation:\n"
               "SignalGenerator Module → Output\n"
               "1. The signal generator module computes the requested waveform using optimized algorithms.\n"
               "2. For complex outputs (CF32), generates both I and Q components appropriately.\n"
               "3. For real outputs (F32), generates only the real component.\n"
               "4. Maintains phase continuity across buffer boundaries for continuous signals.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            generator, "generator", {
                .signalType = config.signalType,
                .sampleRate = config.sampleRate,
                .frequency = config.frequency,
                .amplitude = config.amplitude,
                .phase = config.phase,
                .dcOffset = config.dcOffset,
                .noiseVariance = config.noiseVariance,
                .chirpStartFreq = config.chirpStartFreq,
                .chirpEndFreq = config.chirpEndFreq,
                .chirpDuration = config.chirpDuration,
                .bufferSize = config.bufferSize,
            }, {},
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, generator->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (generator) {
            JST_CHECK(instance().eraseModule(generator->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        // Signal Type Selection
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Signal Type");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##signalType", config.signalType.string().c_str())) {
            for (const auto& [key, value] : config.signalType.rmap()) {
                bool isSelected = (config.signalType == key);
                if (ImGui::Selectable(value.c_str(), isSelected)) {
                    config.signalType = key;
                    reloadBlock();
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        // Sample Rate
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / JST_MHZ;
        if (ImGui::InputFloat("##sampleRate", &sampleRate, 0.001f, 0.01f, "%.3f MHz")) {
            if (sampleRate > 0) {
                config.sampleRate = sampleRate * JST_MHZ;
                JST_MODULE_UPDATE(generator, setSampleRate(config.sampleRate));
            }
        }

        // Frequency (for most signal types)
        if (config.signalType != SignalType::Noise &&
            config.signalType != SignalType::DC &&
            config.signalType != SignalType::Chirp) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Frequency");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            F32 frequency = config.frequency / JST_MHZ;
            if (ImGui::InputFloat("##frequency", &frequency, 0.0001f, 0.001f, "%.3f MHz")) {
                config.frequency = frequency * JST_MHZ;
                JST_MODULE_UPDATE(generator, setFrequency(config.frequency));
            }
        }

        // Chirp Frequency
        if (config.signalType == SignalType::Chirp) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Start Freq");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            F32 chirpStartFreq = config.chirpStartFreq / JST_MHZ;
            if (ImGui::InputFloat("##chirpStartFreq", &chirpStartFreq, 0.0001f, 0.001f, "%.3f MHz")) {
                config.chirpStartFreq = chirpStartFreq * JST_MHZ;
                JST_MODULE_UPDATE(generator, setChirpStartFreq(config.chirpStartFreq));
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("End Freq");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            F32 chirpEndFreq = config.chirpEndFreq / JST_MHZ;
            if (ImGui::InputFloat("##chirpEndFreq", &chirpEndFreq, 0.0001f, 0.001f, "%.3f MHz")) {
                config.chirpEndFreq = chirpEndFreq * JST_MHZ;
                JST_MODULE_UPDATE(generator, setChirpEndFreq(config.chirpEndFreq));
            }
        }

        // Amplitude
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Amplitude");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 amplitude = config.amplitude;
        if (ImGui::InputFloat("##amplitude", &amplitude, 0.1f, 1.0f, "%.3f")) {
            if (amplitude >= 0) {
                config.amplitude = amplitude;
                JST_MODULE_UPDATE(generator, setAmplitude(config.amplitude));
            }
        }

        // Phase (for non-noise, non-DC signals)
        if (config.signalType != SignalType::Noise && config.signalType != SignalType::DC) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Phase (rad)");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            F32 phase = config.phase;
            if (ImGui::InputFloat("##phase", &phase, 0.1f, 1.0f, "%.3f")) {
                config.phase = phase;
                JST_MODULE_UPDATE(generator, setPhase(config.phase));
            }
        }

        // DC Offset
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("DC Offset");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 dcOffset = config.dcOffset;
        if (ImGui::InputFloat("##dcOffset", &dcOffset, 0.1f, 1.0f, "%.3f")) {
            config.dcOffset = dcOffset;
            JST_MODULE_UPDATE(generator, setDcOffset(config.dcOffset));
        }

        // Noise-specific parameters
        if (config.signalType == SignalType::Noise) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Noise Variance");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            F32 noiseVariance = config.noiseVariance;
            if (ImGui::InputFloat("##noiseVariance", &noiseVariance, 0.01f, 0.1f, "%.4f")) {
                if (noiseVariance > 0) {
                    config.noiseVariance = noiseVariance;
                    JST_MODULE_UPDATE(generator, setNoiseVariance(config.noiseVariance));
                }
            }
        }

        // Chirp-specific parameters
        if (config.signalType == SignalType::Chirp) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Duration (s)");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            F32 chirpDuration = config.chirpDuration;
            if (ImGui::InputFloat("##chirpDuration", &chirpDuration, 0.1f, 1.0f, "%.2f")) {
                if (chirpDuration > 0) {
                    config.chirpDuration = chirpDuration;
                    JST_MODULE_UPDATE(generator, setChirpDuration(config.chirpDuration));
                }
            }
        }

        // Buffer Size
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Buffer Size");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 bufferSize = static_cast<F32>(config.bufferSize);
        if (ImGui::InputFloat("##bufferSize", &bufferSize, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (bufferSize > 0) {
                config.bufferSize = static_cast<U64>(bufferSize);
                reloadBlock();
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::SignalGenerator<D, IT>> generator;

    void reloadBlock() {
        JST_DISPATCH_ASYNC([&](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
            JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
        });
    }

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(SignalGenerator, is_specialized<Jetstream::SignalGenerator<D, IT>>::value &&
                                  std::is_same<OT, void>::value)

#endif
