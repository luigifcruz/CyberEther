#ifndef JETSTREAM_BLOCK_SPECTRUM_ENGINE_BASE_HH
#define JETSTREAM_BLOCK_SPECTRUM_ENGINE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/macros.hh"
#include "jetstream/modules/tensor_modifier.hh"
#include "jetstream/modules/window.hh"
#include "jetstream/modules/multiply.hh"
#include "jetstream/modules/fft.hh"
#include "jetstream/modules/invert.hh"
#include "jetstream/modules/agc.hh"
#include "jetstream/modules/amplitude.hh"
#include "jetstream/modules/scale.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class SpectrumEngine : public Block {
 public:
    // Configuration

    struct Config {
        U64 axis = 1;
        bool enableAGC = false;
        bool enableScale = false;
        Range<OT> range = {-120.0, 0.0};

        JST_SERDES(axis, enableAGC, enableScale, range);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "spectrum-engine";
    }

    std::string name() const {
        return "Spectrum Engine";
    }

    std::string summary() const {
        return "Computes the spectrum using windowing, FFT, and optional AGC/scaling.";
    }

    std::string description() const {
        return "The Spectrum Engine block computes the frequency spectrum of the input signal through "
               "a configurable processing chain. It applies a window function, performs FFT, and "
               "optionally applies AGC and scaling to produce the final spectrum output.\n\n"

               "## Parameters\n"
               "- **Axis**: The axis along which to compute the spectrum (determines window size).\n"
               "- **Enable AGC**: Whether to apply automatic gain control after the FFT.\n"
               "- **Enable Scale**: Whether to apply scaling to the final output.\n"
               "- **Scale Range**: The minimum and maximum values for scaling (in dBFS).\n\n"

               "## Processing Chain:\n"
               "Input → Window → Multiply → FFT → [AGC] → Amplitude → [Scale] → Output\n"
               "1. Window module generates a window function sized to the specified input axis.\n"
               "2. Multiply module applies the window to the input signal.\n"
               "3. FFT module computes the forward Fourier transform.\n"
               "4. Optional AGC module applies automatic gain control.\n"
               "5. Amplitude module computes the magnitude of the complex spectrum.\n"
               "6. Optional Scale module applies the specified scaling range.\n\n"

               "## Useful For:\n"
               "- Spectral analysis and visualization\n"
               "- Power spectral density computation\n"
               "- Frequency domain signal processing\n\n"

               "## Examples:\n"
               "- Time-domain spectrum analysis:\n"
               "  Config: Axis=1, Enable AGC=true, Enable Scale=true\n"
               "  Input: CF32[8192, 1024] → Output: OT[8192, 1024]";
    }

    // Constructor

    Result create() {
        // Get the window size from the input shape at the specified axis

        U64 windowSize = 0;

        JST_CHECK(instance().addModule(
            modifier, "modifier", {
                .callback = [&](auto&) {
                    if (config.axis >= input.buffer.rank()) {
                        JST_ERROR("Axis {} is out of bounds for input tensor rank {}.", config.axis, input.buffer.rank());
                        return Result::ERROR;
                    }

                    windowSize = input.buffer.shape()[config.axis];

                    return Result::SUCCESS;
                }
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            window, "window", {
                .size = windowSize,
            }, {},
            locale()
        ));

        JST_CHECK(instance().addModule(
            invert, "invert", {}, {
                .buffer = window->getOutputWindow(),
            },
            locale()
        ))

        JST_CHECK(instance().addModule(
            multiply, "multiply", {}, {
                .factorA = modifier->getOutputBuffer(),
                .factorB = invert->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            fft, "fft", {
                .forward = true,
            }, {
                .buffer = multiply->getOutputProduct(),
            },
            locale()
        ));

        auto fftOutput = fft->getOutputBuffer();

        // Optional AGC
        if (config.enableAGC) {
            JST_CHECK(instance().addModule(
                agc, "agc", {}, {
                    .buffer = fftOutput,
                },
                locale()
            ));
            fftOutput = agc->getOutputBuffer();
        }

        JST_CHECK(instance().addModule(
            amplitude, "amplitude", {}, {
                .buffer = fftOutput,
            },
            locale()
        ));

        auto amplitudeOutput = amplitude->getOutputBuffer();

        // Optional Scale
        if (config.enableScale) {
            JST_CHECK(instance().addModule(
                scale, "scale", {
                    .range = config.range,
                }, {
                    .buffer = amplitudeOutput,
                },
                locale()
            ));
            JST_CHECK(Block::LinkOutput("buffer", output.buffer, scale->getOutputBuffer()));
        } else {
            JST_CHECK(Block::LinkOutput("buffer", output.buffer, amplitudeOutput));
        }

        return Result::SUCCESS;
    }

    Result destroy() {
        if (scale) {
            JST_CHECK(instance().eraseModule(scale->locale()));
        }

        if (amplitude) {
            JST_CHECK(instance().eraseModule(amplitude->locale()));
        }

        if (agc) {
            JST_CHECK(instance().eraseModule(agc->locale()));
        }

        if (fft) {
            JST_CHECK(instance().eraseModule(fft->locale()));
        }

        if (multiply) {
            JST_CHECK(instance().eraseModule(multiply->locale()));
        }

        if (invert) {
            JST_CHECK(instance().eraseModule(invert->locale()));
        }

        if (window) {
            JST_CHECK(instance().eraseModule(window->locale()));
        }

        if (modifier) {
            JST_CHECK(instance().eraseModule(modifier->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Axis");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 axis = config.axis;
        if (ImGui::InputFloat("##axis", &axis, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (axis >= 0 && axis < input.buffer.rank()) {
                config.axis = static_cast<U64>(axis);

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Enable AGC");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Checkbox("##enableAGC", &config.enableAGC)) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Enable Scale");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Checkbox("##enableScale", &config.enableScale)) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        if (config.enableScale && scale) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Range (dBFS)");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            auto [min, max] = scale->range();
            if (ImGui::DragFloatRange2("##ScaleRange", &min, &max,
                        1, -300, 0, "Min: %.0f", "Max: %.0f")) {
                config.range = scale->range({min, max});
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> modifier;
    std::shared_ptr<Jetstream::Window<D, IT>> window;
    std::shared_ptr<Jetstream::Invert<D, IT>> invert;
    std::shared_ptr<Jetstream::Multiply<D, IT>> multiply;
    std::shared_ptr<Jetstream::FFT<D, IT, IT>> fft;
    std::shared_ptr<Jetstream::AGC<D, IT>> agc;
    std::shared_ptr<Jetstream::Amplitude<D, IT, OT>> amplitude;
    std::shared_ptr<Jetstream::Scale<D, OT>> scale;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(SpectrumEngine, is_specialized<Jetstream::Window<D, IT>>::value &&
                                 is_specialized<Jetstream::Invert<D, IT>>::value &&
                                 is_specialized<Jetstream::Multiply<D, IT>>::value &&
                                 is_specialized<Jetstream::FFT<D, IT, IT>>::value &&
                                 is_specialized<Jetstream::AGC<D, IT>>::value &&
                                 is_specialized<Jetstream::Amplitude<D, IT, OT>>::value &&
                                 is_specialized<Jetstream::Scale<D, OT>>::value &&
                                 !std::is_same<OT, void>::value)

#endif
