#ifndef JETSTREAM_BLOCK_SPECTROSCOPE_BASE_HH
#define JETSTREAM_BLOCK_SPECTROSCOPE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/lineplot.hh"
#include "jetstream/modules/waterfall.hh"
#include "jetstream/modules/spectrogram.hh"
#include "jetstream/modules/scale.hh"
#include "jetstream/modules/amplitude.hh"
#include "jetstream/modules/fft.hh"
#include "jetstream/modules/multiply.hh"
#include "jetstream/modules/agc.hh"
#include "jetstream/modules/invert.hh"
#include "jetstream/modules/window.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Spectroscope : public Block {
 public:
    // Configuration

    struct Config {
        bool spectrogram = false;
        bool lineplot = true;
        bool waterfall = true;
        Range<OT> range = {-1.0, +1.0};
        Size2D<U64> viewSize = {512, 512};

        JST_SERDES(spectrogram, lineplot, waterfall, range, viewSize);
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
        Tensor<D, OT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, OT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "spectroscope";
    }

    std::string name() const {
        return "Spectroscope";
    }

    std::string summary() const {
        return "Visualization for time-domain signals.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "High-level visualization for time-domain signals.";
    }

    // Constructor

    Result create() {
        U64 signalMaxRank = input.buffer.rank() - 1;
        const U64 signalSize = input.buffer.shape()[signalMaxRank];

        numberOfRows = ((config.spectrogram) ? 1 : 0) + 
                       ((config.lineplot) ? 1 : 0) + 
                       ((config.waterfall) ? 1 : 0);

        auto individualViewSize = config.viewSize;
        if (numberOfRows > 0) {
            individualViewSize.height /= numberOfRows;
        }

        JST_CHECK(instance().addModule(
            window, "window", {
                .size = signalSize,
            }, {},
            locale()
        ));

        JST_CHECK(instance().addModule(
            invert, "invert", {}, {
                .buffer = window->getOutputWindow(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            multiply, "multiply", {}, {
                .factorA = invert->getOutputBuffer(),
                .factorB = input.buffer,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            agc, "agc", {}, {
                .buffer = multiply->getOutputProduct(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            fft, "fft", {
                .forward = true,
            }, {
                .buffer = agc->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            amplitude, "amplitude", {}, {
                .buffer = fft->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            scale, "scale", {
                .range = config.range,
            }, {
                .buffer = amplitude->getOutputBuffer(),
            },
            locale()
        ));

        if (config.spectrogram) {
            JST_CHECK(instance().addModule(
                spectrogram, "spectrogram", {
                    .height = 512,
                    .viewSize = individualViewSize,
                }, {
                    .buffer = scale->getOutputBuffer(),
                },
                locale()
            ));
        }

        if (config.lineplot) {
            JST_CHECK(instance().addModule(
                lineplot, "lineplot", {
                    .numberOfVerticalLines = numberOfVerticalLines,
                    .numberOfHorizontalLines = numberOfHorizontalLines,
                    .viewSize = individualViewSize,
                }, {
                    .buffer = scale->getOutputBuffer(),
                },
                locale()
            ));
        }

        if (config.waterfall) {
            JST_CHECK(instance().addModule(
                waterfall, "waterfall", {
                    .zoom = 1.0,
                    .offset = 0,
                    .height = 512,
                    .viewSize = individualViewSize,
                }, {
                    .buffer = scale->getOutputBuffer(),
                },
                locale()
            ));
        }

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, scale->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (spectrogram) {
            JST_CHECK(instance().eraseModule(spectrogram->locale()));
        }
    
        if (lineplot) {
            JST_CHECK(instance().eraseModule(lineplot->locale()));
        }

        if (waterfall) {
            JST_CHECK(instance().eraseModule(waterfall->locale()));
        }

        JST_CHECK(instance().eraseModule(scale->locale()));
        JST_CHECK(instance().eraseModule(amplitude->locale()));
        JST_CHECK(instance().eraseModule(fft->locale()));
        JST_CHECK(instance().eraseModule(agc->locale()));
        JST_CHECK(instance().eraseModule(multiply->locale()));
        JST_CHECK(instance().eraseModule(invert->locale()));
        JST_CHECK(instance().eraseModule(window->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
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

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Views");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);

        auto spectrogram = config.spectrogram;
        if (ImGui::Checkbox("SPC", &spectrogram)) {
            config.spectrogram = spectrogram;

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }
        ImGui::SameLine();
        auto lineplot = config.lineplot;
        if (ImGui::Checkbox("LPT", &lineplot)) {
            config.lineplot = lineplot;

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }
        ImGui::SameLine();
        auto waterfall = config.waterfall;
        if (ImGui::Checkbox("WTF", &waterfall)) {
            config.waterfall = waterfall;

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }    
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

    void drawPreview(const F32& maxWidth) {
        if (spectrogram) {
            const auto& size = spectrogram->viewSize();
            const auto& ratio = size.ratio();
            const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
            ImGui::Image(spectrogram->getTexture().raw(), ImVec2(width, width/ratio));
        }

        if (lineplot) {
            const auto& size = lineplot->viewSize();
            const auto& ratio = size.ratio();
            const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
            ImGui::Image(lineplot->getTexture().raw(), ImVec2(width, width/ratio));
        }

        if (waterfall) {
            const auto& size = waterfall->viewSize();
            const auto& ratio = size.ratio();
            const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
            ImGui::Image(waterfall->getTexture().raw(), ImVec2(width, width/ratio));
        }
    }

    constexpr bool shouldDrawPreview() const {
        return spectrogram || lineplot || waterfall;
    }

    void drawView() {
        auto [width, height] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;

        const auto paddingHeight = ImGui::GetStyle().FramePadding.y * 2.0f * numberOfRows;
        const U64 availableWidth = width * scale.x;
        const U64 availableHeight = std::max(0.0f, (height * scale.y) - paddingHeight);

        const auto blockSize = Size2D<U64>{
            availableWidth,
            availableHeight / numberOfRows,
        };

        // TODO: Add support for zoom and translation.

        if (spectrogram) {
            auto [width, height] = spectrogram->viewSize(blockSize);
            ImGui::Image(spectrogram->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));
        }

        if (lineplot) {
            auto [width, height] = lineplot->viewSize(blockSize);
            ImGui::Image(lineplot->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));
        }
        
        if (waterfall) {
            auto [width, height] = waterfall->viewSize(blockSize);
            ImGui::Image(waterfall->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));
        }
    }

    constexpr bool shouldDrawView() const {
        return spectrogram || lineplot || waterfall;
    }

 private:
    U64 numberOfRows = 0;
    U64 numberOfVerticalLines = 20;
    U64 numberOfHorizontalLines = 5;

    std::shared_ptr<Jetstream::Window<D, IT>> window;
    std::shared_ptr<Jetstream::Invert<D, IT>> invert;
    std::shared_ptr<Jetstream::Multiply<D, IT>> multiply;
    std::shared_ptr<Jetstream::AGC<D, IT>> agc;
    std::shared_ptr<Jetstream::FFT<D, IT, IT>> fft;
    std::shared_ptr<Jetstream::Amplitude<D, IT, OT>> amplitude;
    std::shared_ptr<Jetstream::Scale<D, OT>> scale;

    std::shared_ptr<Jetstream::Spectrogram<D, OT>> spectrogram;
    std::shared_ptr<Jetstream::Lineplot<D, OT>> lineplot;
    std::shared_ptr<Jetstream::Waterfall<D, OT>> waterfall;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Spectroscope, is_specialized<Jetstream::Spectrogram<D, OT>>::value &&
                               is_specialized<Jetstream::Waterfall<D, OT>>::value &&
                               is_specialized<Jetstream::Lineplot<D, OT>>::value &&
                               is_specialized<Jetstream::Scale<D, OT>>::value &&
                               is_specialized<Jetstream::Amplitude<D, IT, OT>>::value &&
                               is_specialized<Jetstream::FFT<D, IT, IT>>::value &&
                               is_specialized<Jetstream::AGC<D, IT>>::value &&
                               is_specialized<Jetstream::Multiply<D, IT>>::value &&
                               is_specialized<Jetstream::Invert<D, IT>>::value &&
                               is_specialized<Jetstream::Window<D, IT>>::value)

#endif
