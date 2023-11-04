#ifndef JETSTREAM_BUNDLES_SCALE_BASE_HH
#define JETSTREAM_BUNDLES_SCALE_BASE_HH

#include "jetstream/bundle.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/scale.hh"

namespace Jetstream::Bundles {

template<Device D, typename T = F32>
class Scale : public Bundle {
 public:
   // Configuration

    struct Config {
        Range<T> range = {-1.0, +1.0};

        JST_SERDES(
            JST_SERDES_VAL("range", range);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "scale-view";
    }

    std::string_view prettyName() const {
        return "Scale";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Scale, D, T>(
            scale, "ui", {
                .range = config.range,
            }, {
                .buffer = input.buffer,
            },
            locale().id
        ));

        JST_CHECK(this->linkOutput("buffer", output.buffer, scale->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().removeModule("ui", locale().id));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Range (dBFS)");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        auto [min, max] = scale->range();
        if (ImGui::DragFloatRange2("##ScaleRange", &min, &max,
                    1, -300, 0, "Min: %.0f", "Max: %.0f")) {
            scale->range({min, max});
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Scale<D, T>> scale;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Bundles

#endif
