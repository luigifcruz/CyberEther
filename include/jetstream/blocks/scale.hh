#ifndef JETSTREAM_BLOCK_SCALE_BASE_HH
#define JETSTREAM_BLOCK_SCALE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/scale.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Scale : public Block {
 public:
    // Configuration

    struct Config {
        Range<IT> range = {-1.0, +1.0};

        JST_SERDES(range);
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
        return "scale";
    }

    std::string name() const {
        return "Scale";
    }

    std::string summary() const {
        return "Scales input data by a factor.";
    }

    std::string description() const {
        return "Multiplies each data point in the input by a specified scaling factor, adjusting its magnitude.\n\n"
               "The Scale block performs element-wise multiplication of the input tensor by a constant scaling factor. "
               "This operation is useful for adjusting signal levels, normalizing data, or compensating for gain "
               "differences in a processing chain.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor of any shape and supported data type (F32, CF32).\n"
               "  - For real-valued data (F32), each value is multiplied by the scale factor.\n"
               "  - For complex data (CF32), both real and imaginary parts are multiplied by the scale factor.\n\n"
               "Outputs:\n"
               "- buffer: Output tensor with the same shape and data type as the input, but with values scaled.\n\n"
               "Configuration Parameters:\n"
               "- constant: The scaling factor (default: 1.0) that multiplies each element.\n\n"
               "Mathematical Operation:\n"
               "- For real values: output[i] = input[i] * constant\n"
               "- For complex values: output[i] = input[i] * constant (both real and imaginary parts are scaled)\n\n"
               "Common Applications:\n"
               "- Signal level adjustment\n"
               "- Normalization of data values\n"
               "- Gain compensation\n"
               "- Attenuation or amplification\n"
               "- Preparing data for visualization\n\n"
               "Performance Notes:\n"
               "- This operation is computationally efficient and has minimal impact on performance\n"
               "- Implemented with hardware acceleration where available";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            scale, "scale", {
                .range = config.range,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, scale->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(scale->locale()));

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
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Scale<D, IT>> scale;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Scale, is_specialized<Jetstream::Scale<D, IT>>::value &&
                        std::is_same<OT, void>::value)

#endif
