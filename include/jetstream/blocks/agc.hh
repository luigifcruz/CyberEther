#ifndef JETSTREAM_BLOCK_AGC_BASE_HH
#define JETSTREAM_BLOCK_AGC_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/agc.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class AGC : public Block {
 public:
    // Configuration

    struct Config {
        JST_SERDES();
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
        return "agc";
    }

    std::string name() const {
        return "AGC";
    }

    std::string summary() const {
        return "Stabilizes signal amplitude.";
    }

    std::string description() const {
        return "Automatically adjusts the gain of an input signal to maintain a relatively constant output level.\n\n"
               "The AGC (Automatic Gain Control) block dynamically adjusts amplification or attenuation applied to "
               "the input signal to maintain a consistent output power level. It's particularly useful for signals "
               "with varying amplitudes, such as radio communications where signal strength can fluctuate.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor containing the signal with varying amplitude.\n"
               "  - Can be real-valued (F32) or complex-valued (CF32).\n\n"
               "Outputs:\n"
               "- buffer: Output tensor with stabilized amplitude.\n"
               "  - Same type and shape as the input tensor, but with normalized amplitude.\n\n"
               "Operation Details:\n"
               "- Continuously monitors the input signal level\n"
               "- Computes a running average of the signal power\n"
               "- Adjusts a variable gain factor to normalize the output level\n"
               "- Uses attack and decay time constants to control adaptation speed\n"
               "- Applies soft limiting to prevent sudden amplitude spikes\n\n"
               "Key Applications:\n"
               "- Radio receivers for compensating signal fading\n"
               "- Audio processing to normalize volume levels\n"
               "- Communication systems for maintaining optimal signal levels\n"
               "- Preprocessing for demodulation stages\n"
               "- Systems handling signals with unpredictable strength\n\n"
               "Performance Considerations:\n"
               "- AGC introduces some delay due to the averaging window\n"
               "- Very fast signals may experience some transient distortion\n"
               "- More effective for slowly varying signal envelopes\n"
               "- Impact on signal phase is minimized by applying gain uniformly";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            agc, "agc", {}, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, agc->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(agc->locale()));

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::AGC<D, IT>> agc;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(AGC, is_specialized<Jetstream::AGC<D, IT>>::value &&
                      std::is_same<OT, void>::value)

#endif
