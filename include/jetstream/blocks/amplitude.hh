#ifndef JETSTREAM_BLOCK_AMPLITUDE_BASE_HH
#define JETSTREAM_BLOCK_AMPLITUDE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/amplitude.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Amplitude : public Block {
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
        return "amplitude";
    }

    std::string name() const {
        return "Amplitude";
    }

    std::string summary() const {
        return "Calculates the amplitude of a signal.";
    }

    std::string description() const {
        return "Calculates the amplitude (magnitude) of a complex signal, converting complex values to their absolute values.\n\n"
               "The Amplitude block computes the magnitude of each complex value in the input tensor, effectively "
               "calculating the distance from the origin in the complex plane for each sample. This operation "
               "is fundamental in signal processing for analyzing signal strength and envelope detection.\n\n"
               "Inputs:\n"
               "- buffer: Complex-valued tensor (CF32) containing the signal to analyze.\n"
               "  - Each complex value is represented as a pair of real and imaginary components.\n\n"
               "Outputs:\n"
               "- buffer: Real-valued tensor (F32) containing the amplitude values.\n"
               "  - The output has the same shape as the input, but with each complex value converted to its magnitude.\n\n"
               "Mathematical Operation:\n"
               "- For complex value z = a + bi:\n"
               "  - Amplitude = |z| = sqrt(a² + b²)\n"
               "  - Where a is the real component and b is the imaginary component\n\n"
               "Key Applications:\n"
               "- Signal strength measurement\n"
               "- Envelope detection\n"
               "- AM demodulation\n"
               "- Power spectrum calculation (when applied after FFT)\n"
               "- Signal thresholding and detection\n\n"
               "Technical Details:\n"
               "- Uses optimized calculation methods based on the target hardware\n"
               "- For CPU targets, uses sqrt(real² + imag²) with potential SIMD acceleration\n"
               "- For GPU targets, uses hardware-accelerated vector operations\n"
               "- Handles both scalar values and multi-dimensional tensors\n\n"
               "Usage Notes:\n"
               "- Often used after an FFT block to convert complex frequency components to magnitudes\n"
               "- Commonly followed by logarithmic conversion for visualization purposes\n"
               "- For power calculations, the output can be squared to get power instead of amplitude";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            amplitude, "amplitude", {}, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, amplitude->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(amplitude->locale()));

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::Amplitude<D, IT, OT>> amplitude;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Amplitude, is_specialized<Jetstream::Amplitude<D, IT, OT>>::value &&
                            !std::is_same<OT, void>::value)

#endif
