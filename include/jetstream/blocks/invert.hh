#ifndef JETSTREAM_BLOCK_INVERT_BASE_HH
#define JETSTREAM_BLOCK_INVERT_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/invert.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Invert : public Block {
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
        return "invert";
    }

    std::string name() const {
        return "Invert";
    }

    std::string summary() const {
        return "Inverts the complex input signal.";
    }

    std::string description() const {
        return "Inverts the complex-valued input signal by swapping the sign of both real and imaginary components.\n\n"
               "The Invert block performs complex conjugation followed by sign inversion on complex input data. "
               "This operation is particularly useful in signal processing for frequency spectrum manipulation "
               "and mirroring operations.\n\n"
               "Inputs:\n"
               "- buffer: Complex-valued tensor (CF32) containing the signal to invert.\n\n"
               "Outputs:\n"
               "- buffer: Complex-valued tensor with the same shape as the input, but with inverted values.\n\n"
               "Mathematical Operation:\n"
               "- For complex values z = a + bi:\n"
               "  - First computes the complex conjugate: z* = a - bi\n"
               "  - Then negates both components: -z* = -a + bi\n"
               "- Effectively reverses the sign of the real component only\n\n"
               "Key Applications:\n"
               "- Spectrum inversion in signal processing\n"
               "- Upper/lower sideband conversion in SSB modulation\n"
               "- Frequency mirroring operations\n"
               "- Preparing data for specific visualization techniques\n"
               "- Correcting for IQ swapping in some SDR receivers\n\n"
               "Technical Details:\n"
               "- Preserves the magnitude (absolute value) of the complex numbers\n"
               "- Changes the phase by adding π (180 degrees) to the original phase\n"
               "- Computationally efficient with minimal performance impact\n"
               "- Implemented with hardware acceleration where available";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            invert, "invert", {}, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, invert->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(invert->locale()));

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::Invert<D, IT>> invert;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Invert, is_specialized<Jetstream::Invert<D, IT>>::value &&
                         std::is_same<OT, void>::value)

#endif
