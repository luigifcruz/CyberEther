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
        // TODO: Add decent block description describing internals and I/O.
        return "Calculates the amplitude of a complex signal.";
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
        if (amplitude) {
            JST_CHECK(instance().eraseModule(amplitude->locale()));
        }

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
