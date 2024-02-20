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
        // TODO: Add decent block description describing internals and I/O.
        return "Inverts the complex-valued input signal. Useful for ploting the spectrum of a signal.";
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
