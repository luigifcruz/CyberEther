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
        // TODO: Add decent block description describing internals and I/O.
        return "Adjusts the gain of the input signal to a constant level.";
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
        if (agc) {
            JST_CHECK(instance().eraseModule(agc->locale()));
        }

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
