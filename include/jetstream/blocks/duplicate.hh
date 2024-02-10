#ifndef JETSTREAM_BLOCK_DUPLICATE_BASE_HH
#define JETSTREAM_BLOCK_DUPLICATE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/duplicate.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Duplicate : public Block {
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
        return "duplicate";
    }

    std::string name() const {
        return "Duplicate";
    }

    std::string summary() const {
        return "Copies the input signal.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Duplicates the input signal by copying it to the output buffer. "
               "This block also converts non-contiguous input buffers to contiguous output buffers.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            duplicate, "duplicate", {}, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, duplicate->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(duplicate->locale()));

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::Duplicate<D, IT>> duplicate;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Duplicate, is_specialized<Jetstream::Duplicate<D, IT>>::value &&
                            std::is_same<OT, void>::value)

#endif
