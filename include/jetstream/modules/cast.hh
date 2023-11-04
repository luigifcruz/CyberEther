#ifndef JETSTREAM_MODULES_CAST_HH
#define JETSTREAM_MODULES_CAST_HH

#include <algorithm>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename IT = F32, typename OT = I16>
class Cast : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        F32 scaler = 0.0f;

        JST_SERDES(
            JST_SERDES_VAL("scaler", scaler);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, OT> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, OT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "cast";
    }

    std::string_view prettyName() const {
        return "Cast";
    }

    void summary() const final; 

    // Constructor

    Result create();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

    JST_DEFINE_IO();
};

}  // namespace Jetstream

#endif
