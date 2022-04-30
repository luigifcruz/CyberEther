#ifndef JETSTREAM_MODULES_MULTIPLY_HH
#define JETSTREAM_MODULES_MULTIPLY_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D, typename T = CF32>
class Multiply : public Module {
 public:
    struct Config {
        U64 size;
    };

    struct Input {
        const Vector<D, T>& factorA;
        const Vector<D, T>& factorB;
    };

    struct Output {
        Vector<D, T> product;
    };

    explicit Multiply(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    constexpr const Vector<D, T>& getProductBuffer() const {
        return this->output.product;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 protected:
    const Result compute(const RuntimeMetadata& meta = {}) final;

 private:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Jetstream

#endif
