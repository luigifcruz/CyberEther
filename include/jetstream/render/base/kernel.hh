#ifndef JETSTREAM_RENDER_BASE_KERNEL_HH
#define JETSTREAM_RENDER_BASE_KERNEL_HH

#include <utility>
#include <memory>
#include <vector>
#include <unordered_map>
#include <tuple>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class Kernel {
 public:
    struct Config {
        std::tuple<U64, U64, U64> gridSize;
        std::vector<std::shared_ptr<Buffer>> buffers;
        std::unordered_map<Device, std::vector<std::vector<U8>>> kernels;
    };

    explicit Kernel(const Config& config) : config(config) {}
    virtual ~Kernel() = default;

    const Config& getConfig() const {
        return config;
    }

    // TODO: Add an update method.

    template<Device D> 
    static std::shared_ptr<Kernel> Factory(const Config& config) {
        return std::make_shared<KernelImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
