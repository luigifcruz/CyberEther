#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/parser.hh"
#include "jetstream/tensor_link.hh"
#include "jetstream/runtime.hh"
#include "jetstream/provider.hh"
#include "jetstream/render/base/window.hh"

namespace Jetstream {

class JETSTREAM_API Module {
 public:
    // Types & Nested Structs

    struct Impl;
    struct Context;
    struct Interface;
    struct Surface;

    struct Config {
        virtual ~Config() = default;

        virtual std::string type() const = 0;

        virtual Result serialize(Parser::Map& data) const = 0;
        virtual Result deserialize(const Parser::Map& data) = 0;
        virtual std::size_t hash() const = 0;
    };

    enum class Taint : U64 {
        CLEAN               = 0 << 0, ///< No taint set, data is in its original state.
        IN_PLACE            = 1 << 0, ///< Module will overwrite input, modifying it directly.
        DISCONTIGUOUS       = 1 << 1, ///< Accepts non-contiguous data buffers for input tensors.
        SURFACE             = 1 << 2, ///< It renders a surface to be drawn on the screen.
        BROWSER_MAIN_THREAD = 1 << 3, ///< Requires main thread for create and destroy when running in the browser.
    };

    // Constructor

    Module(const DeviceType& device,
           const RuntimeType& runtime,
           const ProviderType& provider,
           const std::shared_ptr<Module::Impl>& impl,
           const std::shared_ptr<Module::Context>& context,
           const std::shared_ptr<Module::Config>& stagedConfig,
           const std::shared_ptr<Module::Config>& candidateConfig);

    // Lifecycle

    Result create(const std::string& name,
                  const Module::Config& config,
                  const TensorMap& inputs,
                  const std::shared_ptr<Render::Window>& render = nullptr);
    Result create(const std::string& name,
                  const Parser::Map& config,
                  const TensorMap& inputs,
                  const std::shared_ptr<Render::Window>& render = nullptr);
    Result destroy();

    // Configuration

    Result reconfigure(const Parser::Map& config, const bool& validateOnly = false);
    Result config(Parser::Map& config) const;
    const Module::Config& config() const;

    // Identity

    const std::string& name() const;
    const DeviceType& device() const;
    const RuntimeType& runtime() const;
    const ProviderType& provider() const;
    const Module::Taint& taint() const;

    // I/O

    const TensorMap& inputs() const;
    const TensorMap& outputs() const;
    const std::shared_ptr<Module::Interface>& interface() const;

    // Components

    const std::shared_ptr<Module::Context>& context();
    const std::shared_ptr<Module::Surface>& surface();

    // Implementation access

    template<typename T>
    T* getImpl() {
        return dynamic_cast<T*>(impl.get());
    }

    template<typename T>
    const T* getImpl() const {
        return dynamic_cast<const T*>(impl.get());
    }

 private:
    std::shared_ptr<Impl> impl;
};

inline Module::Taint operator&(Module::Taint lhs, Module::Taint rhs) {
    return static_cast<Module::Taint>(static_cast<U64>(lhs) & static_cast<U64>(rhs));
}

inline Module::Taint operator|(Module::Taint lhs, Module::Taint rhs) {
    return static_cast<Module::Taint>(static_cast<U64>(lhs) | static_cast<U64>(rhs));
}

}  // namespace Jetstream

#ifndef JST_MODULE_TYPE
#define JST_MODULE_TYPE(TYPE) \
    std::string type() const override { \
        static const std::string type = #TYPE; \
        return type; \
    }
#endif  // JST_MODULE_TYPE

#ifndef JST_MODULE_PARAMS
#define JST_MODULE_PARAMS(...) \
    Result serialize(Parser::Map& data) const override { \
        (void)data; \
        FOR_EACH(JST_SERDES_SERIALIZE, __VA_ARGS__) \
        return Result::SUCCESS; \
    } \
    Result deserialize(const Parser::Map& data) override { \
        (void)data; \
        FOR_EACH(JST_SERDES_DESERIALIZE, __VA_ARGS__) \
        return Result::SUCCESS; \
    } \
    std::size_t hash() const override { \
        std::size_t h = 0; \
        FOR_EACH(JST_HASH_FIELD, __VA_ARGS__) \
        return h; \
    }
#endif  // JST_MODULE_PARAMS

#endif  // JETSTREAM_MODULE_HH
