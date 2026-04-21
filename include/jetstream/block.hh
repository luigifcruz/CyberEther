#ifndef JETSTREAM_BLOCK_HH
#define JETSTREAM_BLOCK_HH

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "jetstream/runtime.hh"
#include "jetstream/scheduler.hh"
#include "jetstream/types.hh"
#include "jetstream/parser.hh"
#include "jetstream/tensor_link.hh"
#include "jetstream/module.hh"
#include "jetstream/render/base.hh"

namespace Jetstream {

class Instance;
class Compositor;
class Flowgraph;
class Scheduler;

class JETSTREAM_API Block {
 public:
    // Types & Nested Structs

    struct Impl;
    struct Interface;

    struct Config {
        virtual ~Config() = default;

        virtual std::string type() const = 0;
        virtual std::string title() const = 0;
        virtual std::string summary() const = 0;
        virtual std::string description() const = 0;

        virtual Result serialize(Parser::Map& data) const = 0;
        virtual Result deserialize(const Parser::Map& data) = 0;
        virtual std::size_t hash() const = 0;
    };

    enum class State : U8 {
        None = 0,
        Creating,
        Created,
        Incomplete,
        Errored,
        Destroying,
        Destroyed,
    };

    // Constructor

    Block(const std::shared_ptr<Block::Impl>& impl,
          const std::shared_ptr<Block::Config>& stagedConfig,
          const std::shared_ptr<Block::Config>& candidateConfig);

    // Lifecycle

    Result create(const std::string& name,
                  const DeviceType& device,
                  const RuntimeType& runtime,
                  const ProviderType& provider,
                  const Parser::Map& config,
                  const TensorMap& inputs,
                  const std::shared_ptr<Instance>& instance,
                  const std::shared_ptr<Render::Window>& render,
                  const std::shared_ptr<Scheduler>& scheduler);
    Result destroy();

    // Configuration

    Result reconfigure(const Parser::Map& config);
    Result config(Parser::Map& config) const;
    const Block::Config& config() const;

    // Identity

    const std::string& name() const;
    const DeviceType& device() const;
    const RuntimeType& runtime() const;
    const ProviderType& provider() const;
    const State& state() const;
    const std::string& diagnostic() const;

    // I/O

    const TensorMap& inputs() const;
    const TensorMap& outputs() const;
    const std::shared_ptr<Block::Interface>& interface() const;

    // Components

    const std::vector<std::shared_ptr<Module::Surface>>& surfaces() const;
    const std::vector<std::string>& modules() const;

 private:
    std::shared_ptr<Impl> impl;
};

JETSTREAM_API const char* GetBlockStateName(const Block::State& state);
JETSTREAM_API const char* GetBlockStatePrettyName(const Block::State& state);

inline std::ostream& operator<<(std::ostream& os, const Block::State& state) {
    return os << GetBlockStatePrettyName(state);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::Block::State> : jst::fmt::ostream_formatter {};

#ifndef JST_BLOCK_TYPE
#define JST_BLOCK_TYPE(TYPE) \
    std::string type() const override { \
        static const std::string type = #TYPE; \
        return type; \
    }
#endif  // JST_BLOCK_TYPE

#ifndef JST_BLOCK_PARAMS
#define JST_BLOCK_PARAMS(...) \
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
#endif  // JST_BLOCK_PARAMS

#ifndef JST_BLOCK_DESCRIPTION
#define JST_BLOCK_DESCRIPTION(TITLE, SUMMARY, DESCRIPTION) \
    std::string title() const override { \
        static const std::string title = TITLE; \
        return title; \
    } \
    std::string summary() const override { \
        static const std::string summary = SUMMARY; \
        return summary; \
    } \
    std::string description() const override { \
        static const std::string description = DESCRIPTION; \
        return description; \
    }
#endif  // JST_BLOCK_DESCRIPTION

#endif  // JETSTREAM_BLOCK_HH
