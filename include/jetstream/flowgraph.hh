#ifndef JETSTREAM_FLOWGRAPH_HH
#define JETSTREAM_FLOWGRAPH_HH

#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "jetstream/scheduler.hh"
#include "jetstream/parser.hh"
#include "jetstream/tensor_link.hh"
#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/block.hh"
#include "jetstream/compositor.hh"
#include "jetstream/render/base/window.hh"

namespace Jetstream {

class JETSTREAM_API Flowgraph {
 public:
    struct Config {
        SchedulerType scheduler = SchedulerType::SYNCHRONOUS;
    };
    struct Impl;

    Flowgraph();
    ~Flowgraph();

    Result create(const Config& config,
                  const std::shared_ptr<Instance>& instance,
                  const std::shared_ptr<Render::Window>& render,
                  const std::shared_ptr<Compositor>& compositor);
    Result start();
    Result stop();
    Result destroy();

    const std::string& title() const;
    const std::string& summary() const;
    const std::string& author() const;
    const std::string& license() const;
    const std::string& description() const;
    const std::string& path() const;

    Result setTitle(const std::string& title);
    Result setSummary(const std::string& summary);
    Result setAuthor(const std::string& author);
    Result setLicense(const std::string& license);
    Result setDescription(const std::string& description);

    Result blockCreate(const std::string name,
                       const Block::Config& config,
                       const TensorMap& inputs,
                       const DeviceType& device = DeviceType::CPU,
                       const RuntimeType& runtime = RuntimeType::NATIVE,
                       const ProviderType& provider = "generic");
    Result blockCreate(const std::string name,
                       const std::string type,
                       const Parser::Map& config,
                       const TensorMap& inputs,
                       const DeviceType& device = DeviceType::CPU,
                       const RuntimeType& runtime = RuntimeType::NATIVE,
                       const ProviderType& provider = "generic");
    Result blockDestroy(const std::string name,
                        bool propagate = true);
    Result blockConnect(const std::string blockName,
                        const std::string inputPort,
                        const std::string sourceBlock,
                        const std::string sourcePort);
    Result blockDisconnect(const std::string blockName,
                           const std::string inputPort);
    Result blockReconfigure(const std::string name,
                            const Parser::Map& config);
    Result blockRecreate(const std::string name,
                         const Parser::Map& config);
    Result blockRecreate(const std::string name,
                         const Parser::Map& config,
                         const DeviceType& device,
                         const RuntimeType& runtime,
                         const ProviderType& provider);
    Result blockConfig(const std::string name,
                       Parser::Map& config) const;
    const std::unordered_map<std::string, std::shared_ptr<Block>>& blockList() const;

    Result importFromFile(const std::string& path);
    Result importFromBlob(const std::vector<char>& blob);

    Result exportToFile(const std::string& path);
    Result exportToBlob(std::vector<char>& blob);

    Result compute();
    Result present();

    bool hasPersistentMeta(const std::string& key, const std::string& block = {}) const;

    template<typename T>
    Result getPersistentMeta(const std::string& key, T& data, const std::string& block = {}) const {
        Parser::Map stored;
        JST_CHECK(getPersistentMeta(key, stored, block));

        if (stored.empty()) {
            return Result::SUCCESS;
        }

        Parser::Map encoded;
        encoded[key] = stored;
        return Parser::Deserialize(encoded, key, data);
    }

    Result getPersistentMeta(const std::string& key, Parser::Map& data, const std::string& block = {}) const;

    template<typename T>
    bool tryGetPersistentMeta(const std::string& key, T& data, const std::string& block = {}) const {
        if (!hasPersistentMeta(key, block)) {
            return false;
        }

        return getPersistentMeta(key, data, block) == Result::SUCCESS;
    }

    template<typename T>
    Result setPersistentMeta(const std::string& key, const T& data, const std::string& block = {}) {
        Parser::Map encoded;
        JST_CHECK(Parser::Serialize(encoded, key, data));

        if (!encoded.contains(key) || encoded.at(key).type() != typeid(Parser::Map)) {
            JST_ERROR("[FLOWGRAPH] Persistent meta '{}' must serialize to a map.", key);
            return Result::ERROR;
        }

        return setPersistentMeta(key, std::any_cast<const Parser::Map&>(encoded.at(key)), block);
    }

    Result setPersistentMeta(const std::string& key, const Parser::Map& data, const std::string& block = {});
    Result clearPersistentMeta(const std::string& key, const std::string& block = {});
    Result clearAllPersistentMeta();

    bool hasVolatileMeta(const std::string& key,
                         U64 timestamp = std::numeric_limits<U64>::min()) const;

    template<typename T>
    Result getVolatileMeta(const std::string& key,
                           T& data,
                           U64 timestamp = std::numeric_limits<U64>::min()) const {
        Parser::Map stored;
        JST_CHECK(getVolatileMeta(key, stored, timestamp));

        if (stored.empty()) {
            return Result::SUCCESS;
        }

        Parser::Map encoded;
        encoded[key] = stored;
        return Parser::Deserialize(encoded, key, data);
    }

    Result getVolatileMeta(const std::string& key,
                           Parser::Map& data,
                           U64 timestamp = std::numeric_limits<U64>::min()) const;

    template<typename T>
    bool tryGetVolatileMeta(const std::string& key,
                            T& data,
                            U64 timestamp = std::numeric_limits<U64>::min()) const {
        if (!hasVolatileMeta(key, timestamp)) {
            return false;
        }

        return getVolatileMeta(key, data, timestamp) == Result::SUCCESS;
    }

    template<typename T>
    Result setVolatileMeta(const std::string& key,
                           const T& data,
                           U64 start = std::numeric_limits<U64>::min(),
                           U64 end = std::numeric_limits<U64>::max()) {
        Parser::Map encoded;
        JST_CHECK(Parser::Serialize(encoded, key, data));

        if (!encoded.contains(key) || encoded.at(key).type() != typeid(Parser::Map)) {
            JST_ERROR("[FLOWGRAPH] Volatile meta '{}' must serialize to a map.", key);
            return Result::ERROR;
        }

        return setVolatileMeta(key, std::any_cast<const Parser::Map&>(encoded.at(key)), start, end);
    }

    Result setVolatileMeta(const std::string& key,
                           const Parser::Map& data,
                           U64 start = std::numeric_limits<U64>::min(),
                           U64 end = std::numeric_limits<U64>::max());
    Result clearVolatileMeta(const std::string& key);
    Result clearAllVolatileMeta();

 private:
    std::shared_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_FLOWGRAPH_HH
