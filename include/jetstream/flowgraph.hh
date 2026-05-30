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
    class Metadata;
    class Environment;
    class View;

    struct Config {
        SchedulerType scheduler = SchedulerType::SYNCHRONOUS;
    };
    struct Impl;

    Flowgraph();
    ~Flowgraph();

    Flowgraph(const Flowgraph&) = delete;
    Flowgraph& operator=(const Flowgraph&) = delete;

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

    Result importFromFile(const std::string& path);
    Result importFromBlob(const std::vector<char>& blob);

    Result exportToFile(const std::string& path);
    Result exportToBlob(std::vector<char>& blob);

    Result compute();
    Result present();

    Metadata& metadata();
    const Metadata& metadata() const;

    Environment& environment();
    const Environment& environment() const;

    View& view();
    const View& view() const;

 private:
    std::shared_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_FLOWGRAPH_HH
