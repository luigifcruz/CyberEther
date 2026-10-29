#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_GALLERY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_GALLERY_HH

#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "jetstream/types.hh"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

class JETSTREAM_API Gallery {
 public:
    struct Query {
        std::string owner;
        std::string version;
        std::string platform;
        std::string arch;
    };

    struct Source {
        std::function<Platform::Http::Client::Response(const Platform::Http::Client::Request&)> send;
        std::optional<std::filesystem::path> folder;

        static Source Default();
    };

    static constexpr const char* Host = "https://cyberether.org";
    static constexpr const char* Endpoint = "/api/v1/flowgraphs";
    static constexpr const char* Owner = "cyberether";

    static Query Current();
    static std::string Path(const Query& query);
    static Result Load(std::vector<Registry::FlowgraphRegistration>& flowgraphs);
    static Result Fetch(const Query& query,
                        const std::function<bool()>& cancelled,
                        std::vector<Registry::FlowgraphRegistration>& flowgraphs);
    static Result LoadFrom(const Source& source,
                           const Query& query,
                           std::vector<Registry::FlowgraphRegistration>& flowgraphs);
    static Result FetchFrom(const Source& source,
                            const Query& query,
                            const std::function<bool()>& cancelled,
                            std::vector<Registry::FlowgraphRegistration>& flowgraphs);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_GALLERY_HH
