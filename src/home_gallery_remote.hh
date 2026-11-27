#ifndef JETSTREAM_HOME_GALLERY_REMOTE_HH
#define JETSTREAM_HOME_GALLERY_REMOTE_HH

#include "jetstream/home.hh"
#include "jetstream/platform.hh"
#include "jetstream/types.hh"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Jetstream {

class JETSTREAM_API Home::Gallery::Remote {
 public:
    using Flowgraphs = std::vector<Flowgraph>;

    struct Reference {
        std::string owner;
        std::string slug;
    };

    struct Query {
        std::string owner;
        std::string version;
        std::string platform;
        std::string arch;
    };

    struct JETSTREAM_API Source {
        std::function<Platform::Http::Client::Response(const Platform::Http::Client::Request&)> send;
        std::optional<std::filesystem::path> folder;

        static Source Default();
    };

    static constexpr const char* Host = "https://cyberether.org";
    static constexpr const char* Endpoint = "/api/v1/flowgraphs";
    static constexpr const char* PagePath = "/gallery/flowgraphs/";
    static constexpr const char* LinkPrefix = "cyberether://flowgraph/";

    static Query Current();
    static std::string Path(const Query& query);
    static std::optional<std::string> Slug(const std::string& url);
    static std::optional<Reference> Link(const std::string& url);
    static bool ValidOwner(std::string_view owner);
    static std::vector<Section> Sections(const Flowgraphs& flowgraphs);
    static Result Load(Flowgraphs& flowgraphs);
    static Result Fetch(const Query& query, const std::function<bool()>& cancelled, Flowgraphs& flowgraphs);
    static Result LoadFrom(const Source& source, const Query& query, Flowgraphs& flowgraphs);
    static Result FetchFrom(const Source& source,
                            const Query& query,
                            const std::function<bool()>& cancelled,
                            Flowgraphs& flowgraphs);
};

}  // namespace Jetstream

#endif  // JETSTREAM_HOME_GALLERY_REMOTE_HH
