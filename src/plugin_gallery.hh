#ifndef JETSTREAM_PLUGIN_GALLERY_HH
#define JETSTREAM_PLUGIN_GALLERY_HH

#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/types.hh"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Jetstream {

class JETSTREAM_API Plugin::Gallery {
 public:
    struct Entry {
        std::string id;
        std::string title;
        std::string summary;
        std::string description;
        std::string author;
        std::string license;
        std::string version;
        std::string url;
        U64 size = 0;
        std::string digest;
        std::vector<std::string> blocks;

        bool available() const {
            return !version.empty() && !url.empty();
        }
    };

    using Entries = std::vector<Entry>;

    struct Query {
        std::string owner;
        std::string version;
        std::string platform;
        std::string arch;
        std::string build;
        std::vector<std::string> ids;
    };

    struct JETSTREAM_API Source {
        std::function<Platform::Http::Client::Response(const Platform::Http::Client::Request&)> send;

        static Source Default();
    };

    static constexpr const char* HostVariable = "CYBERETHER_PLUGIN_GALLERY_HOST";
    static constexpr const char* Endpoint = "/api/v1/plugins";
    static constexpr const char* PagePath = "/gallery/plugins/";

    static bool ValidId(std::string_view id);
    static std::string Host();
    static Query Current(std::vector<std::string> ids = {});
    static std::string Path(const Query& query);

    static Result Fetch(const Source& source,
                        const Query& query,
                        const std::function<bool()>& cancelled,
                        Entries& entries);
    static Result Download(const Source& source,
                           const Entry& entry,
                           const std::filesystem::path& destination,
                           const std::function<bool()>& cancelled);
    static Result Install(const Source& source,
                          const std::string& id,
                          const std::function<bool()>& cancelled = {});
    static Result Update(const Source& source,
                         std::vector<std::string>& updated,
                         const std::function<bool()>& cancelled = {});
};

}  // namespace Jetstream

#endif  // JETSTREAM_PLUGIN_GALLERY_HH
