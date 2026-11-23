#ifndef JETSTREAM_PLUGIN_HH
#define JETSTREAM_PLUGIN_HH

#include <filesystem>
#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

class JETSTREAM_API Plugin {
 public:
    class Gallery;

    enum class Status {
        Loaded,
        Restart,
        Incompatible,
        Waiting,
        Failed,
    };

    struct Info {
        std::string id;
        std::string title;
        std::string summary;
        std::string description;
        std::string author;
        std::string license;
        std::string version;
        std::string cyberether;
        std::string path;
        std::string source;
        std::string pending;
        bool temporary = false;
        bool gallery = false;
        Status status = Status::Failed;
        std::vector<std::string> blocks;
        std::vector<std::string> examples;
    };

    static Result Install(const std::string& path, std::string& id);
    static Result Remove(const std::string& id);
    static Result LoadInstalled();
    static Result Load(const std::string& path);
    static Result Reload(const std::string& id);
    static std::vector<Info> List();
    static std::filesystem::path Path(const std::string& id);
    static std::string Owner(const std::string& block, const std::string& provider);
    static bool IsCepPath(const std::string& path);

 private:
    struct Impl;
    static Impl& plugin();
};

}  // namespace Jetstream

#endif  // JETSTREAM_PLUGIN_HH
