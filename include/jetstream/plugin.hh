#ifndef JETSTREAM_PLUGIN_HH
#define JETSTREAM_PLUGIN_HH

#include <stdint.h>
#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

class JETSTREAM_API Plugin {
 public:
    struct TargetInfo {
        std::string path;
        std::string system;
        std::string device;
        std::string arch;
        bool compatible = false;
        bool loaded = false;
    };

    struct Info {
        std::string path;
        std::string name;
        std::string version;
        std::string minimumJetstreamVersion;
        std::string status;
        std::vector<TargetInfo> targets;
        U64 registeredModules = 0;
        U64 registeredBlocks = 0;
        U64 registeredExamples = 0;
        U64 registeredBenchmarks = 0;
    };

    static Result Load(const std::string& path);
    static Result Reload(const std::string& path);
    static std::vector<Info> List();

 private:
    struct Impl;
    static Impl& plugin();
};

}  // namespace Jetstream

#define JETSTREAM_PLUGIN_ABI_SYMBOL "jetstream_plugin_abi"
#define JETSTREAM_PLUGIN_ABI_MAGIC UINT32_C(0x4a535450)
#define JETSTREAM_PLUGIN_ABI_VERSION UINT32_C(1)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JetstreamPluginAbi {
    uint32_t magic;
    uint32_t size;
    uint32_t abi_version;
} JetstreamPluginAbi;

#ifdef __cplusplus
}
#endif

#ifndef JETSTREAM_PLUGIN_API
#if defined(_WIN32)
#define JETSTREAM_PLUGIN_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define JETSTREAM_PLUGIN_API __attribute__((visibility("default")))
#else
#define JETSTREAM_PLUGIN_API
#endif
#endif  // JETSTREAM_PLUGIN_API

#ifdef __cplusplus
#define JETSTREAM_PLUGIN_EXTERN_C extern "C"
#else
#define JETSTREAM_PLUGIN_EXTERN_C
#endif

#define JST_REGISTER_PLUGIN() \
    JETSTREAM_PLUGIN_EXTERN_C JETSTREAM_PLUGIN_API const JetstreamPluginAbi jetstream_plugin_abi = { \
        JETSTREAM_PLUGIN_ABI_MAGIC, \
        (uint32_t)sizeof(JetstreamPluginAbi), \
        JETSTREAM_PLUGIN_ABI_VERSION, \
    }

#endif  // JETSTREAM_PLUGIN_HH
