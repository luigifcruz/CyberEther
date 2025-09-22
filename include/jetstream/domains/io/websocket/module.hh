#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Websocket : public Module::Config {
    std::string url = "ws://localhost:8765";
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(websocket);
    JST_MODULE_PARAMS(url, dataType, numberOfBatches, numberOfTimeSamples,
                      bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_HH
